"""
Generate the CLAP "global" conditioning dataset used to train the DiT.

The trained CLAP model (clap.py / train_clap.py) maps BOTH modalities into a
single shared 512-d space:

    audio: raw waveform [B, 1, T]  --encode_audio(full raw-audio tower)-->  [B, 512]
    text : [B, 256, 1024] T5 tokens --encode_text(frozen-T5 head)-------->  [B, 512]

NOTE: the audio tower is a FULL RAW-AUDIO encoder (see clap.AudioTower). It does
NOT consume the frozen 128-d contrast style vectors anymore, so this script reads
raw wavs and reconstructs the per-measure windows itself.

Alignment strategy (the whole point of this script):
  * We do NOT re-derive the train/val split or the measure row ordering. The
    ground truth is the DiT's per-measure layout:
      - `..._map.json` gives each song's contiguous measure-row block
        [measure_start, measure_stop) in the DiT's per-measure bins, per split.
      - within a song, measure i == the i-th downbeat interval, reproduced from
        the SAME beat logic as generate_continuous_measures_dataset.py
        (parse_beat_file -> downbeat_indices, break when frame_end > len(wav)).
    Per-song blocks are independent, so a per-song count assert contains any
    misalignment to that one song rather than shifting everything after it.
  * For each measure we feed a 10s raw window CENTERED on the measure's time
    span (clamped to the song) through the raw-audio tower -- matching how CLAP
    was trained (~10s crops) and mirroring generate_meta_dataset's centered
    windows. Adjacent measures share most of their window -> smooth per-measure
    style, which is expected/desired.

Outputs (float16 unless noted):
  * `{clap_prefix}_clap_style_{train,val}.bin`  shape (N_measures, 512)
        per-measure CLAP AUDIO embedding, row-aligned with the DiT's per-measure
        signals. Drop-in 512-d style conditioning.
  * `{clap_prefix}_clap_text_{train,val}.bin`   shape (N_text_rows, 512)
        per-caption CLAP TEXT embedding, in the SAME shuffled row order as the
        T5 caption bin.
  * `{clap_prefix}_clap_index_{train,val}.npz`  the JOIN between the two:
        row_to_song[N_text_rows]   text row -> song id
        row_to_tier[N_text_rows]   text row -> 0=short 1=medium 2=long
        song_measure_start/stop[N_songs]  song id -> DiT measure row block
        song_paths[N_songs]        song id -> wav file_path
        Without this the text bin is unusable: it is in shuffled caption order
        with no way back to a song's measures.
  * `{clap_prefix}_clap_modality_{train,val}.npz`
        mean_audio[512], mean_text[512], plus matched-pair cosine stats. Lets you
        apply the centroid-shift correction at inference without a second pass.

Both modalities live in the same CLAP space, so during DiT training you can
randomly swap the audio-derived per-measure vector for one of the song's
text-derived vectors to make the DiT robust to the modality gap (text-only
inference). See the modality npz for how large that gap actually is.

Single GPU:
    $ python generate_clap_dataset.py
"""

import os
import json
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import librosa
from tqdm import tqdm

from clap import CLAP

# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------
device = 'cuda:0'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# trained CLAP checkpoint -- must match train_clap.py's out_dir
clap_dir = 'clap_nopre'
clap_prefix = clap_dir

# dims (validated against the checkpoint below)
proj_dim = 512
text_dim = 1024
n_text_tokens = 256

# raw-audio window fed to the audio tower per measure (must match train_clap.n_samples)
rate = 16000
n_samples = 16383 * 10        # ~10s centered window per measure

# raw audio + beats (same locations as generate_continuous_measures_dataset.py)
wav_dir = '/data/wavs'
beat_dir = '/data/beats'

bin_dir = '/data/binaries'

# song -> [measure_start, measure_stop) row blocks (ground-truth alignment + split)
AUDIO_MAPS = {
    'train': os.path.join(bin_dir, 'low_large_24576_subset_chroma_rms_density_zcr_flatness_train_map.json'),
    'val':   os.path.join(bin_dir, 'low_large_24576_subset_chroma_rms_density_zcr_flatness_val_map.json'),
}
# expected measure-row counts, used only as a cross-check -- the real sizes are
# DERIVED from the maps below so a stale constant can never mis-size a memmap.
N_MEASURES_EXPECTED = {'train': 4490789, 'val': 99131}

# T5 caption bins (row order == shuffled row order) + metadata for row counts
TEXT_BINS = {
    'train': os.path.join(bin_dir, 'caption_embeddings_expanded_shuffled_train.bin'),
    'val':   os.path.join(bin_dir, 'caption_embeddings_expanded_shuffled_val.bin'),
}
TEXT_META = os.path.join(bin_dir, 'caption_embeddings_expanded_shuffled_split_metadata.pkl')
# same caption source train_clap.py uses to build its song ordering
CAPTIONS_JSONL = '/home/dylandeshler/Jazz/preprocess/final_llm_captions_expanded.jsonl'

audio_measure_batch = 256     # centered windows per audio forward chunk
text_batch_rows = 512         # caption rows per text forward chunk

NUM_TIERS, NUM_VARS = 3, 6

# -----------------------------------------------------------------------------
# load frozen CLAP  (lazy: --check must not claim a GPU while training runs)
# -----------------------------------------------------------------------------
model = None


def load_clap():
    global model
    if model is not None:
        return model
    ckpt_path = os.path.join(clap_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    m = CLAP(**model_args).to(device)
    state_dict = checkpoint['model']
    # train_clap.py compiles the TOWERS (submodules), not the whole model, so the
    # '_orig_mod.' that torch.compile inserts lands MID-key:
    #   audio_tower._orig_mod.blocks.0.norm1.weight
    # A startswith() strip silently misses these and load_state_dict then fails
    # with every tower key missing+unexpected. Strip every occurrence, as
    # train_clap.py does.
    for k, v in list(state_dict.items()):
        if '_orig_mod.' in k:
            state_dict[k.replace('_orig_mod.', '')] = state_dict.pop(k)
    m.load_state_dict(state_dict)
    # eval() is load-bearing, not cosmetic: it disables SpecAugment (clap.py gates
    # it on self.training), DropPath, and the text-side dropout. Generating this
    # dataset in train mode would bake augmentation noise into every vector.
    m.eval()

    # validate dims so a silently-wrong memmap can never be written
    assert model_args['text_dim'] == text_dim, f"CLAP text_dim {model_args['text_dim']} != {text_dim}"
    assert model_args['proj_dim'] == proj_dim, f"CLAP proj_dim {model_args['proj_dim']} != {proj_dim}"
    print(f"Loaded CLAP from {ckpt_path} (iter {checkpoint.get('iter_num', '?')}): "
          f"raw-audio tower -> proj_dim={proj_dim}")
    model = m
    return model

with open(TEXT_META, 'rb') as f:
    text_meta = pickle.load(f)

# -----------------------------------------------------------------------------
# song ordering -- reproduced EXACTLY as train_clap.py builds it, because the
# song_idx recovered from shuffled_indices indexes into this list and nothing
# else. Any divergence here silently mis-joins every caption to the wrong song.
# -----------------------------------------------------------------------------
_raw_captions = None


def raw_captions():
    """Lazy: the jsonl is large and --check has no use for it."""
    global _raw_captions
    if _raw_captions is None:
        with open(CAPTIONS_JSONL, 'r', encoding='utf-8') as f:
            _raw_captions = [json.loads(line) for line in f]
    return _raw_captions


def song_paths_for(split):
    with open(AUDIO_MAPS[split], 'r') as f:
        audio_map = json.load(f)
    paths = [c.get('file_path', '') for c in raw_captions() if c.get('file_path', '') in audio_map]
    return paths, audio_map


def text_rows_for(split, n_songs):
    """Row count of the shuffled caption bin, cross-checked two independent ways."""
    meta_rows = int(np.prod(text_meta[split]['orig_sub_shape']))
    derived = n_songs * NUM_TIERS * NUM_VARS
    assert meta_rows == derived, (
        f"[{split}] caption row count disagreement: metadata orig_sub_shape gives "
        f"{meta_rows} but n_songs*3*6 gives {derived}. The song list and the caption "
        f"bin were built from different filters -- joining them would be garbage.")
    # third check: the bin on disk must actually be this big
    want = meta_rows * n_text_tokens * text_dim * 2      # float16
    got = os.path.getsize(TEXT_BINS[split])
    assert got == want, (
        f"[{split}] {TEXT_BINS[split]} is {got} bytes, expected {want} for shape "
        f"({meta_rows}, {n_text_tokens}, {text_dim}) float16.")
    return meta_rows


# -----------------------------------------------------------------------------
# crash durability
#
# This job is long enough that a mid-run machine crash is expected. The recovery
# contract is: the per-split modality npz is written LAST and ATOMICALLY, so
# `check_complete(split)` is trustworthy. Without the atomic rename, a crash
# during np.savez leaves a truncated .npz that still *exists* but fails to open,
# which would make an existence check actively misleading.
# -----------------------------------------------------------------------------
def _sync_memmap(mm):
    """flush() + fsync so the bin survives a kernel-level crash, not just a
    process death. numpy's flush() is msync, which pushes dirty pages, but we
    fsync the underlying file too so the write is durable before the npz marker
    that claims it finished."""
    mm.flush()
    try:
        with open(mm.filename, 'rb+') as f:
            os.fsync(f.fileno())
    except (OSError, AttributeError) as e:
        print(f'  NOTE: could not fsync {getattr(mm, "filename", "?")}: {e}')


def _atomic_savez(path, **arrays):
    """np.savez to a temp file, fsync, then os.replace (atomic on POSIX).
    Guarantees the file is either absent or complete -- never half-written."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check_complete(split, verbose=True):
    """Is `split` fully generated? Safe to call after a crash.

    Verifies the marker opens (not just exists), that no measure rows were left
    zero-filled, and that the companion bins/index are present and correctly
    sized. Returns True only if the split is genuinely usable."""
    marker = os.path.join(bin_dir, f'{clap_prefix}_clap_modality_{split}.npz')
    if not os.path.exists(marker):
        if verbose:
            print(f'[{split}] INCOMPLETE: no marker at {marker}')
        return False
    try:
        z = np.load(marker)
        n_written = int(z['audio_written'])
        n_total = int(z['audio_total'])
        complete = bool(z['complete'])
    except Exception as e:                      # truncated/corrupt npz
        if verbose:
            print(f'[{split}] INCOMPLETE: marker exists but is unreadable ({e})')
        return False

    ok = True
    style = os.path.join(bin_dir, f'{clap_prefix}_clap_style_{split}.bin')
    want = n_total * proj_dim * 2               # float16
    if not os.path.exists(style) or os.path.getsize(style) != want:
        got = os.path.getsize(style) if os.path.exists(style) else 'missing'
        if verbose:
            print(f'[{split}] INCOMPLETE: style bin is {got}, expected {want}')
        ok = False
    for name in (f'{clap_prefix}_clap_text_{split}.bin', f'{clap_prefix}_clap_index_{split}.npz'):
        if not os.path.exists(os.path.join(bin_dir, name)):
            if verbose:
                print(f'[{split}] INCOMPLETE: missing {name}')
            ok = False
    if not complete and verbose:
        print(f'[{split}] FINISHED BUT NOT COMPLETE: {n_total - n_written} of {n_total} '
              f'measure rows were never written and are zero vectors. The run did not '
              f'crash -- these are missing wavs/beats or measure-count mismatches.')
    if ok and complete and verbose:
        print(f'[{split}] complete: {n_written}/{n_total} measure rows.')
    return ok and complete


# -----------------------------------------------------------------------------
# beat parsing (copied verbatim from generate_continuous_measures_dataset.py so
# the measure enumeration matches the DiT bins exactly)
# -----------------------------------------------------------------------------
def parse_beat_file(beat_path):
    beat_data = []
    with open(beat_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    ts = float(parts[0])
                    bn = 0
                    if len(parts) >= 2:
                        try:
                            bn = int(float(parts[1]))
                            if bn > 0:
                                bn = ((bn - 1) % 4) + 1
                        except ValueError:
                            pass
                    beat_data.append({'time': ts, 'beat': bn})
                except ValueError:
                    continue
    return beat_data


def measure_spans(wav_len, beat_data):
    """Reproduce generate_continuous_measures_dataset.py's measure enumeration:
    consecutive downbeat intervals, dropping the tail once frame_end > wav_len.
    Returns a list of (frame_start, frame_end) in samples."""
    downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
    spans = []
    for i in range(len(downbeat_indices) - 1):
        t_start = beat_data[downbeat_indices[i]]['time']
        t_end = beat_data[downbeat_indices[i + 1]]['time']
        frame_start = int(t_start * rate)
        frame_end = int(t_end * rate)
        if frame_end > wav_len:
            break
        spans.append((frame_start, frame_end))
    return spans


def _resolve_wav(file_path):
    base = os.path.basename(file_path)
    p = os.path.join(wav_dir, base)
    return p if os.path.exists(p) else file_path


def _beat_path(file_path):
    return os.path.join(beat_dir, os.path.basename(file_path))


# -----------------------------------------------------------------------------
# AUDIO: raw wav -> per-measure 512-d CLAP audio embedding (centered 10s windows)
# -----------------------------------------------------------------------------
@torch.no_grad()
def build_audio(split, audio_map):
    # size the memmap from the map itself; a stale constant would shift every row
    n_measures_total = max(int(v[1]) for v in audio_map.values())
    if n_measures_total != N_MEASURES_EXPECTED.get(split):
        print(f"  NOTE: derived {n_measures_total} measure rows for '{split}', "
              f"expected constant says {N_MEASURES_EXPECTED.get(split)}. Using derived.")

    out_path = os.path.join(bin_dir, f'{clap_prefix}_clap_style_{split}.bin')
    dst = np.memmap(out_path, dtype=np.float16, mode='w+', shape=(n_measures_total, proj_dim))

    half = n_samples // 2
    written = 0
    mismatches = 0
    emb_sum = np.zeros(proj_dim, dtype=np.float64)
    # process songs in ascending row order (irrelevant to correctness, nice for I/O)
    items = sorted(audio_map.items(), key=lambda kv: int(kv[1][0]))
    for file_path, (m_start, m_stop) in tqdm(items, desc=f'audio[{split}]'):
        m_start, m_stop = int(m_start), int(m_stop)
        n_expected = m_stop - m_start
        if n_expected <= 0:
            continue

        wav_path = _resolve_wav(file_path)
        beat_path = _beat_path(file_path)
        if not (os.path.exists(wav_path) and os.path.exists(beat_path)):
            print(f'  WARNING: missing wav/beat for {file_path}; leaving {n_expected} rows unwritten')
            continue

        wav, _ = librosa.load(wav_path, sr=rate)
        spans = measure_spans(len(wav), parse_beat_file(beat_path))

        # per-song block is independent: clamp to the min so a mismatch corrupts
        # only this song's block, never shifting subsequent songs' rows.
        n = min(n_expected, len(spans))
        if len(spans) != n_expected:
            mismatches += 1
        if n <= 0:
            continue

        # build one centered 10s window per measure
        windows = np.empty((n, n_samples), dtype=np.float32)
        for i in range(n):
            fs, fe = spans[i]
            center = (fs + fe) // 2
            lo = center - half
            hi = lo + n_samples
            if lo < 0:
                lo, hi = 0, n_samples
            if hi > len(wav):
                hi, lo = len(wav), max(0, len(wav) - n_samples)
            seg = wav[lo:hi]
            if len(seg) < n_samples:
                seg = np.pad(seg, (0, n_samples - len(seg)))
            windows[i] = seg

        # encode in chunks
        out = np.empty((n, proj_dim), dtype=np.float16)
        for c in range(0, n, audio_measure_batch):
            chunk = torch.from_numpy(windows[c:c + audio_measure_batch]).unsqueeze(1)
            chunk = chunk.pin_memory().to(device, non_blocking=True)
            with ctx:
                emb = model.encode_audio(chunk)          # [b, 512] L2-normalized
            out[c:c + chunk.shape[0]] = emb.float().cpu().numpy().astype(np.float16)

        dst[m_start:m_start + n] = out
        emb_sum += out.astype(np.float64).sum(0)
        written += n

    _sync_memmap(dst)
    print(f'audio[{split}]: wrote {written}/{n_measures_total} measures -> {out_path}')
    if mismatches:
        print(f'  NOTE: {mismatches} songs had a beat-derived measure count != map count '
              f'(clamped per-song; alignment preserved for other songs)')
    if written != n_measures_total:
        print(f'  WARNING: {n_measures_total - written} rows left unwritten -- these stay '
              f'ZERO vectors in the bin (memmap w+ is zero-filled), which are NOT valid '
              f'unit-norm conditioning. Counts are recorded in the modality npz.')
    return emb_sum / max(written, 1), dict(
        audio_written=written, audio_total=n_measures_total,
        audio_missing=n_measures_total - written, audio_song_mismatches=mismatches,
    )


# -----------------------------------------------------------------------------
# TEXT: per-caption T5 tokens -> per-caption 512-d CLAP text embedding
# (written in the SAME shuffled row order as the source caption bin)
# -----------------------------------------------------------------------------
@torch.no_grad()
def build_text(split, n_rows):
    src = np.memmap(TEXT_BINS[split], dtype=np.float16, mode='r',
                    shape=(n_rows, n_text_tokens, text_dim))

    out_path = os.path.join(bin_dir, f'{clap_prefix}_clap_text_{split}.bin')
    dst = np.memmap(out_path, dtype=np.float16, mode='w+', shape=(n_rows, proj_dim))

    emb_sum = np.zeros(proj_dim, dtype=np.float64)
    for r in tqdm(range(0, n_rows, text_batch_rows), desc=f'text[{split}]'):
        rows = slice(r, min(r + text_batch_rows, n_rows))
        # keep fp16 across the wire (halves H2D vs upcasting on CPU) and build the
        # mask ON DEVICE -- both exactly as train_clap.py does it, so these
        # embeddings are produced by the identical code path the model trained under.
        text = torch.from_numpy(np.ascontiguousarray(src[rows])).to(device, non_blocking=True)
        tmask = (text.abs().sum(-1) > 0)     # T5 zero-padded; valid == any nonzero
        tmask[:, 0] = True                    # guard fully-empty rows
        with ctx:
            emb = model.encode_text(text, tmask)   # [b, 512] L2-normalized
        out = emb.float().cpu().numpy().astype(np.float16)
        dst[rows] = out
        emb_sum += out.astype(np.float64).sum(0)

    _sync_memmap(dst)
    print(f'text[{split}]: wrote {n_rows} rows -> {out_path}')
    return emb_sum / max(n_rows, 1)


# -----------------------------------------------------------------------------
# INDEX: the join between the shuffled caption rows and the DiT measure blocks
# -----------------------------------------------------------------------------
def build_index(split, song_paths, audio_map, n_rows):
    """Emit text_row -> (song, tier) and song -> (measure_start, measure_stop).

    The caption bin is in shuffled row order; recovering the song requires the
    same unravel train_clap.map_rows does. Doing it once here (vectorized) means
    the DiT loader needs no caption metadata at all -- just this npz."""
    shuffled = np.asarray(text_meta[split]['shuffled_indices'])
    sub_shape = tuple(text_meta[split]['orig_sub_shape'])
    assert len(shuffled) == n_rows, f'[{split}] shuffled_indices {len(shuffled)} != n_rows {n_rows}'

    song_idx, tier_idx, _var_idx = np.unravel_index(shuffled, sub_shape)

    n_songs = len(song_paths)
    m_start = np.zeros(n_songs, dtype=np.int64)
    m_stop = np.zeros(n_songs, dtype=np.int64)
    for i, p in enumerate(song_paths):
        s, e = audio_map[p]
        m_start[i], m_stop[i] = int(s), int(e)

    out_path = os.path.join(bin_dir, f'{clap_prefix}_clap_index_{split}.npz')
    _atomic_savez(
        out_path,
        row_to_song=song_idx.astype(np.int32),
        row_to_tier=tier_idx.astype(np.int8),
        song_measure_start=m_start,
        song_measure_stop=m_stop,
        song_paths=np.array(song_paths),      # plain unicode array: no pickle needed
    )
    print(f'index[{split}]: {n_rows} rows / {n_songs} songs -> {out_path}')


def modality_stats(split, mean_audio, mean_text, integrity):
    """Quantify the audio<->text modality gap AND act as the split's DONE marker.

    Both towers L2-normalize, but CLIP-family models still place the two
    modalities in separate cones. Training the DiT purely on audio vectors and
    prompting with text at inference feeds it an out-of-distribution vector; the
    centroid offset saved here is what a shift correction needs.

    This file is written LAST for the split and written ATOMICALLY, so its
    existence is a reliable "this split finished" marker (see check_complete).
    It also carries the row-integrity counters, because finishing and being
    complete are different things: unwritten measure rows stay zero-filled."""
    gap = float(np.linalg.norm(mean_audio - mean_text))
    ca, ct = np.linalg.norm(mean_audio), np.linalg.norm(mean_text)
    cos_centroids = float(mean_audio @ mean_text / (ca * ct + 1e-12))
    out_path = os.path.join(bin_dir, f'{clap_prefix}_clap_modality_{split}.npz')
    _atomic_savez(out_path,
                  mean_audio=mean_audio.astype(np.float32),
                  mean_text=mean_text.astype(np.float32),
                  centroid_l2_gap=np.float32(gap),
                  centroid_cosine=np.float32(cos_centroids),
                  complete=np.bool_(integrity['audio_missing'] == 0),
                  **{k: np.int64(v) for k, v in integrity.items()})
    print(f'modality[{split}]: centroid L2 gap {gap:.4f}, centroid cosine {cos_centroids:.4f}, '
          f'|mean_audio|={ca:.4f} |mean_text|={ct:.4f} -> {out_path}')
    print('  (a large gap / low cosine == text prompts are OOD for an audio-trained DiT; '
          'use conditioning-noise augmentation or swap in text vectors during DiT training)')


if __name__ == '__main__':
    import sys

    # `python generate_clap_dataset.py --check` verifies a previous (possibly
    # crashed) run without touching the GPU or rewriting anything.
    if '--check' in sys.argv:
        all_ok = all(check_complete(s) for s in ('train', 'val'))
        print('ALL SPLITS COMPLETE' if all_ok else 'NOT COMPLETE -- see above')
        sys.exit(0 if all_ok else 1)

    for split in ('train', 'val'):
        # skip splits already finished, so a crashed run resumes at split
        # granularity instead of redoing everything
        if check_complete(split, verbose=False):
            print(f'[{split}] already complete, skipping')
            continue

        load_clap()
        song_paths, audio_map = song_paths_for(split)
        n_rows = text_rows_for(split, len(song_paths))
        print(f"[{split}] {len(song_paths)} songs, {n_rows} caption rows")

        mean_audio, integrity = build_audio(split, audio_map)
        mean_text = build_text(split, n_rows)
        build_index(split, song_paths, audio_map, n_rows)
        # LAST for the split, and atomic -- this is the DONE marker
        modality_stats(split, mean_audio, mean_text, integrity)

    print('\n--- final verification ---')
    if all(check_complete(s) for s in ('train', 'val')):
        print('CLAP conditioning dataset generation complete.')
    else:
        print('CLAP conditioning dataset generation FINISHED WITH GAPS (see above).')
