"""
Within-song BPM variation analysis.

Reproduces the per-measure BPM extraction used by generate_bpm_dataset.py
    instant_bpm = (TARGET_SIG / (t_downbeat[i+1] - t_downbeat[i])) * 60
but keeps every song's measure sequence separate, so no statistic ever crosses
a song boundary (the flattened *_bpm_{train,val}.bin files have no boundaries in
them at all -- that information only exists here, in the beat files).

Answers two questions:
  1. how much does BPM move within a song?
  2. what does that imply for how the DiT should be conditioned on it?

Usage:
    python3 analyze_bpm_variation.py
    python3 analyze_bpm_variation.py --limit 500          # quick pass
    python3 analyze_bpm_variation.py --csv per_song_bpm.csv --boundaries bpm_song_boundaries.json
    python3 analyze_bpm_variation.py --plot bpm_plots     # needs matplotlib
"""

import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
import concurrent.futures
from multiprocessing import cpu_count

# ------------------------------------------------------------------ config --
TARGET_SIG = 4                      # generate_bpm_dataset.py assumes 4/4
WAV_DIR = '/data/wavs'
BEAT_DIR = '/data/beats'
VALID_JSON = '/data/valid_files_by_bpm.json'
TOTAL_WRITE_BATCHES = 48            # shard count used by generate_bpm_dataset.py

BPM_MIN, BPM_MAX = 30.0, 400.0      # anything outside is a tracker failure
BPM_BINS = np.arange(40, 300, 5, dtype=np.float32)   # matches training code


def parse_beat_file(beat_path):
    """Verbatim copy of the parser in generate_bpm_dataset.py / train_adapter_low_dito6.py,
    including the 8/4 -> 4/4 renumbering fix."""
    beat_data = []
    if not os.path.exists(beat_path):
        return beat_data

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


def measure_bpms(beat_path):
    """Per-measure BPM sequence + downbeat start times, in song order.

    Returns (bpms, times, n_raw) where n_raw is the count *before* the
    plausibility filter, so we can report how many measures the tracker fumbled.
    Index i of the returned array is the i-th measure written into the flat bin.
    """
    beat_data = parse_beat_file(beat_path)
    downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]

    bpms, times, n_raw = [], [], 0
    for i in range(len(downbeat_indices) - 1):
        t_start = beat_data[downbeat_indices[i]]['time']
        t_end = beat_data[downbeat_indices[i + 1]]['time']
        duration_sec = t_end - t_start
        if duration_sec <= 0:
            continue
        n_raw += 1
        bpm = (TARGET_SIG / duration_sec) * 60.0
        bpms.append(bpm)
        times.append(t_start)

    return np.asarray(bpms, dtype=np.float64), np.asarray(times, dtype=np.float64), n_raw


# --------------------------------------------------------- octave handling --
def fold_octaves(bpms, ref):
    """Halve/double each measure toward `ref`. Beat trackers routinely drop or
    insert a downbeat, which shows up as an exact 2x / 0.5x outlier -- that is a
    annotation artifact, not musical tempo variation, and mixing the two would
    make the corpus look far more unstable than it is."""
    out = bpms.copy()
    for _ in range(3):
        out = np.where(out > 1.45 * ref, out / 2.0, out)
        out = np.where(out < 0.69 * ref, out * 2.0, out)
    return out


def analyze_song(name, beat_path):
    bpms, times, n_raw = measure_bpms(beat_path)
    if len(bpms) < 4:
        return None

    valid = (bpms >= BPM_MIN) & (bpms <= BPM_MAX)
    n_implausible = int((~valid).sum())
    bpms, times = bpms[valid], times[valid]
    if len(bpms) < 4:
        return None

    ref = float(np.median(bpms))
    folded = fold_octaves(bpms, ref)
    n_folded = int(np.sum(~np.isclose(folded, bpms)))
    ref = float(np.median(folded))   # re-centre after folding

    def stats(x):
        q25, q50, q75 = np.percentile(x, [25, 50, 75])
        d = np.diff(x)
        return {
            'mean': float(np.mean(x)),
            'median': float(q50),
            'std': float(np.std(x)),
            'cv': float(np.std(x) / q50) if q50 > 0 else float('nan'),
            'iqr': float(q75 - q25),
            'iqr_pct': float((q75 - q25) / q50 * 100) if q50 > 0 else float('nan'),
            'p05': float(np.percentile(x, 5)),
            'p95': float(np.percentile(x, 95)),
            'range_pct': float((np.percentile(x, 95) - np.percentile(x, 5)) / q50 * 100) if q50 > 0 else float('nan'),
            'max_jump': float(np.max(np.abs(d))) if len(d) else 0.0,
            'med_jump': float(np.median(np.abs(d))) if len(d) else 0.0,
            'med_jump_pct': float(np.median(np.abs(d)) / q50 * 100) if len(d) and q50 > 0 else 0.0,
        }

    s_raw = stats(bpms)
    s = stats(folded)

    # --- shape of the variation: drift vs jitter -----------------------------
    n = len(folded)
    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, folded, 1)
    trend = slope * t + intercept
    resid = folded - trend
    var_tot = float(np.var(folded))
    drift_frac = float(1.0 - np.var(resid) / var_tot) if var_tot > 1e-9 else 0.0

    # white noise has std(diff) == sqrt(2)*std; smooth drift has much less.
    d = np.diff(folded)
    jitter_index = float(np.std(d) / (np.sqrt(2.0) * np.std(folded))) if np.std(folded) > 1e-9 else float('nan')

    # lag-1 autocorrelation of deviations from the song median
    dev = folded - ref
    if n > 2 and np.std(dev) > 1e-9:
        ac1 = float(np.corrcoef(dev[:-1], dev[1:])[0, 1])
    else:
        ac1 = float('nan')

    # how much a 5-measure median filter removes -> is it locally smooth?
    k = 5
    if n >= k:
        pad = k // 2
        padded = np.pad(folded, pad, mode='edge')
        smooth = np.array([np.median(padded[i:i + k]) for i in range(n)])
        smooth_resid_std = float(np.std(folded - smooth))
    else:
        smooth = folded.copy()
        smooth_resid_std = float('nan')

    # --- predictor errors: what do you lose with a constant per-song BPM? ----
    mae_const = float(np.mean(np.abs(folded - ref)))
    mae_trend = float(np.mean(np.abs(resid)))
    mae_prev = float(np.mean(np.abs(d))) if len(d) else 0.0     # previous-measure predictor

    # --- bin-level: does within-song drift even change the conditioning bin? -
    # the raw series crosses bin edges from jitter alone, which carries no
    # information; the smoothed series is the one that says whether a per-measure
    # target is meaningfully different from a per-song constant.
    bin_song = int(np.argmin(np.abs(BPM_BINS - ref)))
    bin_meas = np.argmin(np.abs(folded[:, None] - BPM_BINS[None, :]), axis=1)
    bin_smooth = np.argmin(np.abs(smooth[:, None] - BPM_BINS[None, :]), axis=1)
    bin_mismatch = float(np.mean(bin_meas != bin_song))
    bin_mismatch_smooth = float(np.mean(bin_smooth != bin_song))
    bin_off_by = float(np.mean(np.abs(bin_meas - bin_song)))

    return {
        'name': name,
        'n_measures': n,
        'n_raw': n_raw,
        'n_implausible': n_implausible,
        'n_folded': n_folded,
        'folded_frac': n_folded / max(1, len(bpms)),
        'duration_min': float((times[-1] - times[0]) / 60.0) if n > 1 else 0.0,
        'song_bpm': ref,
        'raw_cv': s_raw['cv'],
        'raw_range_pct': s_raw['range_pct'],
        **{k2: v for k2, v in s.items()},
        'slope_bpm_per_measure': float(slope),
        'slope_bpm_per_min': float(slope) * (60.0 / (TARGET_SIG * 60.0 / ref)) if ref > 0 else 0.0,
        'total_drift_bpm': float(slope) * (n - 1),
        'drift_frac_of_var': drift_frac,
        'jitter_index': jitter_index,
        'autocorr_lag1': ac1,
        'smooth_resid_std': smooth_resid_std,
        'mae_const': mae_const,
        'mae_trend': mae_trend,
        'mae_prev': mae_prev,
        'bin_mismatch': bin_mismatch,
        'bin_mismatch_smooth': bin_mismatch_smooth,
        'bin_off_by': bin_off_by,
        'series': folded,
    }


# ------------------------------------------------------------------ report --
def pct_line(label, arr, unit='', fmt='{:7.2f}'):
    arr = np.asarray([a for a in arr if np.isfinite(a)], dtype=np.float64)
    if len(arr) == 0:
        print(f'  {label:<34} (no finite values)')
        return
    qs = np.percentile(arr, [5, 25, 50, 75, 95])
    vals = ' '.join(fmt.format(q) for q in qs)
    print(f'  {label:<34} {vals}   mean {fmt.format(np.mean(arr))}{unit}')


def report(songs):
    n = len(songs)
    all_measures = int(sum(s['n_measures'] for s in songs))
    print()
    print('=' * 92)
    print(f'WITHIN-SONG BPM VARIATION   |   {n} songs, {all_measures} measures')
    print('=' * 92)

    tot_raw = sum(s['n_raw'] for s in songs)
    tot_imp = sum(s['n_implausible'] for s in songs)
    tot_fold = sum(s['n_folded'] for s in songs)
    print(f'\nBeat-tracker hygiene (these are annotation artifacts, not tempo variation):')
    print(f'  measures parsed                    {tot_raw}')
    print(f'  dropped, BPM outside [{BPM_MIN:.0f},{BPM_MAX:.0f}]     {tot_imp} ({tot_imp / max(1, tot_raw) * 100:.2f}%)')
    print(f'  octave-folded (2x / 0.5x)          {tot_fold} ({tot_fold / max(1, tot_raw) * 100:.2f}%)')

    print(f'\nSong-level tempo (median of per-measure BPM)')
    print(f'  {"":<34} {"p5":>7} {"p25":>7} {"p50":>7} {"p75":>7} {"p95":>7}')
    pct_line('song BPM', [s['song_bpm'] for s in songs])
    pct_line('measures per song', [s['n_measures'] for s in songs], fmt='{:7.0f}')
    pct_line('song duration (min)', [s['duration_min'] for s in songs])

    print(f'\nWithin-song spread  (per song, across its own measures)')
    print(f'  {"":<34} {"p5":>7} {"p25":>7} {"p50":>7} {"p75":>7} {"p95":>7}')
    pct_line('std (BPM)', [s['std'] for s in songs])
    pct_line('coefficient of variation (%)', [s['cv'] * 100 for s in songs])
    pct_line('IQR (BPM)', [s['iqr'] for s in songs])
    pct_line('IQR (% of song BPM)', [s['iqr_pct'] for s in songs])
    pct_line('p5-p95 range (% of song BPM)', [s['range_pct'] for s in songs])
    pct_line('median |measure-to-measure| (BPM)', [s['med_jump'] for s in songs])
    pct_line('max |measure-to-measure| (BPM)', [s['max_jump'] for s in songs])
    print(f'\n  (for comparison, before octave folding:)')
    pct_line('raw CV (%)', [s['raw_cv'] * 100 for s in songs])
    pct_line('raw p5-p95 range (%)', [s['raw_range_pct'] for s in songs])

    print(f'\nShape of the variation  -- is it drift or jitter?')
    print(f'  {"":<34} {"p5":>7} {"p25":>7} {"p50":>7} {"p75":>7} {"p95":>7}')
    pct_line('total linear drift over song (BPM)', [s['total_drift_bpm'] for s in songs])
    pct_line('|total drift| (BPM)', [abs(s['total_drift_bpm']) for s in songs])
    pct_line('var explained by linear drift (%)', [s['drift_frac_of_var'] * 100 for s in songs])
    pct_line('jitter index (1.0 = white noise)', [s['jitter_index'] for s in songs], fmt='{:7.3f}')
    pct_line('lag-1 autocorrelation', [s['autocorr_lag1'] for s in songs], fmt='{:7.3f}')
    pct_line('resid std after 5-measure median', [s['smooth_resid_std'] for s in songs])

    speeding = sum(1 for s in songs if s['total_drift_bpm'] > 2)
    slowing = sum(1 for s in songs if s['total_drift_bpm'] < -2)
    print(f'\n  songs speeding up >2 BPM end-to-end:  {speeding} ({speeding / n * 100:.1f}%)')
    print(f'  songs slowing down >2 BPM end-to-end: {slowing} ({slowing / n * 100:.1f}%)')

    # ---- variance decomposition -------------------------------------------
    grand = np.concatenate([s['series'] for s in songs])
    grand_mean = float(np.mean(grand))
    ss_within = float(sum(np.sum((s['series'] - np.mean(s['series'])) ** 2) for s in songs))
    ss_between = float(sum(s['n_measures'] * (np.mean(s['series']) - grand_mean) ** 2 for s in songs))
    ss_total = ss_within + ss_between
    print(f'\nVariance decomposition over all {all_measures} measures')
    print(f'  between songs   {ss_between / ss_total * 100:6.2f}%   (std {np.sqrt(ss_between / all_measures):6.2f} BPM)')
    print(f'  within songs    {ss_within / ss_total * 100:6.2f}%   (std {np.sqrt(ss_within / all_measures):6.2f} BPM)')

    # ---- predictors --------------------------------------------------------
    print(f'\nPredictor error, per measure (MAE in BPM)')
    print(f'  {"":<34} {"p5":>7} {"p25":>7} {"p50":>7} {"p75":>7} {"p95":>7}')
    pct_line('constant = song median', [s['mae_const'] for s in songs])
    pct_line('linear drift over the song', [s['mae_trend'] for s in songs])
    pct_line('previous measure', [s['mae_prev'] for s in songs])
    w = np.asarray([s['n_measures'] for s in songs], dtype=np.float64)
    for label, key in (('constant', 'mae_const'), ('linear', 'mae_trend'), ('prev-measure', 'mae_prev')):
        v = np.asarray([s[key] for s in songs], dtype=np.float64)
        print(f'  measure-weighted MAE, {label:<14} {np.sum(v * w) / np.sum(w):6.3f} BPM')

    # ---- what it means for the conditioning bins ---------------------------
    print(f'\nConditioning-bin impact  (bpm_bins = arange(40, 300, 5), sigma 5)')
    print(f'  {"":<34} {"p5":>7} {"p25":>7} {"p50":>7} {"p75":>7} {"p95":>7}')
    pct_line('measures in a different bin (%)', [s['bin_mismatch'] * 100 for s in songs])
    pct_line('  same, after 5-measure smooth (%)', [s['bin_mismatch_smooth'] * 100 for s in songs])
    pct_line('mean |bin offset| from song bin', [s['bin_off_by'] for s in songs], fmt='{:7.3f}')
    wm = float(np.sum(np.asarray([s['bin_mismatch'] for s in songs]) * w) / np.sum(w))
    wms = float(np.sum(np.asarray([s['bin_mismatch_smooth'] for s in songs]) * w) / np.sum(w))
    print(f'  measure-weighted mismatch      {wm * 100:6.2f}%  of measures land off the song bin')
    print(f'  ... of which real (smoothed)   {wms * 100:6.2f}%  the rest is jitter across a bin edge')

    # ---- how many measures are "close enough" to the song tempo ------------
    for tol in (1, 2, 3, 5, 10):
        hits = float(sum(np.sum(np.abs(s['series'] - s['song_bpm']) <= tol) for s in songs))
        print(f'  within +/-{tol:2d} BPM of song median: {hits / all_measures * 100:6.2f}% of measures')

    # ---- recommendation ----------------------------------------------------
    med_cv = float(np.median([s['cv'] for s in songs])) * 100
    med_jit = float(np.nanmedian([s['jitter_index'] for s in songs]))
    med_drift = float(np.median([abs(s['total_drift_bpm']) for s in songs]))
    print('\n' + '-' * 92)
    print('READ-OUT')
    print('-' * 92)
    print(f'  median within-song CV is {med_cv:.2f}%; median |end-to-end drift| is {med_drift:.2f} BPM;')
    print(f'  {ss_within / ss_total * 100:.1f}% of all BPM variance is within-song, {ss_between / ss_total * 100:.1f}% is between-song.')
    if med_jit > 0.8:
        print('  jitter index near 1.0 => the measure-to-measure wobble is essentially white noise')
        print('  (measurement noise in the downbeat timestamps), NOT a musical trajectory.')
        print('  => condition on a single per-song tempo; a per-measure BPM token would mostly')
        print('     be feeding the model annotation noise. If you want per-measure resolution,')
        print('     smooth the series first (median filter) so you keep drift and drop jitter.')
    else:
        print('  jitter index well below 1.0 => the series is locally smooth, so within-song')
        print('  motion is a real trajectory the model can learn. A per-measure (smoothed) BPM')
        print('  conditioning signal carries information a single per-song value would lose.')
    if wms < 0.15:
        print(f'  only {wms * 100:.1f}% of measures leave the song\'s 5-BPM bin once jitter is smoothed')
        print(f'  out (vs {wm * 100:.1f}% raw), and bpm_sigma=5 already smears one bin either way => the')
        print('  per-measure target is near-indistinguishable from a per-song constant.')
    else:
        print(f'  {wms * 100:.1f}% of measures leave the song\'s bin even after smoothing => the drift is')
        print('  real and per-measure BPM is doing work a song-level constant would lose.')
    print('-' * 92)


def write_csv(songs, path):
    keys = [k for k in songs[0].keys() if k != 'series']
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for s in songs:
            f.write(','.join(str(s[k]) for k in keys) + '\n')
    print(f'\nwrote per-song stats -> {path}')


def write_boundaries(songs_in_order, path):
    """Song boundaries into the flat concatenated BPM stream.

    generate_bpm_dataset.py concatenates every song's measure sequence with no
    record of where one ends -- this reconstructs that index so downstream code
    can respect song boundaries. Only songs that contributed >=1 measure appear,
    in the same order, which is the order the flat bin was built in.
    """
    offsets, cur = [], 0
    for s in songs_in_order:
        offsets.append({'name': s['name'], 'start': cur, 'stop': cur + s['n_measures'],
                        'song_bpm': round(s['song_bpm'], 4)})
        cur += s['n_measures']
    with open(path, 'w') as f:
        json.dump({'total_measures': cur, 'songs': offsets}, f)
    print(f'wrote {len(offsets)} song boundaries ({cur} measures) -> {path}')
    print('  NOTE: this uses the plausibility/octave filter, so it matches the flat bin only')
    print('  if that filter dropped nothing. Compare total_measures against the .bin length.')


def make_plots(songs, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].hist([s['cv'] * 100 for s in songs], bins=80, range=(0, 25))
    ax[0, 0].set_title('within-song CV (%)')
    ax[0, 1].hist([s['total_drift_bpm'] for s in songs], bins=80, range=(-30, 30))
    ax[0, 1].set_title('end-to-end linear drift (BPM)')
    ax[1, 0].hist([s['jitter_index'] for s in songs], bins=60, range=(0, 1.5))
    ax[1, 0].set_title('jitter index (1.0 = white noise)')
    ax[1, 1].scatter([s['song_bpm'] for s in songs], [s['std'] for s in songs], s=2, alpha=0.3)
    ax[1, 1].set_xlabel('song BPM'); ax[1, 1].set_ylabel('within-song std (BPM)')
    ax[1, 1].set_title('spread vs tempo')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'bpm_variation_summary.png'), dpi=120)
    plt.close(fig)

    # a few example trajectories across the variability range
    order = sorted(songs, key=lambda s: s['cv'])
    picks = [order[int(q * (len(order) - 1))] for q in (0.1, 0.35, 0.6, 0.8, 0.93, 0.99)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.1 * len(picks)), sharex=False)
    for a, s in zip(np.atleast_1d(axes), picks):
        a.plot(s['series'], lw=0.8)
        a.axhline(s['song_bpm'], color='r', ls='--', lw=0.8)
        a.set_title(f"{s['name'][:60]}  CV={s['cv'] * 100:.1f}%  drift={s['total_drift_bpm']:+.1f}", fontsize=8)
        a.set_ylabel('BPM', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'bpm_example_trajectories.png'), dpi=120)
    plt.close(fig)
    print(f'wrote plots -> {out_dir}/')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wavs', default=WAV_DIR)
    p.add_argument('--beats', default=BEAT_DIR)
    p.add_argument('--valid-json', default=VALID_JSON)
    p.add_argument('--limit', type=int, default=0, help='only process the first N songs')
    p.add_argument('--csv', default='', help='write per-song stats to this CSV')
    p.add_argument('--boundaries', default='', help='write song boundary index JSON')
    p.add_argument('--plot', default='', help='directory for summary plots')
    p.add_argument('--workers', type=int, default=max(1, cpu_count() // 2))
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.wavs, '*')))
    if os.path.exists(args.valid_json):
        with open(args.valid_json, 'r') as f:
            beat_paths = json.load(f)
        paths = [q for q in paths if os.path.basename(q) in beat_paths]
    else:
        print(f'WARNING: {args.valid_json} not found, using every wav in {args.wavs}')
    if args.limit:
        paths = paths[:args.limit]
    print(f'Total valid files: {len(paths)}')
    if not paths:
        raise SystemExit('no input files')

    results = [None] * len(paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(analyze_song, os.path.basename(q),
                          os.path.join(args.beats, os.path.basename(q))): i
                for i, q in enumerate(paths)}
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc='Parsing beats'):
            results[futs[fut]] = fut.result()

    songs = [r for r in results if r is not None]   # still in corpus order
    skipped = len(paths) - len(songs)
    if skipped:
        print(f'skipped {skipped} songs with <4 usable measures')
    if not songs:
        raise SystemExit('no songs with usable beat data')

    report(songs)
    if args.csv:
        write_csv(songs, args.csv)
    if args.boundaries:
        write_boundaries(songs, args.boundaries)
    if args.plot:
        make_plots(songs, args.plot)


if __name__ == '__main__':
    main()
