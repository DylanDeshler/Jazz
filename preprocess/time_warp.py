import os
import glob
import json
import librosa
import argparse
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

TARGET_SR = 16000
TARGET_SAMPLES = 24576
TARGET_SIG = 4
NUM_WORKERS = 48

def parse_beat_file(beat_path):
    """
    Parses the beat_this output file.
    Expected format per line: <timestamp> <beat_number>
    
    Returns a list of dictionaries: {'time': float, 'beat': int}
    """
    beat_data = []
    with open(beat_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    ts = float(parts[0])
                    # specific beat number (1, 2, 3, 4)
                    # Default to 0 if not present
                    bn = 0 
                    if len(parts) >= 2:
                        try:
                            bn = int(float(parts[1]))
                        except ValueError:
                            pass
                    
                    beat_data.append({'time': ts, 'beat': bn})
                except ValueError:
                    continue
    
    return beat_data

def get_time_signature(beat_data):
    """
    Estimates the numerator of the time signature (e.g., 4 for 4/4) 
    based on the most frequent measure length found in the file.
    """
    if not beat_data:
        return 0
        
    beat_nums = np.array([b['beat'] for b in beat_data])
    downbeat_indices = np.where(beat_nums == 1)[0]
    
    if len(downbeat_indices) < 2:
        return 0
        
    beats_per_measure = []
    for i in range(len(downbeat_indices) - 1):
        start = downbeat_indices[i]
        end = downbeat_indices[i+1]
        count = end - start
        beats_per_measure.append(count)
        
    if not beats_per_measure:
        return 0
        
    vals, counts = np.unique(beats_per_measure, return_counts=True)
    mode_time_sig = vals[np.argmax(counts)]
    return mode_time_sig

def process_measure(y):
    current_samples = len(y)
    stretch_factor = current_samples / TARGET_SAMPLES
    duration_sec = current_samples / TARGET_SR
    instant_bpm = (TARGET_SIG / duration_sec) * 60
    
    y_warped = pyrb.time_stretch(y, TARGET_SR, stretch_factor)

    if len(y_warped) > TARGET_SAMPLES:
        y_warped = y_warped[:TARGET_SAMPLES]
    elif len(y_warped) < TARGET_SAMPLES:
        y_warped = np.pad(y_warped, (0, TARGET_SAMPLES - len(y_warped)))
        
    return y_warped, stretch_factor, instant_bpm

def restore_measure(audio, stretch_ratio, sr=16000):
    """
    Restores a time-warped measure to its original duration.
    
    Args:
        audio (np.array): The fixed-length audio (from VAE or .npy file).
                          Can be shape (1, 24576) or (24576,).
        stretch_ratio (float): The ratio saved in your metadata 
                               (Original Length / Target Length).
        sr (int): Sampling rate (default 16000).
        
    Returns:
        np.array: The restored audio array at original duration.
    """
    
    if audio.dtype == np.float16:
        audio = audio.astype(np.float32)
    restore_rate = 1.0 / stretch_ratio
    
    y_restored = pyrb.time_stretch(audio, sr, restore_rate)
    return y_restored

def test(n_samples):
    paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures/*.npz')
    paths = np.random.choice(paths, n_samples)
    
    for j, path in enumerate(paths):
        data = np.load(path)
        audio = data['audio']
        idx = np.random.randint(len(audio))
        
        restored = restore_measure(audio[idx], data['ratio'][idx], TARGET_SR)
        
        wav_path = path.replace('jazz_data_16000_full_clean_measures', 'jazz_data_16000_full_clean').replace('.npz', '.wav')
        beat_path = path.replace('jazz_data_16000_full_clean_measures', 'jazz_data_16000_full_clean_beats').replace('.npz', '.beats')
        beat_data = parse_beat_file(beat_path)
        
        y, sr = librosa.load(wav_path, sr=None)
        assert sr == TARGET_SR
        
        downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
        start_idx = downbeat_indices[idx]
        end_idx = downbeat_indices[idx+1]
        
        t_start = beat_data[start_idx]['time']
        t_end = beat_data[end_idx]['time']
        
        frame_start = int(t_start * sr)
        frame_end = int(t_end * sr)
        
        y = y[frame_start:frame_end]
        print(restored.shape, y.shape)
        
        sf.write(f'{j}_real.wav', y, TARGET_SR)
        sf.write(f'{j}_restored.wav', restored, TARGET_SR)

def generate_audio_measures(paths):
    """
    Clips full measures (Downbeat '1' to the next Downbeat '1').
    """
    audio_path, beat_path = paths
    
    out_path = audio_path.replace('jazz_data_16000_full_clean', 'jazz_data_16000_full_clean_measures').replace('.wav', '.npz')
    if os.path.exists(out_path):
        return
    
    beat_data = parse_beat_file(beat_path)
    
    downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
    
    if len(downbeat_indices) < 10:
        return
    
    y, sr = librosa.load(audio_path, sr=None)
    assert sr == TARGET_SR
    
    if len(y) < 1000:
        return
    
    if np.max(np.abs(y)) < 0.001:
        return

    audios, stretch_ratios, instant_bpms = [], [], []
    for i in range(len(downbeat_indices) - 1):
        start_idx = downbeat_indices[i]
        end_idx = downbeat_indices[i+1]
        
        t_start = beat_data[start_idx]['time']
        t_end = beat_data[end_idx]['time']
        
        frame_start = int(t_start * sr)
        frame_end = int(t_end * sr)
        
        audio, stretch_ratio, instant_bpm = process_measure(y[frame_start:frame_end])
        audios.append(audio)
        stretch_ratios.append(stretch_ratio)
        instant_bpms.append(instant_bpm)
    
    if np.mean(instant_bpms) < 40 or np.mean(instant_bpms) > 330:
        return
    
    np.savez_compressed(
        out_path, 
        audio=np.stack(audios, axis=0).astype(np.float16), 
        ratio=np.array(stretch_ratios, dtype=np.float32), 
        bpm=np.array(instant_bpms, dtype=np.float32)
    )

def main():
    print("Gathering files...")
    audio_paths = sorted(glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/*.wav'))
    beat_paths = sorted(glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_beats/*.beats'))

    valid_audio, valid_beats = [], []
    print(f"Filtering for songs with Time Signature: 4/4 ...")
    for audio_p, beat_p in zip(audio_paths, beat_paths):
        beat_data = parse_beat_file(beat_p)
        detected_sig = get_time_signature(beat_data)
        
        if detected_sig == TARGET_SIG:
            valid_audio.append(audio_p)
            valid_beats.append(beat_p)

    print(f"Found {len(valid_beats)} matching songs (out of {len(audio_paths)} total).")
    audio_paths = valid_audio
    beat_paths = valid_beats
    del valid_audio
    del valid_beats

    assert len(audio_paths) == len(beat_paths)
    
    tasks = []
    for audio_path, beat_path in zip(audio_paths, beat_paths):
        tasks.append((audio_path, beat_path))
        
    print(f"Found {len(tasks)} files. Processing with {NUM_WORKERS} cores...")
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(generate_audio_measures, tasks), total=len(tasks)))
        
    print("Done!")

def crunch():
    paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures/*.npz')
    
    length = 0
    for path in tqdm(paths, desc='Calculating Total Length'):
        length += len(np.load(path)['audio'])
    
    audio_mmap = np.memmap(
        '/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures_audio.npy', 
        dtype=np.float16, 
        mode='w+', 
        shape=(length, TARGET_SAMPLES)
    )
    meta_mmap = np.memmap(
        '/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures_meta.npy', 
        dtype=np.float32, 
        mode='w+', 
        shape=(length, 2)
    )
    
    curr_index = 0
    song_index = {
        'sample_rate': 16000,
        'samples': 24576,
        'signature': 4,
        'data': {}
    }
    for path in tqdm(paths, desc='Writing Contiguous Data'):
        data = np.load(path)
        
        audio = data['audio']
        ratio = data['ratio']
        bpm = data['bpm']
        
        audio_mmap[curr_index:curr_index+len(audio)] = audio
        meta_mmap[curr_index:curr_index+len(audio), 0] = ratio
        meta_mmap[curr_index:curr_index+len(audio), 1] = bpm
        
        name = os.path.basename(path)
        name, root = os.path.splitext(name)
        song_index['data'][name] = [curr_index, curr_index + len(audio)]
        
        curr_index += len(audio)
    
    audio_mmap.flush()
    meta_mmap.flush()
    
    with open('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures_songs.json', 'w') as f:
        json.dump(song_index, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beat sampling and statistical analysis tool.")
    
    parser.add_argument("--test", action='store_true', default=False, help="True to test the effects of warping and unwarping measures")
    parser.add_argument("--n", type=int, default=20, help="The number of measures to test")
    
    args = parser.parse_args()
    
    if args.test:
        test(args.n)
    else:
        crunch()