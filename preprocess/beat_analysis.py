import argparse
import os
import random
import sys
import numpy as np
import librosa
import soundfile as sf
import glob

def parse_beat_file(beat_path):
    """
    Parses the beat_this output file.
    Expected format per line: <timestamp> <beat_number>
    
    Returns a list of dictionaries: {'time': float, 'beat': int}
    """
    beat_data = []
    try:
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
    except FileNotFoundError:
        print(f"Error: Beat file not found at {beat_path}")
        sys.exit(1)
    
    return beat_data

def calculate_file_stats(beat_data):
    """
    Calculates statistical metrics for a single beat dataset.
    """
    if not beat_data or len(beat_data) < 2:
        return None

    timestamps = np.array([b['time'] for b in beat_data])
    beat_nums = np.array([b['beat'] for b in beat_data])

    # 1. Inter-Beat Intervals (IBI)
    # The time difference between consecutive beats
    ibis = np.diff(timestamps)
    if len(ibis) == 0:
        return None

    avg_ibi = np.mean(ibis)
    std_ibi = np.std(ibis) # Low std dev = steady tempo, High = variable/expressive
    
    # 2. BPM (Beats Per Minute)
    # We calculate instantaneous BPM for every interval to find the spread
    inst_bpms = 60.0 / ibis
    avg_bpm = np.mean(inst_bpms)
    median_bpm = np.median(inst_bpms)
    std_bpm = np.std(inst_bpms)
    min_bpm = np.min(inst_bpms)
    max_bpm = np.max(inst_bpms)

    # 3. Measures / Time Signature Estimation
    # We look for the distance between "Beat 1"s (Downbeats)
    downbeat_indices = np.where(beat_nums == 1)[0]
    beats_per_measure = []
    measure_durations = []

    if len(downbeat_indices) > 1:
        for i in range(len(downbeat_indices) - 1):
            start = downbeat_indices[i]
            end = downbeat_indices[i+1]
            
            # Count beats in this measure
            count = end - start
            beats_per_measure.append(count)
            
            # Measure duration in seconds
            dur = timestamps[end] - timestamps[start]
            measure_durations.append(dur)

    if beats_per_measure:
        # The most frequent beat count is likely the Time Signature (e.g., 4)
        vals, counts = np.unique(beats_per_measure, return_counts=True)
        mode_time_sig = vals[np.argmax(counts)]
        avg_measure_dur = np.mean(measure_durations)
    else:
        # Fallback if no downbeats marked
        mode_time_sig = 0
        avg_measure_dur = 0

    return {
        'total_beats': len(timestamps),
        'duration_seconds': timestamps[-1] - timestamps[0],
        'bpm_median': median_bpm,
        'bpm_mean': avg_bpm,
        'bpm_std': std_bpm,
        'bpm_min': min_bpm,
        'bpm_max': max_bpm,
        'ibi_mean': avg_ibi,
        'sec_per_beat': avg_ibi,
        'measure_beats_mode': mode_time_sig,
        'sec_per_measure': avg_measure_dur,
        'consistency': std_ibi # Lower is tighter
    }

def analyze_folder_stats(folder_path):
    """
    Scans a folder for .beats files and prints aggregate statistics.
    """
    # Look for .beats and .txt files
    files = glob.glob(os.path.join(folder_path, "*.beats"))
    if not files:
        files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not files:
        print(f"No .beats or .txt files found in: {folder_path}")
        return

    print(f"Found {len(files)} beat files. Analyzing...\n")

    # Table Header
    print(f"{'Filename':<30} | {'BPM':<6} | {'Sig':<5} | {'StDev':<6} | {'Dur(s)':<7}")
    print("-" * 75)

    all_stats = []

    for fpath in files:
        fname = os.path.basename(fpath)
        beat_data = parse_beat_file(fpath)
        stats = calculate_file_stats(beat_data)
        
        if stats:
            stats['filename'] = fname
            all_stats.append(stats)
            
            # Print row
            sig_str = f"{stats['measure_beats_mode']}/4" if stats['measure_beats_mode'] > 0 else "?"
            print(f"{fname[:30]:<30} | {stats['bpm_median']:<6.1f} | {sig_str:<5} | {stats['consistency']:<6.4f} | {stats['duration_seconds']:<7.1f} | {stats['sec_per_measure']:<7.1f}")

    if not all_stats:
        print("Could not calculate stats for any files.")
        return

    # Aggregate Statistics
    print("\n" + "="*30)
    print("AGGREGATE STATISTICS")
    print("="*30)

    bpms = [s['bpm_median'] for s in all_stats]
    time_sigs = [s['measure_beats_mode'] for s in all_stats if s['measure_beats_mode'] > 0]
    durations = [s['duration_seconds'] for s in all_stats]
    consistencies = [s['consistency'] for s in all_stats]
    measure_durations = [s['sec_per_measure'] for s in all_stats]

    print(f"Total Files Processed: {len(all_stats)}")
    print(f"Total Audio Duration:  {sum(durations)/60:.1f} minutes")
    print("-" * 30)
    print("BPM Distribution:")
    print(f"  Mean:   {np.mean(bpms):.2f}")
    print(f"  Median: {np.median(bpms):.2f}")
    print(f"  Min:    {np.min(bpms):.2f}")
    print(f"  Max:    {np.max(bpms):.2f}")
    print(f"  StdDev: {np.std(bpms):.2f}")
    print("-" * 30)
    print("Time Signatures (Detected Measures):")
    if time_sigs:
        vals, counts = np.unique(time_sigs, return_counts=True)
        for v, c in zip(vals, counts):
            print(f"  {v} beats/measure: {c} files ({c/len(time_sigs)*100:.1f}%)")
    else:
        print("  No measure info detected.")
    print("-" * 30)
    print("Measure Distribution:")
    print(f"  Mean: {np.mean(measure_durations)}")
    print(f"  Median: {np.median(measure_durations)}")
    print(f"  Std: {np.std(measure_durations)}")
    print(f"  Quantiles (5%, 10%, 50%, 90%, 95%): {np.quantile(measure_durations, [0.05, 0.1, 0.5, 0.9, 0.95])}")
    print(f"  Min: {np.min(measure_durations)}")
    print(f"  Max: {np.max(measure_durations)}")
    print("-" * 30)
    print("Rhythmic Consistency (IBI StdDev):")
    print(f"  Average StdDev: {np.mean(consistencies):.4f}s")
    print(f"  Most Robotic:   {min(all_stats, key=lambda x: x['consistency'])['filename']} ({np.min(consistencies):.4f}s)")
    print(f"  Most Expressive:{max(all_stats, key=lambda x: x['consistency'])['filename']} ({np.max(consistencies):.4f}s)")


def sample_audio_beats(audio_path, beat_path, n_samples, m_beats, output_dir):
    """
    Samples any m contiguous beats, regardless of bar alignment.
    """
    print(f"Reading beats from: {beat_path}")
    beat_data = parse_beat_file(beat_path)
    timestamps = [b['time'] for b in beat_data]
    total_beats = len(timestamps)

    if total_beats < m_beats + 1:
        print(f"Error: Not enough beats in file. Needed {m_beats + 1}, found {total_beats}.")
        return

    max_start_index = total_beats - m_beats - 1
    
    if max_start_index < 0:
        print("Error: Song is too short for the requested batch size.")
        return

    possible_indices = list(range(max_start_index + 1))
    
    if n_samples > len(possible_indices):
        print(f"Warning: Requested {n_samples} samples, but only {len(possible_indices)} unique sets exist.")
        selected_indices = possible_indices
    else:
        selected_indices = random.sample(possible_indices, n_samples)

    print(f"Loading audio: {audio_path}...")
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Extracting {len(selected_indices)} clips of {m_beats} beats...")
    
    for i, start_idx in enumerate(selected_indices):
        t_start = timestamps[start_idx]
        t_end = timestamps[start_idx + m_beats]
        
        frame_start = int(t_start * sr)
        frame_end = int(t_end * sr)
        
        audio_slice = y[frame_start:frame_end]
        
        out_name = f"sample_{i+1}_beat_{start_idx+1}_to_{start_idx+m_beats}.wav"
        out_path = os.path.join(output_dir, out_name)
        
        sf.write(out_path, audio_slice, sr)
        print(f"Saved: {out_name} ({t_start:.2f}s - {t_end:.2f}s)")

    print("\nProcessing complete.")

def sample_audio_measures(audio_path, beat_path, n_samples, output_dir):
    """
    Samples full measures (Downbeat '1' to the next Downbeat '1').
    """
    print(f"Reading beats from: {beat_path}")
    beat_data = parse_beat_file(beat_path)
    
    downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
    
    if len(downbeat_indices) < 2:
        print("Error: Not enough downbeats (Beat '1') found to form a measure.")
        return

    measure_intervals = []
    for i in range(len(downbeat_indices) - 1):
        start_idx = downbeat_indices[i]
        end_idx = downbeat_indices[i+1]
        measure_intervals.append((start_idx, end_idx))
    
    possible_indices = list(range(len(measure_intervals)))
    
    if n_samples > len(possible_indices):
        selected_indices = possible_indices
    else:
        selected_indices = sorted(random.sample(possible_indices, n_samples))

    print(f"Loading audio: {audio_path}...")
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Extracting {len(selected_indices)} random measures...")
    
    for i, idx in enumerate(selected_indices):
        start_beat_idx, end_beat_idx = measure_intervals[idx]
        
        t_start = beat_data[start_beat_idx]['time']
        t_end = beat_data[end_beat_idx]['time']
        
        frame_start = int(t_start * sr)
        frame_end = int(t_end * sr)
        
        audio_slice = y[frame_start:frame_end]
        
        out_name = f"measure_{i+1}_start_{t_start:.2f}s.wav"
        out_path = os.path.join(output_dir, out_name)
        
        sf.write(out_path, audio_slice, sr)
        print(f"Saved: {out_name} (Duration: {t_end - t_start:.2f}s)")
        
    print("\nProcessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beat sampling and statistical analysis tool.")
    
    # audio_file is now optional (nargs='?') because 'stats' mode doesn't need it.
    parser.add_argument("audio_file", nargs='?', help="Path to input audio file. (Ignored in 'stats' mode)")
    
    parser.add_argument("--mode", choices=['beats', 'measures', 'stats'], default='beats', 
                        help="Action mode: 'beats' (random clips), 'measures' (random bars), 'stats' (analyze folder).")
    
    parser.add_argument("--n", type=int, default=5, help="Number of samples to extract")
    parser.add_argument("--m", type=int, default=4, help="Number of beats per sample (only used in 'beats' mode)")
    parser.add_argument("--output", default="beat_samples", help="Output directory")

    args = parser.parse_args()
    
    if args.mode == 'stats':
        # Check if user passed a folder
        beat_path = '/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_beats'
        if os.path.isdir(beat_path):
            analyze_folder_stats(beat_path)
        else:
            print(f"Error: Mode is 'stats' but '{args.beat_path}' is not a directory.")
    elif args.mode == 'measures':
        if not args.audio_file:
            print("Error: Audio file required for 'measures' mode.")
        else:
            sample_audio_measures(args.audio_file, args.beat_path, args.n, args.output)
    else: # mode == beats
        if not args.audio_file:
            print("Error: Audio file required for 'beats' mode.")
        else:
            sample_audio_beats(args.audio_file, args.beat_path, args.n, args.m, args.output)