import torch
import glob
import argparse
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import os
import numpy as np
import librosa
import tempfile
import soundfile as sf

def get_musical_key(wav_path, rate=16000, hop_length=1024):
    try:
        y, _ = librosa.load(wav_path, sr=rate, mono=True)
        
        if len(y) == 0 or np.max(np.abs(y)) == 0:
            return "Unknown"
            
        # Extract Pitch Class Profile (Chromagram)
        chromagram = librosa.feature.chroma_cqt(y=y, sr=rate, hop_length=hop_length)
        chroma_vals = np.sum(chromagram, axis=1)
        
        if np.std(chroma_vals) == 0:
            return "Unknown"
            
        # Standard Krumhansl-Schmuckler major/minor templates
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        with np.errstate(all='ignore'):
            maj_corrs = [np.corrcoef(chroma_vals, np.roll(maj_profile, i))[0, 1] for i in range(12)]
            min_corrs = [np.corrcoef(chroma_vals, np.roll(min_profile, i))[0, 1] for i in range(12)]
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Find the best match
        if max(maj_corrs) > max(min_corrs):
            return f"{keys[np.argmax(maj_corrs)]} major"
        else:
            return f"{keys[np.argmax(min_corrs)]} minor"
            
    except Exception as e:
        return "Unknown"

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
                            
                            # found an issue where 4/4 is frequently being annotated as 8/4 this fixes it and safe because were only annotating 4/4 songs
                            if bn > 0:
                                bn = ((bn - 1) % 4) + 1
                        except ValueError:
                            pass
                    
                    beat_data.append({'time': ts, 'beat': bn})
                except ValueError:
                    continue
    
    return beat_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU ID to use (e.g., 1, 2, or 3)")
    parser.add_argument("--rank", type=int, default=None, help="Logical worker rank (0, 1, or 2)")
    parser.add_argument("--world_size", type=int, default=3, help="Total number of active workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of audio files to process at once")
    args = parser.parse_args()
    
    if args.rank is None:
        args.rank = args.gpu
    print(f'GPU: {args.gpu} RANK: {args.rank} WORLD SIZE: {args.world_size} BATCH SIZE: {args.batch_size}')
    
    TARGET_SIG = 4
    rate = 16000
    OFFSET = 0
    MAX_DURATION = 60 * 4

    device = f"cuda:{args.gpu}"
    model_id = "nvidia/audio-flamingo-next-think-hf"

    print(f"Loading {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, 
        device_map=device, 
        torch_dtype=torch.bfloat16
    )

    # Load the dataset WITHOUT sorting to strictly preserve glob's native order
    all_wavs = glob.glob('/data/wavs/*.wav')
    
    # Slice the original glob order into contiguous chunks
    chunk_size = len(all_wavs) // args.world_size
    start_idx = args.rank * chunk_size
    end_idx = start_idx + chunk_size if args.rank < args.world_size - 1 else len(all_wavs)
    
    my_wavs = all_wavs[start_idx:end_idx]
    
    # Check for previously processed files and filter them out
    output_filename = f"captions_part_{args.rank}.jsonl"
    processed_files = set()
    
    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_files.add(data["file_path"])
                except json.JSONDecodeError:
                    # Ignore corrupted lines from a hard crash
                    pass
    
    original_len = len(my_wavs)
    my_wavs = [wav for wav in my_wavs if wav not in processed_files]
    
    print(f"GPU {args.gpu} rank {args.rank} targeting {original_len} files.")
    if len(processed_files) > 0:
        print(f"Skipping {original_len - len(my_wavs)} already processed files.")
    print(f"Files remaining to process: {len(my_wavs)}")

    text_prompt = "Write a short, detailed, and concise caption for this track without mentioning BPM, length, chords, or lyrics."
    
    if len(my_wavs) == 0:
        print("All files for this rank have been processed! Exiting.")
        return

    # Open the output file in APPEND ("a") mode so we don't overwrite previous runs
    with open(output_filename, "a", encoding="utf-8") as outfile:
        for i in tqdm(range(0, len(my_wavs), args.batch_size), desc=f"GPU {args.gpu}"):
            batch_wavs = my_wavs[i:i + args.batch_size]
            
            conversations = []
            batch_bpms = []
            batch_keys = []
            temp_files = []
            
            for wav in batch_wavs:
                y, _ = librosa.load(wav, sr=rate, offset=OFFSET, duration=MAX_DURATION)
                
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(temp_wav.name, y, rate)
                temp_files.append(temp_wav.name)
                
                wav_len = len(y)
                print(wav_len)
                
                beat_path = os.path.join('/data/beats', os.path.basename(wav))
                bpms = []
                if os.path.exists(beat_path):
                    beat_data = parse_beat_file(beat_path)
                    downbeat_indices = [k for k, b in enumerate(beat_data) if b['beat'] == 1]
                    
                    for k in range(len(downbeat_indices) - 1):    
                        start_idx = downbeat_indices[k]
                        end_idx = downbeat_indices[k+1]
                        
                        t_start = beat_data[start_idx]['time']
                        t_end = beat_data[end_idx]['time']
                        
                        frame_start = int(t_start * rate)
                        frame_end = int(t_end * rate)
                        
                        if frame_end > wav_len:
                            break
                        
                        duration_sec = (frame_end - frame_start) / rate
                        if duration_sec > 0: # Prevent division by zero
                            instant_bpm = (TARGET_SIG / duration_sec) * 60
                            bpms.append(instant_bpm)
                            
                batch_bpms.append(np.median(bpms) if bpms else 0)
                batch_keys.append(get_musical_key(wav, rate=rate))
                
                conversations.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {"type": "audio", "path": temp_wav.name},
                        ],
                    }
                ])

            inputs = processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                processor_kwargs={
                    "padding": True
                }
            ).to(device)

            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(model.dtype)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1000, 
                    do_sample=False      
                )

            # Decode the batch and save the path + caption immediately
            input_length = inputs.input_ids.shape[1]
            for j, output in enumerate(outputs):
                generated_tokens = output[input_length:]
                decoded_output = processor.decode(generated_tokens, skip_special_tokens=True)
                
                # Create a dictionary for the result
                result = {
                    "file_path": batch_wavs[j],
                    "caption": decoded_output.strip().replace('\u2011', '-'),
                    "bpm": batch_bpms[j],
                    "key": batch_keys[j],
                }
                
                # Write to the file as a JSON string and flush to disk
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()
            
            for temp_file in temp_files:
                os.remove(temp_file)

if __name__ == "__main__":
    main()