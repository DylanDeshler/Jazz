import torch
import glob
import argparse
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import os
import numpy as np
import librosa

def get_musical_key(wav_path):
    try:
        # Load audio (downmix to mono, resample to 22050Hz)
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        
        # Extract Pitch Class Profile (Chromagram)
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_vals = np.sum(chromagram, axis=1)
        
        # Standard Krumhansl-Schmuckler major/minor templates
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        # Calculate Pearson correlation for all 12 shifts (12 keys)
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
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (0, 1, 2, or 3)")
    parser.add_argument("--world_size", type=int, default=4, help="Total number of GPUs")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of audio files to process at once")
    args = parser.parse_args()
    
    TARGET_SIG = 4
    rate = 16000

    device = f"cuda:{args.gpu}"
    model_id = "nvidia/audio-flamingo-next-think-hf"

    print(f"Loading {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, 
        device_map=device, 
        torch_dtype=torch.bfloat16
    )

    # 1. Load the dataset WITHOUT sorting to strictly preserve glob's native order
    all_wavs = glob.glob('/data/wavs/*.wav')
    
    # Slice the original glob order into contiguous chunks
    chunk_size = len(all_wavs) // args.world_size
    start_idx = args.gpu * chunk_size
    end_idx = start_idx + chunk_size if args.gpu < args.world_size - 1 else len(all_wavs)
    
    my_wavs = all_wavs[start_idx:end_idx]
    print(f"GPU {args.gpu} processing {len(my_wavs)} files...")

    text_prompt = "Summarize the track with precision: mention its musical style, BPM, key, arrangement, production choices, and the emotions or story it conveys."
    # text_prompt = "Write a short, detailed, and concise caption for this track without mentioning BPM, length, chords, or lyrics."
    
    # 2. Open an output file specific to this GPU to save incrementally
    output_filename = f"captions_part_{args.gpu}.jsonl"
    
    with open(output_filename, "w", encoding="utf-8") as outfile:
        # 3. Process in batches
        for i in tqdm(range(0, len(my_wavs), args.batch_size), desc=f"GPU {args.gpu}"):
            batch_wavs = my_wavs[i:i + args.batch_size]
            
            conversations = []
            batch_bpms = []
            batch_keys = []
            for wav in batch_wavs:
                beat_path = os.path.join('/data/beats', os.path.basename(wav))
                beat_data = parse_beat_file(beat_path)
                downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
                
                bpms = []
                for i in range(len(downbeat_indices) - 1):    
                    start_idx = downbeat_indices[i]
                    end_idx = downbeat_indices[i+1]
                    
                    t_start = beat_data[start_idx]['time']
                    t_end = beat_data[end_idx]['time']
                    
                    frame_start = int(t_start * rate)
                    frame_end = int(t_end * rate)
                    
                    if frame_end > len(wav):
                        break
                    
                    duration_sec = (frame_end - frame_start) / rate
                    instant_bpm = (TARGET_SIG / duration_sec) * 60
                    
                    bpms.append(instant_bpm)
                batch_bpms.append(np.mean(bpms))
                
                batch_keys.append(get_musical_key(wav))
                
                conversations.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {"type": "audio", "path": wav},
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
                    max_new_tokens=2000, 
                    do_sample=False      
                )

            # 4. Decode the batch and save the path + caption immediately
            input_length = inputs.input_ids.shape[1]
            for j, output in enumerate(outputs):
                generated_tokens = output[input_length:]
                decoded_output = processor.decode(generated_tokens, skip_special_tokens=True)
                
                # Create a dictionary for the result
                result = {
                    "file_path": batch_wavs[j],
                    "caption": decoded_output.strip(),
                    "bpm": batch_bpms[j],
                    "key": batch_keys[j],
                }
                
                # Write to the file as a JSON string and flush to disk
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()

if __name__ == "__main__":
    main()