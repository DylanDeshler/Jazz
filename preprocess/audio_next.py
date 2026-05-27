import torch
import glob
import argparse
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (0, 1, 2, or 3)")
    parser.add_argument("--world_size", type=int, default=4, help="Total number of GPUs")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of audio files to process at once")
    args = parser.parse_args()

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
    text_prompt = "Write a short, detailed, and concise caption for this track without mentioning BPM, length, chords, or lyrics."
    
    # 2. Open an output file specific to this GPU to save incrementally
    output_filename = f"captions_part_{args.gpu}.jsonl"
    
    with open(output_filename, "w", encoding="utf-8") as outfile:
        # 3. Process in batches
        for i in tqdm(range(0, len(my_wavs), args.batch_size), desc=f"GPU {args.gpu}"):
            batch_wavs = my_wavs[i:i + args.batch_size]
            
            conversations = []
            for wav in batch_wavs:
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
                processor_kwargs={
                    "return_dict": True,
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
                    "caption": decoded_output.strip()
                }
                
                # Write to the file as a JSON string and flush to disk
                outfile.write(json.dumps(result) + "\n")
                outfile.flush()

if __name__ == "__main__":
    main()