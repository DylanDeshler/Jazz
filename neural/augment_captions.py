import json
import random
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# --- CONFIGURATION ---
INPUT_JSONL = "/home/dylandeshler/Jazz/preprocess/final_llm_captions.jsonl"
OUTPUT_JSONL = "/home/dylandeshler/Jazz/preprocess/final_llm_captions_expanded.jsonl"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
NUM_VARIATIONS = 5

def build_prompt(caption):
    """Formats the prompt using Llama 3's native chat template."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a strict data augmentation script. You will receive a song caption and produce generations that could be used as prompts to generate a song with the given caption. "
        f"Generate exactly {NUM_VARIATIONS} distinct, paraphrased, reworded, reordered variations of this caption. "
        "Each generation should roughly match the length of the reference caption. "
        "When the reference caption contains BPM or key you MUST maintain at least one of them in your generations, if not both. "
        "Do not change the underlying meaning or attributes. "
        "Do not add or make up additional details that are not in reference caption. "
        "Output ONLY a valid JSON object containing a single key 'variations' which maps to a list of strings."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Caption: {caption}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

def main():
    print("Loading dataset...")
    raw_data = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))

    raw_data = raw_data[:10]
    
    # NEW: Filter data upfront so data, prompts, and outputs track 1-to-1 perfectly
    data = []
    for item in raw_data:
        llm_output = item.get('llm_output', [])
        if isinstance(llm_output, list) or llm_output is None:
            print('skipping bad data')
            continue
        data.append(item)

    print(f"Loaded {len(data)} valid items. Preparing prompts...")
    
    # Extract short, medium, and long captions
    prompts = []
    for item in data:
        llm_output = item['llm_output']
        prompts.append(build_prompt(llm_output.get("short_caption", "")))
        prompts.append(build_prompt(llm_output.get("medium_caption", "")))
        prompts.append(build_prompt(llm_output.get("long_caption", "")))

    print("Initializing vLLM Engine...")
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.95, 
    )

    # Force the LLM to output our exact JSON structure using guided decoding
    json_schema = {
        "type": "object",
        "properties": {
            "variations": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": NUM_VARIATIONS,
                "maxItems": NUM_VARIATIONS
            }
        },
        "required": ["variations"]
    }
    
    sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=256,
        structured_outputs=StructuredOutputsParams(json=json.dumps(json_schema))
    )

    print("Generating variations (this will go very fast)...")
    outputs = llm.generate(prompts, sampling_params)

    print("Parsing outputs and writing to disk...")
    out_idx = 0
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="Saving"):
            llm_output = item['llm_output']
            
            # Extract references
            ref_short = llm_output.get("short_caption", "")
            ref_med = llm_output.get("medium_caption", "")
            ref_long = llm_output.get("long_caption", "")
            
            # 1. Handle Short
            try:
                short_json = json.loads(outputs[out_idx].outputs[0].text)
                short_vars = short_json.get("variations", [ref_short] * NUM_VARIATIONS)
            except (json.JSONDecodeError, IndexError):
                short_vars = [ref_short] * NUM_VARIATIONS
            out_idx += 1
            
            # 2. Handle Medium
            try:
                med_json = json.loads(outputs[out_idx].outputs[0].text)
                med_vars = med_json.get("variations", [ref_med] * NUM_VARIATIONS)
            except (json.JSONDecodeError, IndexError):
                med_vars = [ref_med] * NUM_VARIATIONS
            out_idx += 1
            
            # 3. Handle Long
            try:
                long_json = json.loads(outputs[out_idx].outputs[0].text)
                long_vars = long_json.get("variations", [ref_long] * NUM_VARIATIONS)
            except (json.JSONDecodeError, IndexError):
                long_vars = [ref_long] * NUM_VARIATIONS
            out_idx += 1
            
            # NEW: Re-construct original nested structure with the reference as element 0
            new_item = {
                "file_path": item.get("file_path", ""),
                "llm_output": {
                    "short_caption": [ref_short] + short_vars,
                    "medium_caption": [ref_med] + med_vars,
                    "long_caption": [ref_long] + long_vars
                }
            }
            
            f.write(json.dumps(new_item) + "\n")

    print(f"Successfully saved expanded dataset to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()