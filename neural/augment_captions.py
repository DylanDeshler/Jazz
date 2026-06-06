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

def shuffle_short_tags(tag_string, num_variations=5):
    """Instantly shuffles comma-separated tags using pure Python."""
    tags = [t.strip() for t in tag_string.split(',') if t.strip()]
    if not tags:
        return [tag_string] * num_variations
        
    variations = []
    for _ in range(num_variations):
        random.shuffle(tags)
        variations.append(", ".join(tags))
    return variations

def build_prompt(caption):
    """Formats the prompt using Llama 3's native chat template."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a strict data augmentation script. You will receive a song caption. "
        f"Generate exactly {NUM_VARIATIONS} distinct, paraphrased variations of this caption. "
        "Do not change the underlying meaning or attributes. "
        "Output ONLY a valid JSON object containing a single key 'variations' which maps to a list of strings."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Caption: {caption}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

def main():
    print("Loading dataset...")
    data = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} items. Preparing prompts...")
    
    # Extract medium and long captions to process in one massive batch
    prompts = []
    for item in data[:10]:
        prompts.append(build_prompt(item.get("medium", "")))
        prompts.append(build_prompt(item.get("long", "")))

    print("Initializing vLLM Engine...")
    # gpu_memory_utilization=0.95 allocates almost all your 46GB to the engine.
    # The 8B weights take ~16GB, leaving ~28GB for a massive KV Cache = extreme speed.
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
    # vLLM handles the continuous batching automatically
    outputs = llm.generate(prompts, sampling_params)
    print(outputs)

    print("Parsing outputs and writing to disk...")
    # Map the linear outputs back to our original dictionary structure
    out_idx = 0
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="Saving"):
            new_item = {}
            
            # 1. Handle Short (Pure Python)
            new_item["short"] = shuffle_short_tags(item.get("short", ""), NUM_VARIATIONS)
            
            # 2. Handle Medium (LLM Output)
            try:
                med_json = json.loads(outputs[out_idx].outputs[0].text)
                new_item["medium"] = med_json.get("variations", [item.get("medium")] * NUM_VARIATIONS)
            except json.JSONDecodeError:
                new_item["medium"] = [item.get("medium")] * NUM_VARIATIONS # Fallback on failure
            out_idx += 1
            
            # 3. Handle Long (LLM Output)
            try:
                long_json = json.loads(outputs[out_idx].outputs[0].text)
                new_item["long"] = long_json.get("variations", [item.get("long")] * NUM_VARIATIONS)
            except json.JSONDecodeError:
                new_item["long"] = [item.get("long")] * NUM_VARIATIONS # Fallback on failure
            out_idx += 1
            
            f.write(json.dumps(new_item) + "\n")

    print(f"Successfully saved expanded dataset to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()