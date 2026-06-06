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
NUM_VARIATIONS = 10
NUM_SELECTIONS = 5

def build_generation_prompt(caption):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a strict text augmentation script for music generation prompts. "
        "Your job is to generate highly diverse, reworded, and structurally reordered variations of a reference caption.\n\n"
        "CRITICAL RULES:\n"
        "1. NEVER copy the reference caption verbatim. Every variation must use different phrasing, word orders, or sentence structures.\n"
        "2. DO NOT SUMMARIZE. Every variation must retain the descriptive detail, mood, instruments, historical context, and stylistic adjectives found in the reference.\n"
        "3. ABSOLUTE CONSTRAINTS. If the reference contains a BPM or Key, you MUST preserve at least one of them in every single variation.\n"
        "4. MATCH THE LENGTH. Keep the overall length and core attributes identical, but change the syntax drastically.\n\n"
        "5. ROTATE CLAUSES. Achieve diversity by changing the sentence structures, moving technical details around, or flipping the order of descriptions.\n"
        "EXAMPLE:\n"
        "Reference: 'A slow lo-fi hip hop beat with a dusty piano chord progression and smooth sax. 80 BPM. Key: C minor.'\n"
        "Expected Output:\n"
        '{\n  "variations": [\n'
        '    "Dusty piano chords and a smooth saxophone at 80 BPM lead this slow-tempo lo-fi hip hop track in C minor.",\n'
        '    "80 BPM lo-fi hip hop instrumental featuring a smooth sax melody accompanied by a dusty piano progression in C minor.",\n'
        '    "In C minor, a relaxing and slow lo-fi hip hop groove highlighting smooth saxophone lines over dusty piano chords. 80 BPM."\n'
        '  ]\n}\n'
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Reference Caption: {caption}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

def build_selection_prompt(reference, pool_list):
    """Stage 2: Acts as a strict judge to downselect to the best unique variations."""
    pool_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(pool_list)])
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are an expert data curation judge. Your job is to select the best subset of variations from a provided pool.\n"
        f"Review the Reference Caption and select exactly {NUM_SELECTIONS} variations from the Candidate Pool that meet these criteria:\n"
        "1. ACCURACY: They must perfectly retain 100% of the information, exact numbers, BPM, and musical key from the reference.\n"
        "2. DIVERSITY: They must be structurally and linguistically unique from the reference and from each other.\n"
        "3. QUALITY: Discard any candidates that are verbatim copies of the reference or are heavily truncated summaries.\n"
        f"Output ONLY a valid JSON object containing a single key 'selected' which maps to a list of exactly {NUM_SELECTIONS} strings chosen from the pool."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Reference Caption: {reference}\n\n"
        f"Candidate Pool:\n{pool_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
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

    print(f"Loaded {len(data)} valid items.")
    
    print("Stage 1: Generating candidate pools...")
    gen_prompts = []
    for item in data:
        llm_output = item['llm_output']
        gen_prompts.append(build_generation_prompt(llm_output.get("short_caption", "")))
        gen_prompts.append(build_generation_prompt(llm_output.get("medium_caption", "")))
        gen_prompts.append(build_generation_prompt(llm_output.get("long_caption", "")))
    
    print("Initializing vLLM Engine...")
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.95, 
    )

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
        temperature=1.0, 
        top_p=0.95,
        max_tokens=4096,
        structured_outputs=StructuredOutputsParams(json=json.dumps(json_schema))
    )

    print("Generating variations...")
    outputs = llm.generate(gen_prompts, sampling_params)
    
    print("Stage 2: Preparing verification prompts...")
    select_prompts = []
    references = []
    out_idx = 0
    
    for item in data:
        llm_output = item['llm_output']
        for key in ["short_caption", "medium_caption", "long_caption"]:
            ref_text = llm_output.get(key, "")
            references.append(ref_text)
            
            try:
                pool = json.loads(outputs[out_idx].outputs[0].text).get("variations", [])
            except Exception:
                pool = [ref_text] * NUM_VARIATIONS
                
            select_prompts.append(build_selection_prompt(ref_text, pool))
            out_idx += 1

    print("Stage 2: Running LLM-as-a-Judge downselection...")
    select_schema = {
        "type": "object",
        "properties": {"selected": {"type": "array", "items": {"type": "string"}, "minItems": NUM_SELECTIONS, "maxItems": NUM_SELECTIONS}},
        "required": ["selected"]
    }
    
    select_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        structured_outputs=StructuredOutputsParams(json=json.dumps(select_schema))
    )
    
    outputs = llm.generate(select_prompts, select_params)

    print("Parsing outputs and writing to disk...")
    out_idx = 0
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="Saving"):
            llm_output = item['llm_output']
            captions_data = {}
            
            for key in ["short_caption", "medium_caption", "long_caption"]:
                ref_text = llm_output.get(key, "")
                try:
                    selected_vars = json.loads(outputs[out_idx].outputs[0].text).get("selected", [])
                except Exception:
                    selected_vars = [ref_text] * NUM_SELECTIONS
                
                # Prepend the reference token to build the final list format
                captions_data[key] = [ref_text] + selected_vars
                out_idx += 1
            
            new_item = {
                "file_path": item.get("file_path", ""),
                "llm_output": captions_data
            }
            f.write(json.dumps(new_item) + "\n")

    print(f"Successfully saved expanded dataset to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()