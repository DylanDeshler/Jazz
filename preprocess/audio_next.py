import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import glob
from tqdm import tqdm

model_id = "nvidia/audio-flamingo-next-think-hf"
# model_id = "nvidia/music-flamingo"

print(f"Loading {model_id}...")

# Load processor
processor = AutoProcessor.from_pretrained(model_id)

# Load model in bfloat16 to save memory, mapping automatically to the GPU
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

print("Model loaded successfully!")

wavs = glob.glob('/data/wavs/*.wav')
for wav in tqdm(wavs):

    # Explicitly asking for reasoning triggers the model's Chain-of-Thought
    text_prompt = "Summarize the track with precision: mention its musical style, BPM, key, arrangement, production choices, and the emotions or story it conveys. Do not mention BPM, length, chords, or lyrics."
    text_prompt = "Write a short, detailed, and concise caption for this track without mentioning BPM, length, chords, or lyrics."

    # Audio Flamingo uses a standardized multimodal chat template
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "audio", "path": wav},
            ],
        }
    ]

    # Apply the chat template and process inputs
    print("Processing inputs...")
    inputs = processor.apply_chat_template(
        [conversation],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    # Ensure audio features are in the correct dtype (bfloat16)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)

    # Generate the reasoning trace and response
    print("Generating response (this may take a moment)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500, # Set high to allow room for the <think> blocks
            do_sample=False      # Greedy decoding is recommended for logical reasoning
        )

    # We slice the output to ignore the input prompt tokens and only print the new generation
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    decoded_output = processor.decode(generated_tokens, skip_special_tokens=True)

    print("\n--- Model Output ---")
    print(decoded_output)