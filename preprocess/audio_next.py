import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import pickle
import requests
import tempfile
import soundfile as sf
import librosa

model_id = "nvidia/audio-flamingo-next-think-hf"

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

def process_card(card):
    if not isinstance(card, dict):
        return f"Skipping invalid entry: {card} (not a dictionary)"
    try:
        mp3_url = card['URLS'][0]['FILE']
        # out_name = '-'.join(mp3_url.split('/')[-2:]).replace('.mp3', '.wav')
        # out_path = os.path.join(out_dir, out_name)

        # Skip if already exists (useful for resuming crashes)
        # if os.path.exists(out_path):
        #     return True

        # Download the file
        response = requests.get(mp3_url, timeout=30)
        response.raise_for_status()

        # Save to temp file and process
        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_mp3:
            temp_mp3.write(response.content)
            temp_mp3.flush()
            
            y, sr = librosa.load(temp_mp3.name)
            sf.write('test_input.wav', y, sr)
        
        return True
    except Exception as e:
        return f"Error with {card.get('URLS', [{}])[0].get('FILE')}: {e}"

# Load data
cards = pickle.load(open('/data/JazzSet.0.9.pkl', "rb"))[6:]

for card in cards:
    process_card(card)
    break

# 1. Define the audio source (can be a local file path or a direct URL)
audio_path = "test_input.wav" # Replace with your audio file in Colab

# 2. Define the prompt
# Explicitly asking for reasoning triggers the model's Chain-of-Thought
text_prompt = "Reason step by step with timestamps, then give the final answer: What instruments are introduced in this track, and how does the tempo change?"
text_prompt = "Summarize the track with precision: mention its musical style, BPM, key, arrangement, production choices, and the emotions or story it conveys."

# 3. Format the conversation for the processor
# Audio Flamingo uses a standardized multimodal chat template
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt},
            {"type": "audio", "path": audio_path},
        ],
    }
]

# 4. Apply the chat template and process inputs
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

# 5. Generate the reasoning trace and response
print("Generating response (this may take a moment)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1500, # Set high to allow room for the <think> blocks
        do_sample=False      # Greedy decoding is recommended for logical reasoning
    )

# 6. Decode and print the output
# We slice the output to ignore the input prompt tokens and only print the new generation
input_length = inputs.input_ids.shape[1]
generated_tokens = outputs[0][input_length:]
decoded_output = processor.decode(generated_tokens, skip_special_tokens=True)

print("\n--- Model Output ---")
print(decoded_output)