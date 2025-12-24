import requests
import os
import pickle
import librosa
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import tempfile

# Configuration
rate = 16000
out_dir = '/home/ubuntu/base/Data/wavs'
os.makedirs(out_dir, exist_ok=True)
MAX_WORKERS = 40  # Number of simultaneous downloads/processes

def process_card(card):
    try:
        mp3_url = card['URLS'][0]['FILE']
        out_name = '-'.join(mp3_url.split('/')[-2:]).replace('.mp3', '.wav')
        out_path = os.path.join(out_dir, out_name)

        # Skip if already exists (useful for resuming crashes)
        if os.path.exists(out_path):
            return True

        # Download the file
        response = requests.get(mp3_url, timeout=30)
        response.raise_for_status()

        # Save to temp file and process
        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_mp3:
            temp_mp3.write(response.content)
            temp_mp3.flush()
            
            y, sr = librosa.load(temp_mp3.name, sr=rate)
            sf.write(out_path, y, rate)
        
        return True
    except Exception as e:
        return f"Error with {card.get('URLS', [{}])[0].get('FILE')}: {e}"

# Load data
cards = pickle.load(open('/home/ubuntu/base/Data/JazzSet.0.9.pkl', "rb"))[6:]

# Execute in parallel
print(f"Starting parallel processing with {MAX_WORKERS} workers...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # list() forces the tqdm to track the generator progress
    results = list(tqdm(executor.map(process_card, cards), total=len(cards)))

# Optional: Print errors
errors = [r for r in results if r is not True]
if errors:
    print(f"\nFinished with {len(errors)} errors. Example: {errors[0]}")