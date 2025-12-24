import requests
from io import BytesIO
import os
import pickle
from tqdm import tqdm

import librosa

import soundfile as sf

rate = 16000
out_dir = '/home/ubuntu/base/Data/wavs'
os.makedirs(out_dir, exist_ok=True)

cards = pickle.load(open('/home/ubuntu/base/Data/JazzSet.0.9.pkl', "rb"))[6:]

for card in tqdm(cards):
    mp3_url = card['URLS'][0]['FILE']
    out_name = '-'.join(mp3_url.split('/')[-2:]).replace('.mp3', '.wav')
    out_path = os.path.join(out_dir, out_name)

    try:
        response = requests.get(mp3_url, timeout=10)
        response.raise_for_status() 
        
        mp3_buffer = BytesIO(response.content)
        y, sr = librosa.load(mp3_buffer, sr=rate)

        sf.write(out_path, y, rate)

    except Exception as e:
        print(f"Error processing {mp3_url}: {e}")
        continue
