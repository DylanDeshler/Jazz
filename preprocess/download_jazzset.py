import requests
from io import BytesIO
import os
import pickle
from tqdm import tqdm

import librosa
from pydub import AudioSegment
import numpy as np
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
        response = requests.get(mp3_url, timeout=20)
        response.raise_for_status()

        # 1. Load the bytes into pydub
        # pydub is very reliable at identifying MP3 headers from a stream
        audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")

        # 2. Convert to float32 NumPy array (which librosa uses)
        # We also normalize it to the range [-1.0, 1.0]
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Handle Stereo to Mono if necessary
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        # Normalize bit depth (pydub loads as integers)
        samples /= (2**15) 

        # 3. Resample using librosa if needed
        if audio.frame_rate != rate:
            y = librosa.resample(samples, orig_sr=audio.frame_rate, target_sr=rate)
        else:
            y = samples

        # 4. Save
        sf.write(out_path, y, rate)

    except Exception as e:
        print(f"Error processing {mp3_url}: {e}")
