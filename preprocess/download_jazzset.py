from pydub import AudioSegment
import requests
from io import BytesIO
import os
import pickle
from tqdm import tqdm

import librosa

import numpy as np
import soundfile as sf

rate = 16000
out_dir = '/home/ubuntu/base/Data/wavs'
os.makedirs(out_dir, exist_ok=True)

cards = pickle.load(open('/Users/dylan.d/Documents/research/music/JazzSet.0.9.pkl', "rb"))[6:]

for card in tqdm(cards):
    mp3_url = card['URLS'][0]['FILE']
    out_url = '-'.join(mp3_url.split('/')[-2:])
    out_url = out_url.replace('.mp3', '.wav')
    out_url = os.path.join(out_dir, out_url)

    try:
        response = requests.get(mp3_url)
        mp3_audio = BytesIO(response.content)

        y, sr = librosa.load(mp3_audio, sr=None)
        y = librosa.resample(y, orig_sr=sr, target_sr=rate)

        sf.write(out_url, y, rate)

    except Exception as e:
        print(e)
        continue
