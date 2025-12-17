"""
Requires installing beat_this from https://github.com/CPJKU/beat_this
"""

import os
import glob
from tqdm import tqdm

from beat_this.inference import File2Beats
from beat_this.utils import save_beat_tsv

paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/*.wav')
file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)

for audio_path in tqdm(paths):
    beats, downbeats = file2beats(audio_path)

    outpath = audio_path.replace('jazz_data_16000_full_clean', 'jazz_data_16000_full_clean_beats').replace('.wav', '.beats')
    save_beat_tsv(beats, downbeats, outpath)