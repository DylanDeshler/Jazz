"""
Requires installing beat_this from https://github.com/CPJKU/beat_this
"""

import os
import glob
import multiprocessing as mp
from tqdm import tqdm

from beat_this.inference import File2Beats
from beat_this.utils import save_beat_tsv

worker_model = None
out_dir = '/home/ubuntu/Data/beats'

def init_worker():
    """
    Initializer function called once per worker process.
    Loads the model into VRAM for this specific worker.
    """
    global worker_model
    worker_model = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)

def process_audio(audio_path):
    """
    The target function executed by the worker pool for each file.
    """
    name = os.path.basename(audio_path)
    outpath = os.path.join(out_dir, name)
    
    if os.path.exists(outpath):
        return True
    
    try:
        beats, downbeats = worker_model(audio_path)
        save_beat_tsv(beats, downbeats, outpath)
        return True
    except Exception as e:
        return f"Error on {name}: {e}"

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    paths = glob.glob('/home/ubuntu/Data/wavs/*')
    os.makedirs(out_dir, exist_ok=True)
    NUM_WORKERS = 40
    
    print(f"Processing {len(paths)} files with {NUM_WORKERS} parallel workers...")
    
    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap_unordered(process_audio, paths), total=len(paths)))
        
    errors = [r for r in results if isinstance(r, str)]
    if errors:
        print(f"\nCompleted with {len(errors)} errors:")
        for err in errors[:10]:
            print(err)