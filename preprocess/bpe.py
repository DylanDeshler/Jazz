"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""
"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

import numpy as np

# -----------------------------------------------------------------------------
# Helper functions

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): 
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def render_token(t):
    # Pretty print a token. Since we are using integers, 
    # we just return the string representation of the list of ints.
    return str(t)

# -----------------------------------------------------------------------------
# The base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        self.merges = {} # (int, int) -> int
        # distinct integers -> list of integers
        self.vocab = {}  # int -> list[int]
        self.special_tokens = {} # str -> int

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        # We assume the base vocabulary is the full uint8 range (0-255)
        # for robust loading, even if training only saw a subset.
        vocab = {idx: [idx] for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def save(self, file_prefix):
        """
        Saves the model to .model and .vocab files.
        """
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe_int v1\n") # Modified version header
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe_int v1"
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

# -----------------------------------------------------------------------------
# The Integer Tokenizer

class IntegerTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, inputs, vocab_size, verbose=False):
        """
        Train the tokenizer on a numpy array of uint8 integers.
        """
        # Ensure input is a flat list of integers
        if isinstance(inputs, np.ndarray):
            ids = inputs.flatten().tolist()
        else:
            ids = list(inputs)
        
        # 1. Initialize vocab to the set of integers in the input
        unique_ints = sorted(list(set(ids)))
        vocab = {idx: [idx] for idx in unique_ints}
        
        # We ensure we have enough space for merges.
        # Note: In standard BPE, we often reserve 0-255, so merges start at 256.
        # This prevents collisions between raw values (0-255) and merged tokens.
        # Even if the input only contains [1, 5], we start merges at 256 for safety.
        num_merges = vocab_size - len(unique_ints)
        
        # However, usually vocab_size implies TOTAL tokens. 
        # If we want strictly `vocab_size` total tokens:
        # We need to perform (vocab_size - len(unique_ints)) merges.
        if num_merges < 0:
            raise ValueError(f"vocab_size {vocab_size} is too small for distinct inputs {len(unique_ints)}")

        merges = {}
        
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break # no more pairs to merge

            pair = max(stats, key=stats.get)
            
            # Mint new token ID. 
            # We start at 256 to keep distinct from uint8 raw values.
            idx = 256 + i 
            
            ids = merge(ids, pair, idx)
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        # given ids (list of integers), return original list of integers
        # flattened from the vocab sequences
        res = []
        for idx in ids:
            res.extend(self.vocab[idx])
        return res # Returns list of ints (can be cast to bytes or np.array by user)

    def encode(self, inputs):
        # given a list/array of integers, return the token ids
        if isinstance(inputs, np.ndarray):
            ids = inputs.flatten().tolist()
        else:
            ids = list(inputs)

        while len(ids) >= 2:
            stats = get_stats(ids)
            # find the pair with the lowest merge index
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # nothing else can be merged
            
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

if __name__ == '__main__':
    data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_large_actions_train.bin', dtype=np.uint8, mode='r', shape=(10000 * 2**12, 16))
    data = data.flatten()
    
    tokenizer = IntegerTokenizer()
    tokenizer.train(data, 512, verbose=True)
    tokenizer.save('LAPA_B_FS_64_noagg_BPE')