import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import csv
import numpy as np
import soundfile as sf
from torchinfo import summary

class EMATracker:
    """Tracks the exponential moving average of gradient norms."""
    def __init__(self, beta=0.99):
        self.beta = beta
        self.emas = {}

    def update(self, task_name, current_norm):
        if task_name not in self.emas:
            self.emas[task_name] = current_norm
        else:
            self.emas[task_name] = self.beta * self.emas[task_name] + (1 - self.beta) * current_norm
        return self.emas[task_name]

class GradientBalancer(torch.autograd.Function):
    """
    Intercepts the backward pass to scale gradients based on their EMA norm.
    """
    @staticmethod
    def forward(ctx, x, ema_tracker, task_name, task_weight):
        # Store objects for the backward pass
        ctx.ema_tracker = ema_tracker
        ctx.task_name = task_name
        ctx.task_weight = task_weight
        
        # The forward pass does absolutely nothing to the tensor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Calculate the L2 norm of the incoming gradient from the head
        norm = torch.linalg.norm(grad_output).item()
        
        # 2. Update the EMA for this specific task
        ema_norm = ctx.ema_tracker.update(ctx.task_name, norm)
        
        # 3. Scale the gradient: (Weight / EMA) * Gradient
        # We add 1e-8 to prevent division by zero
        scale_factor = ctx.task_weight / (ema_norm + 1e-8)
        scaled_grad = grad_output * scale_factor
        
        # Return scaled grad for 'x', and None for the other non-tensor arguments
        return scaled_grad, None, None, None

class DropPath(nn.Module):
    """Stochastic Depth: Randomly drops paths (blocks) per sample during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (Batch, 1, 1) to broadcast across channels and time
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation: Dynamically recalibrates channel weights."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        # Squeeze time down to a single vector per channel
        y = self.squeeze(x).view(b, c)
        # Calculate channel attention weights
        y = self.excite(y).view(b, c, 1)
        # Broadcast multiply
        return x * y

class ModernResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_path=0.0):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.se = SEBlock1D(out_channels)
        self.drop_path = DropPath(drop_path)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            layers = []
            if stride != 1:
                layers.append(nn.AvgPool1d(kernel_size=2, stride=stride))
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False))
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        shortcut = self.shortcut(x)
        
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)
        
        out = self.se(out)
        out = self.drop_path(out)
        
        return out + shortcut

class MultiTaskFADResNet(nn.Module):
    def __init__(self, num_instruments, num_labels, style_dim):
        super().__init__()
        
        # 1. Stem (Initial projection from Mel-Spectrogram)
        # Mels usually have 128 bins. We expand to 128 channels immediately.
        self.stem = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(1, 128),
            nn.GELU()
        )
        
        # 2. The ResNet Trunk
        # Gradually increase channels while downsampling the temporal dimension.
        # Assuming input is ~250 frames (4 secs): 
        # Stage 1: 250 frames -> Stage 2: 125 frames -> Stage 3: 62 frames.
        # 62 frames is great for the downbeat head (approx 60ms resolution).
        
        dp_rates = [0.0, 0.05, 0.1, 0.15] # Gradually increase stochastic depth
        
        self.stage1 = nn.Sequential(
            ModernResBlock1D(128, 128, stride=1, drop_path=dp_rates[0]),
            ModernResBlock1D(128, 128, stride=1, drop_path=dp_rates[0])
        )
        self.stage2 = nn.Sequential(
            ModernResBlock1D(128, 256, stride=2, drop_path=dp_rates[1]),
            ModernResBlock1D(256, 256, stride=1, drop_path=dp_rates[1])
        )
        self.stage3 = nn.Sequential(
            ModernResBlock1D(256, 512, stride=2, drop_path=dp_rates[2]),
            ModernResBlock1D(512, 512, stride=1, drop_path=dp_rates[2])
        )
        self.stage4 = nn.Sequential(
            ModernResBlock1D(512, 512, stride=1, drop_path=dp_rates[3]),
            ModernResBlock1D(512, 512, stride=1, drop_path=dp_rates[3]),
            nn.GroupNorm(1, 512),
            nn.GELU()
        )
        
        # 3. The Temporal Head (Downbeats)
        self.head_downbeats = nn.Conv1d(512, 1, kernel_size=3, padding=1)
        
        # 4. The Global Heads
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head_bpm = nn.Linear(512, 1)
        self.head_year = nn.Linear(512, 1)
        self.head_instruments = nn.Linear(512, num_instruments)
        self.head_label = nn.Linear(512, num_labels)
        self.head_style = nn.Linear(512, style_dim)

    def forward(self, x, ema_tracker, task_weights):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        trunk_out = self.stage4(out)
        
        outputs = {}
        
        # --- Temporal Branch ---
        t_downbeats = GradientBalancer.apply(trunk_out, ema_tracker, 'downbeats', task_weights['downbeats'])
        outputs['downbeats'] = self.head_downbeats(t_downbeats).squeeze(1)
        
        # --- Global Branch ---
        global_features = self.global_pool(trunk_out).squeeze(2) 
        
        t_bpm = GradientBalancer.apply(global_features, ema_tracker, 'bpm', task_weights['bpm'])
        outputs['bpm'] = self.head_bpm(t_bpm).squeeze(1)
        
        t_year = GradientBalancer.apply(global_features, ema_tracker, 'year', task_weights['year'])
        outputs['year'] = self.head_year(t_year).squeeze(1)
        
        t_inst = GradientBalancer.apply(global_features, ema_tracker, 'instruments', task_weights['instruments'])
        outputs['instruments'] = self.head_instruments(t_inst)
        
        t_label = GradientBalancer.apply(global_features, ema_tracker, 'label', task_weights['label'])
        outputs['label'] = self.head_label(t_label)
        
        t_style = GradientBalancer.apply(global_features, ema_tracker, 'style', task_weights['style'])
        outputs['style'] = self.head_style(t_style)
        
        return outputs

def train_fad_network():
    batch_size = 8
    seq_len = 250
    num_instruments = 10
    num_labels = 5
    style_dim = 128
    
    model = MultiTaskFADResNet(num_instruments, num_labels, style_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Initialize the EMA Tracker for the gradients
    ema_tracker = EMATracker(beta=0.99)
    
    # Define the weights (the "fractions" of the gradient you want each task to have)
    task_weights = {
        'downbeats': 1.0, 
        'bpm': 1.0, 
        'year': 0.5,         # Maybe year is less critical than rhythm
        'instruments': 2.0,  # Timbre is highly critical for FAD
        'label': 0.5, 
        'style': 1.0
    }
    
    crit_bce = nn.BCEWithLogitsLoss()
    crit_mse = nn.MSELoss()
    crit_ce = nn.CrossEntropyLoss()
    crit_cosine = nn.CosineEmbeddingLoss()

    model.train()
    
    # --- Dummy Training Loop ---
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Simulate a batch of Mel-Spectrograms: (Batch, Channels, Time)
        mels = torch.randn(batch_size, 128, seq_len)
        
        # Simulate Ground Truth Targets
        target_downbeats = torch.empty(batch_size, 63).random_(2) # 63 is the downsampled time dimension
        target_bpm = torch.rand(batch_size) * 60 + 80 # BPMs between 80-140
        target_year = torch.rand(batch_size) * 10 + 1920
        target_inst = torch.empty(batch_size, num_instruments).random_(2) # Multi-label
        target_label = torch.randint(0, num_labels, (batch_size,)) # Single label
        target_style = torch.randn(batch_size, style_dim) # From your contrastive network
        
        # 1. Forward Pass (passes weights and tracker into the network)
        preds = model(mels, ema_tracker, task_weights)
        
        # 2. Calculate Individual Losses
        loss_db = crit_bce(preds['downbeats'], target_downbeats)
        loss_bpm = crit_mse(preds['bpm'], target_bpm)
        loss_year = crit_mse(preds['year'], target_year)
        loss_inst = crit_bce(preds['instruments'], target_inst)
        loss_label = crit_ce(preds['label'], target_label)
        
        # Cosine embedding loss needs a target vector of 1s (saying "make these align")
        target_align = torch.ones(batch_size) 
        loss_style = crit_cosine(preds['style'], target_style, target_align)
        
        # 3. Sum the losses directly!
        # Do NOT multiply by weights here; the custom autograd function handles the math.
        total_loss = loss_db + loss_bpm + loss_year + loss_inst + loss_label + loss_style
        
        # 4. Backward Pass & Optimize
        # As gradients flow backward, they hit the GradientBalancer, get scaled by 
        # the EMA norms, and sum together perfectly at the trunk bottleneck.
        total_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1} | Total Unscaled Loss: {total_loss.item():.4f}")

def read_beat_timestamps(tsv_path):
    """Reads the beat_this TSV file and extracts a list of timestamps."""
    timestamps = []
    try:
        with open(tsv_path, 'r') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            for row in reader:
                if row: # Skip empty rows
                    try:
                        # Assuming the first column is the timestamp in seconds
                        timestamps.append(float(row[0]))
                    except ValueError:
                        # Skip header rows if they exist
                        continue
        return timestamps
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        return []

def calculate_subset_bpm(timestamps, start_time, end_time):
    """Calculates the BPM for a specific time window given a list of beat timestamps."""
    # Filter beats that fall within our window
    subset_beats = [t for t in timestamps if start_time <= t <= end_time]
    
    # We need at least 2 beats to calculate an interval
    if len(subset_beats) < 2:
        return 0.0
    
    # BPM calculation: (number of intervals) / (duration of intervals) * 60 seconds
    num_intervals = len(subset_beats) - 1
    duration_of_intervals = subset_beats[-1] - subset_beats[0]
    
    if duration_of_intervals == 0:
        return 0.0
        
    bpm = (num_intervals / duration_of_intervals) * 60.0
    return bpm

if __name__ == "__main__":
    # train_fad_network()
    
    import pickle
    import pandas as pd
    import numpy as np
    import torch
    
    from sklearn.preprocessing import MultiLabelBinarizer
    
    wav = '/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/JV-9999-1935-QmNmKtxL36DZskiHCYeGz7yRGF6UTxbeqfY6rLGsUiyt93.wav-DV519464.wav'
    beat = '/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_beats/JV-9999-1935-QmNmKtxL36DZskiHCYeGz7yRGF6UTxbeqfY6rLGsUiyt93.beats-DV519464.beats'
    
    timestamps = read_beat_timestamps(beat)
    bpm = calculate_subset_bpm(timestamps, 1, 6)
    print(bpm)
    print(timestamps[-1], len(sf.read(wav)) / 16000)

    cards = pickle.load(open('JazzSet.0.9.pkl', "rb"))
    cards = [card for card in cards if card]

    years = []
    labels = []
    people = []
    instruments = []
    for card in cards:
        years.append(card['DATE']['YEAR'])
        labels.append(card['RECORD']['LABEL'])
        people.append(list(card['PERSONNEL']['PEOPLE'].keys()))
        instruments.append(list(card['PERSONNEL']['INSTRUMENTS'].keys()))

    instrument_map_df = pd.read_csv('/content/instrument_mapping.csv')
    instrument_map_df = instrument_map_df.apply(lambda col: col.astype(str).str.lower())
    instrument_map = {row['Abbreviation']: row['Consolidated_Category'] for i, row in instrument_map_df.iterrows()}
    instrument_categories = set(list(instrument_map.values()))

    from collections import defaultdict
    data = defaultdict(list)
    for instrument_list in instruments:
        categories = {instrument_map[l.lower()] for l in instrument_list if l in instrument_map}
        for cat in instrument_categories:
            if cat in categories:
                data[cat].append(True)
            else:
                data[cat].append(False)

    props = pd.DataFrame(data, columns=list(instrument_categories)).sum(0)
    instrument_categories = {cat for cat in instrument_categories if props[cat] >= 1000}

    data = []
    for instrument_list in instruments:
        categories = {instrument_map[l.lower()] for l in instrument_list if l in instrument_map}
        categories = categories.intersection(instrument_categories)
        data.append(list(filter(None, categories)))

    mlb = MultiLabelBinarizer().fit(data)
    data = mlb.transform(data)
    data = torch.from_numpy(data).float()
    
    def create_gaussian_soft_labels(targets, bins, sigma):
        targets = targets.unsqueeze(1)
        bins = bins.unsqueeze(0).to(targets.device)
        squared_distances = (bins - targets) ** 2
        gaussian_weights = torch.exp(-squared_distances / (2 * sigma ** 2))
        soft_labels = gaussian_weights / gaussian_weights.sum(dim=1, keepdim=True)
        return soft_labels

    bpm_bins = torch.arange(40, 260, 10, dtype=torch.float32)
    year_bins = torch.arange(1900, 1980, 10, dtype=torch.float32)
    bpm_targets = torch.tensor([155])
    year_targets = torch.tensor([1935])

    bpm_sigma, year_sigma = 5, 2.5
    bpm_labels = create_gaussian_soft_labels(bpm_targets, bpm_bins, bpm_sigma)
    year_labels = create_gaussian_soft_labels(year_targets, year_bins, year_sigma)
