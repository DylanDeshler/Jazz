import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
import librosa
import librosa.display
import math

NUM_ACTIONS = 64
ACTIONS = [f"Action {i+1}" for i in range(NUM_ACTIONS)]
GRID_COLS = 8  # Number of dropdowns per row

# --- Helper functions ---
def plot_mel_spectrogram_with_highlight(data, sr, start_sec=None, end_sec=None, title=None, n_mels=128):
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title or "Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    if start_sec is not None and end_sec is not None:
        ax.axvspan(start_sec, end_sec, color='red', alpha=0.3)
        ax.axvline(start_sec, color='black')
        ax.axvline(end_sec, color='black')

    fig.tight_layout()
    return fig

def load_audio(audio_file):
    data, sr = sf.read(audio_file)
    duration = int(len(data) / sr)
    fig = plot_mel_spectrogram_with_highlight(data, sr, title="Full Audio Spectrogram")
    return fig, duration, data, sr

def view_chunk(data, sr, start_sec, end_sec):
    chunk = data[int(start_sec*sr):int(end_sec*sr)]
    fig = plot_mel_spectrogram_with_highlight(chunk, sr, title=f"Chunk Spectrogram ({start_sec}-{end_sec}s)")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, chunk, sr)
    return fig, temp_file.name, int(end_sec - start_sec)

def random_audio_modulation(audio, sr):
    """
    Apply random modulations to an audio array.

    Parameters:
        audio (np.ndarray): 1D or 2D audio array
        sr (int): sample rate

    Returns:
        modulated_audio (np.ndarray): modulated audio
    """
    y = audio.copy()
    
    # 1️⃣ Random volume
    gain = np.random.uniform(0.7, 1.3)  # ±30%
    y = y * gain

    # 2️⃣ Random pitch shift (±2 semitones)
    n_steps = np.random.uniform(-2, 2)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # # 3️⃣ Random time stretch (0.9x to 1.1x)
    # rate = np.random.uniform(0.9, 1.1)
    # y = librosa.effects.time_stretch(y, rate=rate)

    # 4️⃣ Random noise
    noise_level = np.random.uniform(0, 0.01)
    y = y + np.random.randn(len(y)) * noise_level

    # 5️⃣ Random reverse
    if np.random.rand() < 0.3:  # 30% chance to reverse
        y = y[::-1]

    # Clip to [-1, 1] if float32
    y = np.clip(y, -1.0, 1.0)
    return y

def generate_audio(data, sr, start_sec, end_sec, actions):
    chunk = data[int(start_sec*sr):int(end_sec*sr)]
    chunk = random_audio_modulation(chunk, sr)
    data[int(start_sec*sr):int(end_sec*sr)] = chunk
    # Placeholder for action-based modifications
    fig = plot_mel_spectrogram_with_highlight(chunk, sr, title="Generated Audio Spectrogram")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, data, sr)
    return fig, temp_file.name

# --- Build Gradio app ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎛 Audio Chunk + Action Viewer")

    # Upload audio and full spectrogram
    audio_input = gr.Audio(type="filepath", label="Upload audio")
    full_spec_output = gr.Plot(label="Full Audio Spectrogram")

    # Start/Stop integer selectors in a row
    with gr.Row():
        start_sec_input = gr.Number(label="Start (s)", value=0, precision=0)
        end_sec_input = gr.Number(label="End (s)", value=1, precision=0)

    # States
    data_state = gr.State()
    sr_state = gr.State()
    chunk_length_state = gr.State()
    
    view_chunk_btn = gr.Button("View Chunk")
    
    # Chunk audio player (hidden initially)
    chunk_audio_player = gr.Audio(label="Chunk Playback", type="filepath", visible=False)
    chunk_spec_output = gr.Plot(label="Chunk Spectrogram", visible=False)

    # Fixed NUM_ACTIONS dropdowns in a grid (no scrollable container)
    dropdowns = []
    for r in range(math.ceil(NUM_ACTIONS / GRID_COLS)):
        with gr.Row():
            for c in range(GRID_COLS):
                idx = r * GRID_COLS + c
                if idx >= NUM_ACTIONS:
                    continue
                dd = gr.Dropdown(choices=ACTIONS, label=f"Second {idx}", value=ACTIONS[0], visible=False)
                dropdowns.append(dd)

    # Buttons and generated outputs
    generate_btn = gr.Button("Generate", visible=False)
    generate_audio_player = gr.Audio(label="Generated Audio Playback", type="filepath", visible=False)
    generate_output_plot = gr.Plot(label="Generated Spectrogram", visible=False)
    generate_download = gr.File(label="Download Generated Audio", visible=False)

    # --- Callbacks ---
    def on_audio_upload(audio_file):
        fig_full, duration, data, sr = load_audio(audio_file)
        return (
            fig_full,
            gr.update(value=0, minimum=0, maximum=duration),
            gr.update(value=min(1, duration), minimum=0, maximum=duration),
            data,
            sr
        )

    audio_input.change(
        fn=on_audio_upload,
        inputs=audio_input,
        outputs=[full_spec_output, start_sec_input, end_sec_input, data_state, sr_state],
    )

    # Update spectrogram highlight and enforce constraints
    def update_selection(start, end, data, sr):
        duration = len(data) / sr
        start = max(0, int(start))
        end = max(start+1, int(min(end, duration)))
        fig = plot_mel_spectrogram_with_highlight(data, sr, start, end, title="Full Audio Spectrogram")
        return gr.update(value=start), gr.update(value=end), fig

    start_sec_input.change(
        fn=update_selection,
        inputs=[start_sec_input, end_sec_input, data_state, sr_state],
        outputs=[start_sec_input, end_sec_input, full_spec_output]
    )
    end_sec_input.change(
        fn=update_selection,
        inputs=[start_sec_input, end_sec_input, data_state, sr_state],
        outputs=[start_sec_input, end_sec_input, full_spec_output]
    )

    # View chunk callback
    def on_view_chunk(data, sr, start_sec, end_sec):
        fig_chunk, chunk_file, chunk_length = view_chunk(data, sr, start_sec, end_sec)
        updated_dropdowns = []
        for i in range(NUM_ACTIONS):
            if i < chunk_length:
                updated_dropdowns.append(gr.update(visible=True, value=ACTIONS[0], label=f"Second {i}"))
            else:
                updated_dropdowns.append(gr.update(visible=False))

        return [gr.update(visible=True, value=fig_chunk),
                gr.update(visible=True, value=chunk_file),
                *updated_dropdowns,
                chunk_length,
                gr.update(visible=True)]

    view_chunk_btn.click(
        fn=on_view_chunk,
        inputs=[data_state, sr_state, start_sec_input, end_sec_input],
        outputs=[chunk_spec_output, chunk_audio_player, *dropdowns, chunk_length_state, generate_btn],
    )

    # Generate callback
    def on_generate(data, sr, start_sec, end_sec, chunk_length, *actions):
        actions = actions[:chunk_length]
        fig, gen_file = generate_audio(data, sr, start_sec, end_sec, actions)
        return (gr.update(visible=True, value=fig),
                gr.update(visible=True, value=gen_file),
                gr.update(visible=True, value=gen_file))

    generate_btn.click(
        fn=on_generate,
        inputs=[data_state, sr_state, start_sec_input, end_sec_input, chunk_length_state, *dropdowns],
        outputs=[generate_output_plot, generate_audio_player, generate_download]
    )

demo.launch()
