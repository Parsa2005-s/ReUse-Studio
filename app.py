import os
import sys
import tempfile
import gradio as gr
import torch

from engine import ReuseEnhancer

# Initialize model
print("[APP] Starting ReUse-Studio Voice Enhancement App...")
enhancer = ReuseEnhancer()

CUSTOM_CSS = """
/* NVIDIA-inspired modern dark theme */
:root {
    --primary: #76B900;
    --primary-hover: #88cc00;
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --text-main: #f8fafc;
}

body {
    background-color: var(--bg-dark) !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

.main-title {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}

.main-title h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #76B900 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.main-title p {
    color: #94a3b8;
    font-size: 1.05rem;
}

.badge-container {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}

.badge {
    background: #1e293b;
    border: 1px solid #334155;
    color: #e2e8f0;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}

.badge-green {
    border-color: #76B900;
    color: #76B900;
}

.enhance-btn {
    background: linear-gradient(135deg, #76B900 0%, #5d9300 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(118, 185, 0, 0.35) !important;
    transition: all 0.2s ease !important;
}

.enhance-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(118, 185, 0, 0.5) !important;
}

.metrics-box {
    background: #1e293b;
    border-radius: 8px;
    border: 1px solid #334155;
    padding: 1rem;
    margin-top: 1rem;
}
"""

def enhance_single(audio_file, bwe_sr, lp_sr, chunk_seconds, progress=gr.Progress()):
    if not audio_file:
        raise gr.Error("Please record or upload an audio file first!")
    
    target_sr = None
    if bwe_sr and bwe_sr != "Original (Keep)":
        target_sr = int(bwe_sr.replace(" Hz", ""))
        
    low_pass = None
    if lp_sr and lp_sr != "None":
        low_pass = int(lp_sr.replace(" Hz", ""))

    chunk_sec = float(chunk_seconds) if chunk_seconds else 10.0

    out_file, metrics, fig = enhancer.enhance_audio(
        audio_path=audio_file,
        target_sr=target_sr,
        low_pass_sr=low_pass,
        chunk_seconds=chunk_sec,
        progress_cb=lambda p, msg: progress(p, desc=msg)
    )

    metrics_md = f"""
### ⚡ Enhancement Report
* **Processing Time:** `{metrics['elapsed_seconds']}s`
* **Real-Time Factor (RTF):** `{metrics['rtf']}x` *(< 1.0 is faster than real-time)*
* **Audio Duration:** `{metrics['audio_duration']}s`
* **Input Rate:** `{metrics['orig_sr']} Hz` → **Output Rate:** `{metrics['target_sr']} Hz`
* **Channels:** `{metrics['num_channels']}` | **Chunking Mode:** `{'Enabled' if metrics['chunking_used'] else 'Direct Single-Pass'}`
"""

    return audio_file, out_file, fig, metrics_md, gr.update(value=out_file, visible=True)

def enhance_video(video_file, bwe_sr, chunk_seconds, progress=gr.Progress()):
    if not video_file:
        raise gr.Error("Please upload a video file first!")
        
    import subprocess
    progress(0.05, desc="Extracting audio from video...")
    
    # Extract audio using ffmpeg / moviepy / torchaudio
    temp_wav = tempfile.mktemp(suffix=".wav")
    try:
        # Check ffmpeg
        subprocess.run(["ffmpeg", "-y", "-i", video_file, "-vn", "-acodec", "pcm_s16le", temp_wav], 
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        # Fallback to soundfile / pydub
        from pydub import AudioSegment
        audio_seg = AudioSegment.from_file(video_file)
        audio_seg.export(temp_wav, format="wav")

    return enhance_single(temp_wav, bwe_sr, "None", chunk_seconds, progress=progress)

def load_sample():
    sample_path = os.path.join(enhancer.model_dir, "noisy_audio", "mic_test2.wav")
    if os.path.exists(sample_path):
        return sample_path
    return None

def enhance_batch(file_list, bwe_sr, chunk_seconds, progress=gr.Progress()):
    if not file_list:
        raise gr.Error("Please select one or more files for batch processing!")

    results = []
    target_sr = None
    if bwe_sr and bwe_sr != "Original (Keep)":
        target_sr = int(bwe_sr.replace(" Hz", ""))

    chunk_sec = float(chunk_seconds) if chunk_seconds else 10.0

    total = len(file_list)
    for i, file_obj in enumerate(file_list):
        progress((i / total), desc=f"Processing file {i+1}/{total}: {os.path.basename(file_obj.name)}")
        fpath = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        out_file, _, _ = enhancer.enhance_audio(
            audio_path=fpath,
            target_sr=target_sr,
            chunk_seconds=chunk_sec
        )
        results.append(out_file)

    progress(1.0, desc="Batch complete!")
    return results

with gr.Blocks(title="ReUse-Studio - AI Voice Enhancement") as demo:
    gr.HTML("""
    <div class="main-title">
        <h1>ReUse-Studio</h1>
        <p>Studio-quality voice restoration powered by NVIDIA RE-USE & Mamba State-Space Models</p>
    </div>
    <div class="badge-container">
        <span class="badge badge-green">🚀 ReUse-Studio</span>
        <span class="badge">🧠 9.6M Mamba SSM Parameters</span>
        <span class="badge">⚡ GPU Triton Accelerated</span>
        <span class="badge">🎙️ Denoise • Dereverb • Declip • Bandwidth Extension</span>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: Single Audio / Microphone
        with gr.TabItem("🎙️ Voice & Audio Enhancement"):
            with gr.Row():
                with gr.Column(scale=5):
                    input_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record Voice from Microphone or Upload Audio",
                        waveform_options=gr.WaveformOptions(waveform_color="#76B900")
                    )
                    
                    with gr.Row():
                        sample_btn = gr.Button("⚡ Load Sample Audio (Noisy Mic)", size="sm", variant="secondary")

                    with gr.Accordion("⚙️ Enhancement & Bandwidth Options", open=False):
                        with gr.Row():
                            bwe_sr = gr.Dropdown(
                                choices=["Original (Keep)", "16000 Hz", "24000 Hz", "44100 Hz", "48000 Hz"],
                                value="Original (Keep)",
                                label="Bandwidth Extension (BWE)",
                                info="Upscale sample rate and restore high-frequency vocal harmonics."
                            )
                            lp_sr = gr.Dropdown(
                                choices=["None", "8000 Hz", "16000 Hz"],
                                value="None",
                                label="Pre-Low-Pass Filtering",
                                info="Filter before BWE to eliminate cutoff artifacts."
                            )
                        chunk_seconds = gr.Slider(
                            minimum=5.0,
                            maximum=30.0,
                            value=10.0,
                            step=1.0,
                            label="Chunk Duration (Seconds)",
                            info="Processes long recordings in overlapping chunks to prevent GPU out-of-memory."
                        )

                    enhance_btn = gr.Button("✨ Clean & Enhance Voice", elem_classes=["enhance-btn"])

                with gr.Column(scale=6):
                    with gr.Row():
                        with gr.Column():
                            orig_player = gr.Audio(label="Original Voice", type="filepath", interactive=False)
                        with gr.Column():
                            enh_player = gr.Audio(label="Enhanced Studio Voice", type="filepath", interactive=False)
                    
                    download_file = gr.File(label="Download Enhanced Audio (.wav)", visible=False)
                    metrics_output = gr.Markdown()

            with gr.Row():
                spec_plot = gr.Plot(label="Time-Frequency Spectrogram Comparison")

            # Events
            sample_btn.click(fn=load_sample, outputs=input_audio)
            enhance_btn.click(
                fn=enhance_single,
                inputs=[input_audio, bwe_sr, lp_sr, chunk_seconds],
                outputs=[orig_player, enh_player, spec_plot, metrics_output, download_file]
            )

        # TAB 2: Video Audio Enhancer
        with gr.TabItem("🎬 Video Voice Enhancer"):
            gr.Markdown("### Clean speech tracks from video recordings (.mp4, .mov, .mkv, .webm)")
            with gr.Row():
                with gr.Column(scale=5):
                    input_video = gr.File(label="Upload Video File", file_types=[".mp4", ".mov", ".mkv", ".webm", ".avi"])
                    video_bwe = gr.Dropdown(
                        choices=["Original (Keep)", "16000 Hz", "44100 Hz", "48000 Hz"],
                        value="Original (Keep)",
                        label="Target Output Sample Rate"
                    )
                    video_chunk = gr.Slider(5.0, 30.0, value=10.0, step=1.0, label="Chunk Duration (Seconds)")
                    video_btn = gr.Button("✨ Extract & Clean Video Audio", elem_classes=["enhance-btn"])
                with gr.Column(scale=6):
                    with gr.Row():
                        video_orig_player = gr.Audio(label="Original Video Audio", type="filepath", interactive=False)
                        video_enh_player = gr.Audio(label="Enhanced Video Speech", type="filepath", interactive=False)
                    video_download = gr.File(label="Download Clean Audio", visible=False)
                    video_metrics = gr.Markdown()
                    video_plot = gr.Plot(label="Spectrogram Comparison")

            video_btn.click(
                fn=enhance_video,
                inputs=[input_video, video_bwe, video_chunk],
                outputs=[video_orig_player, video_enh_player, video_plot, video_metrics, video_download]
            )

        # TAB 3: Batch Processing
        with gr.TabItem("📦 Batch Enhancement"):
            gr.Markdown("### Enhance multiple recordings in a single queue")
            batch_files = gr.File(label="Select Multiple Audio Files", file_count="multiple")
            with gr.Row():
                batch_bwe = gr.Dropdown(
                    choices=["Original (Keep)", "16000 Hz", "44100 Hz", "48000 Hz"],
                    value="Original (Keep)",
                    label="Bandwidth Extension"
                )
                batch_chunk = gr.Slider(5.0, 30.0, value=10.0, step=1.0, label="Chunk Size (Seconds)")
            batch_run_btn = gr.Button("🚀 Process All Files", variant="primary")
            batch_output = gr.File(label="Enhanced Audio Files", file_count="multiple")

            batch_run_btn.click(
                fn=enhance_batch,
                inputs=[batch_files, batch_bwe, batch_chunk],
                outputs=batch_output
            )

        # TAB 4: Architecture & Info
        with gr.TabItem("ℹ️ About NVIDIA RE-USE"):
            gr.Markdown("""
### What is NVIDIA RE-USE?
**RE-USE (Multilingual Universal Speech Enhancement)** is an open-weights universal speech enhancement model developed by NVIDIA (NemotronLabs).

Unlike traditional noise reduction filters that introduce robotic artifacts or cut off consonants, RE-USE is designed to clean up and restore degraded speech with exceptional acoustic fidelity:
* **Background Noise Removal:** Eliminates street noise, fan hum, typing, keyboard clicks, and cafe chatter.
* **Dereverberation:** Suppresses room echo, hollow acoustics, and bad room reflections.
* **Declipping & De-distortion:** Reconstructs audio clipped at 0 dB or distorted by cheap microphones.
* **Bandwidth Extension (BWE):** Restores missing high-frequency harmonics (e.g. upscaling 8 kHz telephone recordings to 44.1 kHz or 48 kHz studio quality).
* **Architecture:** Compact 9.6M parameter model using **Mamba State-Space Models (SSM)** in a Temporal-Frequency bidirectional structure (`TFMamba`), running with custom GPU Triton kernels.
""")

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=gr.themes.Soft(primary_hue="emerald"), css=CUSTOM_CSS)
