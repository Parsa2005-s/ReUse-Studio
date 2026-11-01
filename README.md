# 🎙️ ReUse-Studio: Universal Speech Enhancement App

A studio-grade voice restoration web application powered by **NVIDIA RE-USE** (Multilingual Universal Speech Enhancement) running with high-speed GPU Triton kernel acceleration.

---

## ✨ Features

- **Microphone & Audio Upload**: Record voice directly from your browser mic or upload audio in any format (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
- **Video Speech Cleaner**: Upload videos (`.mp4`, `.mov`, `.mkv`, `.webm`) to extract and enhance noisy speech tracks.
- **Universal Voice Restoration**:
  - **Denoising**: Eliminates background noise, typing, traffic, fans, hums, and cafe chatter.
  - **Dereverberation**: Suppresses room echo and acoustic reflections.
  - **Declipping**: Reconstructs saturated and clipped microphone audio.
  - **Bandwidth Extension (BWE)**: Restores missing high-frequency harmonics (e.g. upscaling 8 kHz phone audio to 44.1 kHz or 48 kHz studio quality).
- **Auto-Chunking Overlap-Add**: Long recordings are automatically processed in windowed chunks with Hann tapering to prevent GPU Out-Of-Memory (OOM) errors.
- **Real-Time Spectrograms**: Side-by-side time-frequency visualizations showing noise removal and vocal recovery.
- **Batch Processing**: Queue and enhance multiple files at once.

---

## 🚀 Quick Start for GitHub Users

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/nvidia-reuse-speech-enhancer.git
cd nvidia-reuse-speech-enhancer
```

### 2. Install Requirements
Create a virtual environment (recommended) and install the dependencies:
```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

> **Note for Windows Users**: The repository includes a custom Triton selective-scan kernel for Windows to run NVIDIA Mamba SSM at maximum speed on CUDA without C++ build issues.

### 3. Run the App
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:7860`**

---

## ⚡ Sharing with Others over the Internet (Gradio Public Link)

If you want friends or colleagues to access your running app over the internet without installing anything, add `share=True` to the launch command in `app.py`:
```python
demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
```
Gradio will automatically generate a secure public link (e.g., `https://xxxx.gradio.live`) valid for 72 hours.
