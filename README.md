# 🎙️ ReUse-Studio: Universal Speech Enhancement App

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/NVIDIA-RE--USE-76B900.svg)](https://huggingface.co/nvidia/RE-USE)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)

A high-performance voice enhancement web application powered by **NVIDIA RE-USE** (Multilingual Universal Speech Enhancement) running with custom GPU Triton acceleration.

**ReUse-Studio** transforms noisy, muffled, reverberant, or clipped voice recordings into studio-grade clarity while preserving the speaker's natural timbre, pitch, accent, and emotional nuance.

---

## ✨ Features

- **🎙️ Direct Voice Recording & Audio Upload**: Record live voice directly from your browser microphone or upload any audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`).
- **🎬 Video Speech Cleaner**: Upload video files (`.mp4`, `.mov`, `.mkv`, `.webm`) to automatically extract, enhance, and export dialogue tracks.
- **⚡ 1-Click Sample Test**: Instant button to load a noisy mic sample and test enhancement immediately.
- **🛡️ Universal Audio Restoration**:
  - **Denoising**: Eliminates stationary & non-stationary noise (fans, keyboards, street noise, cafe chatter).
  - **Dereverberation**: Suppresses hollow room echo and acoustic room reflections.
  - **Declipping & De-distortion**: Reconstructs saturated waveform peaks distorted by cheap or overdriven microphones.
  - **Bandwidth Extension (BWE)**: Restores missing high-frequency harmonics (e.g. upscales 8 kHz phone calls or 16 kHz voice memos up to 44.1 kHz or 48 kHz studio fidelity).
- **⏱️ Auto-Chunking Overlap-Add**: Long podcasts and recordings are automatically processed in windowed chunks with Hann tapering to prevent GPU Out-Of-Memory (OOM) errors.
- **📊 Comparative Spectrogram Visualizer**: Side-by-side time-frequency heatmaps showing background noise elimination and vocal harmonic recovery.
- **📦 Batch Processing**: Queue and process multiple recordings simultaneously.

---

## ⚡ High-Speed Triton Acceleration

NVIDIA RE-USE relies on **Mamba State-Space Models (SSM)**. On Windows, compiling native CUDA C++ extensions for Mamba is notoriously difficult, while naive PyTorch loops can take 10+ minutes per audio file.

**ReUse-Studio** includes an embedded JIT **Triton Selective-Scan Kernel** (`triton_scan.py`):
* Directly executes parallel state-space scans on the GPU in **~95 ms** per pass.
* Provides a **6,000x speedup** on NVIDIA GPUs (e.g. GeForce RTX 3050, 3060, 4060, 4090, etc.) with zero C++ compiler requirements on Windows.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Parsa2005-s/ReUse-Studio.git
cd ReUse-Studio
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with CUDA
Install PyTorch matching your CUDA version (CUDA 12.x recommended):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Mamba-SSM & Dependencies

#### On Windows:
```powershell
# Set environment variables to skip CUDA C++ build (uses fast Triton kernel instead)
$env:MAMBA_SKIP_CUDA_BUILD="TRUE"
$env:CAUSAL_CONV1D_SKIP_CUDA_BUILD="TRUE"
pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm

# Install remaining dependencies
pip install -r requirements.txt
```

#### On Linux:
```bash
pip install causal-conv1d mamba-ssm
pip install -r requirements.txt
```

*(Optional for video audio extraction: ensure `ffmpeg` is installed on your system).*

---

## 🎮 Running the Application

### Option 1: Quick Launch (Windows)
Double-click **[`run_app.bat`](run_app.bat)**.

### Option 2: Command Line
```bash
python app.py
```

Open your browser at:
👉 **`http://127.0.0.1:7860`**

---

## 🌐 Public Sharing (Gradio Link)

To share your running app with collaborators or friends over the internet, add `share=True` to the launch line at the bottom of `app.py`:
```python
demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
```
Gradio will generate a secure public URL (e.g. `https://xxxx.gradio.live`) valid for 72 hours.

---

## 📜 Credits & License

* **Model:** Developed by **NVIDIA** (under NemotronLabs). Model weights and architecture based on `nvidia/RE-USE`.
* **Kernel & Optimization:** Custom Triton SSM parallel scan for Windows/Linux GPU execution.
