import os
import sys
import math
import time
import torch
import torchaudio
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from safetensors.torch import load_file

# Add local model directory to sys.path
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvidia_reuse_model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from models.generator_SEMamba_time_d4 import SEMamba
from models.stfts import mag_phase_stft, mag_phase_istft
from utils.util import load_config, pad_or_trim_to_match

RELU = nn.ReLU()

def make_even(value: float) -> int:
    val = int(round(value))
    return val if val % 2 == 0 else val + 1

class ReuseEnhancer:
    def __init__(self, model_dir: str = MODEL_DIR, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model_dir = model_dir
        self.config_path = os.path.join(model_dir, "config.json")
        self.weights_path = os.path.join(model_dir, "model.safetensors")
        
        self.cfg = load_config(self.config_path)
        self.n_fft_base = self.cfg['stft_cfg']['n_fft']
        self.hop_size_base = self.cfg['stft_cfg']['hop_size']
        self.win_size_base = self.cfg['stft_cfg']['win_size']
        self.compress_factor = self.cfg['model_cfg']['compress_factor']
        self.base_sampling_rate = self.cfg['stft_cfg']['sampling_rate']
        
        print(f"[RE-USE] Initializing SEMamba model on device: {self.device}...")
        self.model = SEMamba(cfg=self.cfg).to(self.device)
        weights = load_file(self.weights_path)
        self.model.load_state_dict(weights)
        self.model.eval()
        print(f"[RE-USE] Model ready! ({sum(p.numel() for p in self.model.parameters()):,} parameters)")

    def _enhance_chunk(self, noisy_chunk: torch.Tensor, sr: int) -> torch.Tensor:
        """Enhance a single tensor chunk [1, samples] on GPU."""
        n_fft_scaled = make_even(self.n_fft_base * sr // self.base_sampling_rate)
        hop_size_scaled = make_even(self.hop_size_base * sr // self.base_sampling_rate)
        win_size_scaled = make_even(self.win_size_base * sr // self.base_sampling_rate)

        noisy_mag, noisy_pha, _ = mag_phase_stft(
            noisy_chunk,
            n_fft=n_fft_scaled,
            hop_size=hop_size_scaled,
            win_size=win_size_scaled,
            compress_factor=self.compress_factor,
            center=True,
            addeps=False
        )
        
        amp_g, pha_g, _ = self.model(noisy_mag, noisy_pha)
        
        # Remove sweep artifact
        mag = torch.expm1(RELU(amp_g))
        zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0

        audio_g = mag_phase_istft(
            amp_g, pha_g,
            n_fft=n_fft_scaled,
            hop_size=hop_size_scaled,
            win_size=win_size_scaled,
            compress_factor=self.compress_factor
        )
        audio_g = pad_or_trim_to_match(noisy_chunk.detach(), audio_g, pad_value=1e-8)
        return audio_g

    @torch.no_grad()
    def enhance_audio(
        self,
        audio_path: str,
        output_path: str = None,
        target_sr: int = None,
        low_pass_sr: int = None,
        chunk_seconds: float = 10.0,
        hop_fraction: float = 0.5,
        progress_cb=None
    ):
        """
        Enhance an audio file using NVIDIA RE-USE.
        Returns:
            dict containing:
                output_path: path to enhanced file
                metrics: dict with timing, sample rate, duration, RTF
                fig: matplotlib comparison spectrogram
        """
        start_time = time.time()
        
        # Load audio using soundfile / librosa fallback for widest format support
        try:
            noisy_wav, sr = torchaudio.load(audio_path)
        except Exception:
            # Fallback to soundfile / librosa
            data, sr = sf.read(audio_path, dtype="float32")
            if data.ndim == 1:
                noisy_wav = torch.from_numpy(data).unsqueeze(0)
            else:
                noisy_wav = torch.from_numpy(data.T)

        orig_wav = noisy_wav.clone()
        orig_sr = sr
        orig_duration = noisy_wav.shape[1] / sr

        # Optional Pre-Low-Pass Filtering
        if low_pass_sr is not None and int(low_pass_sr) > 0 and int(low_pass_sr) < sr:
            low_pass_sr = int(low_pass_sr)
            noisy_np = noisy_wav.cpu().numpy()
            resampled = librosa.resample(noisy_np, orig_sr=sr, target_sr=low_pass_sr, res_type="kaiser_best")
            noisy_wav = torch.tensor(resampled)
            sr = low_pass_sr

        # Optional Bandwidth Extension (BWE) / Target Sample Rate
        if target_sr is not None and int(target_sr) > 0 and int(target_sr) != sr:
            target_sr = int(target_sr)
            noisy_np = noisy_wav.cpu().numpy()
            resampled = librosa.resample(noisy_np, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
            noisy_wav = torch.tensor(resampled)
            sr = target_sr

        total_samples = noisy_wav.shape[1]
        num_channels = noisy_wav.shape[0]

        # Determine if chunking is needed (auto chunk for clips > 12 seconds to save VRAM)
        use_chunking = (total_samples / sr) > 12.0 or (chunk_seconds is not None and chunk_seconds > 0)
        
        if progress_cb:
            progress_cb(0.1, "Preprocessing audio...")

        if not use_chunking:
            # Single pass
            noisy_wav_gpu = noisy_wav.float().to(self.device)
            enhanced_channels = []
            for c in range(num_channels):
                enh = self._enhance_chunk(noisy_wav_gpu[c:c+1, :], sr)
                enhanced_channels.append(enh)
            audio_enhanced = torch.cat(enhanced_channels, dim=0).cpu()
        else:
            # Chunk-wise overlap-add
            chunk_size = int(chunk_seconds * sr)
            hop_length = int(hop_fraction * chunk_size)
            window = torch.clamp(torch.hann_window(chunk_size, device=self.device), min=1e-7)

            audio_enhanced = torch.zeros((num_channels, total_samples), dtype=torch.float32)
            window_sum = torch.zeros((num_channels, total_samples), dtype=torch.float32)

            num_chunks = max(1, math.ceil((total_samples - chunk_size) / hop_length) + 1)
            
            for c in range(num_channels):
                noisy_ch = noisy_wav[c:c+1, :].to(self.device)
                ch_enhanced = torch.zeros((1, total_samples), device=self.device)
                ch_window_sum = torch.zeros((1, total_samples), device=self.device)

                for i in range(num_chunks):
                    if progress_cb:
                        prog = 0.1 + 0.8 * ((c * num_chunks + i) / (num_channels * num_chunks))
                        progress_cb(prog, f"Processing chunk {i+1}/{num_chunks}...")

                    if i < num_chunks - 1:
                        chunk = noisy_ch[:, i * hop_length : i * hop_length + chunk_size]
                        sl = slice(i * hop_length, i * hop_length + chunk_size)
                    else:
                        chunk = noisy_ch[:, -chunk_size:]
                        sl = slice(-chunk_size, None)

                    enh_chunk = self._enhance_chunk(chunk, sr)
                    ch_enhanced[:, sl] += enh_chunk * window
                    ch_window_sum[:, sl] += window

                ch_window_sum = torch.clamp(ch_window_sum, min=1e-7)
                audio_enhanced[c:c+1, :] = (ch_enhanced / ch_window_sum).cpu()

        if progress_cb:
            progress_cb(0.9, "Generating output and spectrograms...")

        # Normalize output clipping protection
        max_val = torch.max(torch.abs(audio_enhanced))
        if max_val > 0.99:
            audio_enhanced = audio_enhanced / max_val * 0.99

        # Save output
        if output_path is None:
            os.makedirs("enhanced_outputs", exist_ok=True)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join("enhanced_outputs", f"{base_name}_enhanced.wav")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        torchaudio.save(output_path, audio_enhanced, sr)

        elapsed = time.time() - start_time
        rtf = elapsed / max(orig_duration, 0.01)

        # Generate comparison spectrogram
        fig = self._plot_spectrograms(orig_wav[0].cpu().numpy(), orig_sr, audio_enhanced[0].cpu().numpy(), sr)

        metrics = {
            "elapsed_seconds": round(elapsed, 2),
            "audio_duration": round(orig_duration, 2),
            "rtf": round(rtf, 3),
            "orig_sr": orig_sr,
            "target_sr": sr,
            "num_channels": num_channels,
            "chunking_used": use_chunking
        }

        if progress_cb:
            progress_cb(1.0, "Done!")

        return output_path, metrics, fig

    def _plot_spectrograms(self, orig_audio: np.ndarray, orig_sr: int, enh_audio: np.ndarray, enh_sr: int):
        """Plot high-contrast comparative dark-themed spectrograms."""
        plt.style.use('dark_background')
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), dpi=100)
        fig.patch.set_facecolor('#111827')
        
        # Original spectrogram
        D_orig = librosa.stft(orig_audio, n_fft=512, hop_length=256)
        S_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
        img1 = librosa.display.specshow(S_orig, sr=orig_sr, hop_length=256, x_axis="time", y_axis="hz", ax=axs[0], cmap="magma", vmax=0)
        axs[0].set_title(f"Original Input Audio ({orig_sr} Hz)", color='#9ca3af', fontsize=12, fontweight='bold', pad=10)
        axs[0].set_facecolor('#1f2937')
        axs[0].tick_params(colors='#9ca3af')
        
        # Enhanced spectrogram
        D_enh = librosa.stft(enh_audio, n_fft=512, hop_length=256)
        S_enh = librosa.amplitude_to_db(np.abs(D_enh), ref=np.max)
        img2 = librosa.display.specshow(S_enh, sr=enh_sr, hop_length=256, x_axis="time", y_axis="hz", ax=axs[1], cmap="magma", vmax=0)
        axs[1].set_title(f"NVIDIA RE-USE Enhanced Audio ({enh_sr} Hz)", color='#10b981', fontsize=12, fontweight='bold', pad=10)
        axs[1].set_facecolor('#1f2937')
        axs[1].tick_params(colors='#9ca3af')

        fig.colorbar(img2, ax=axs, format="%+2.0f dB", pad=0.02)
        plt.tight_layout()
        return fig
