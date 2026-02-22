import os
import gc
import re
import numpy as np
import time
import torch
import torchaudio
import librosa
import warnings

warnings.filterwarnings('ignore')


def print_vram(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  [VRAM {tag}] Allocated: {alloc:.2f} GB | Reserved: {reserved:.2f} GB")

def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def compute_acoustic_metrics(audio_np: np.ndarray, sr: int, transcript: str):
    """Compute acoustic metrics from a numpy waveform array.
    Mirrors the training pipeline's compute_acoustic_metrics() exactly.
    """
    y = audio_np
    if y.size < int(0.5 * sr):
        return None

    duration_s = float(y.size / sr)
    wc = _word_count(transcript)
    speech_rate_wps = float(wc / max(duration_s, 1e-6))

    intervals = librosa.effects.split(y, top_db=30)
    voiced_samples = int(np.sum([max(0, b - a) for a, b in intervals])) if len(intervals) else 0
    voiced_ratio = float(voiced_samples / max(1, y.size))
    pause_ratio = float(np.clip(1.0 - voiced_ratio, 0.0, 1.0))

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_std = float(np.std(rms))

    n_fft = 1024
    hop = 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total_energy = float(np.sum(S)) + 1e-12

    low_mask = freqs < 1000
    mid_mask = (freqs >= 1000) & (freqs < 4000)
    high_mask = freqs >= 4000

    low_e = float(np.sum(S[low_mask]))
    mid_e = float(np.sum(S[mid_mask]))
    high_e = float(np.sum(S[high_mask]))
    low_pct = float(100.0 * low_e / total_energy)
    mid_pct = float(100.0 * mid_e / total_energy)
    high_pct = float(100.0 * high_e / total_energy)
    low_mid_ratio = float((low_e + mid_e) / max(high_e, 1e-9))

    centroid = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))
    bandwidth_mean = float(np.mean(bandwidth))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]) if onset_env.size else 0.0

    f0_mean = 0.0
    f0_std = 0.0
    try:
        f0 = librosa.yin(y, fmin=60, fmax=350, sr=sr, frame_length=1024, hop_length=hop)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 >= 60) & (f0 <= 350)]
        if f0.size > 0:
            f0_mean = float(np.mean(f0))
            f0_std = float(np.std(f0))
    except Exception:
        pass

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop, n_fft=n_fft)
    d_mfcc = librosa.feature.delta(mfcc)
    formant_transition_proxy = float(np.mean(np.abs(d_mfcc[1:4]))) if d_mfcc.shape[0] >= 4 else float(np.mean(np.abs(d_mfcc)))

    return {
        "speech_rate_wps": speech_rate_wps,
        "pause_ratio": pause_ratio,
        "rms_std": rms_std,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "low_pct": low_pct,
        "mid_pct": mid_pct,
        "high_pct": high_pct,
        "low_mid_ratio": low_mid_ratio,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "bandwidth_mean": bandwidth_mean,
        "tempo_bpm": tempo,
        "formant_transition_proxy": formant_transition_proxy,
    }


def build_acoustic_narrative_prompt(transcript: str, m: dict) -> str:
    """Build the exact prompt used during training (text-only, with acoustic narrative)."""
    acoustic_block = (
        "Acoustic profile (waveform-derived): "
        f"speech rate {m['speech_rate_wps']:.2f} words/s; "
        f"pauses {100.0*m['pause_ratio']:.1f}% of total duration; "
        f"pitch mean {m['f0_mean']:.1f} Hz with variability {m['f0_std']:.1f} Hz; "
        f"spectral energy distribution 0-1kHz {m['low_pct']:.1f}%, 1-4kHz {m['mid_pct']:.1f}%, >4kHz {m['high_pct']:.1f}% "
        f"(low+mid vs high ratio {m['low_mid_ratio']:.2f}); "
        f"centroid {m['centroid_mean']:.0f} Hz (std {m['centroid_std']:.0f}); "
        f"bandwidth {m['bandwidth_mean']:.0f} Hz; "
        f"tempo proxy {m['tempo_bpm']:.1f} BPM; "
        f"prosodic variability RMS std {m['rms_std']:.4f}; "
        f"articulatory transition proxy {m['formant_transition_proxy']:.4f}. "
        "When reasoning, prioritize low and mid frequency bands (0-1kHz and 1-4kHz) over the top band."
    )
    return (
        "The following is a verbatim transcript of a person's speech. "
        "Assess the linguistic and cognitive characteristics of this speech.\n\n"
        f"Transcript: {transcript}\n\n"
        f"{acoustic_block}"
    )


class MultimodalFeatureExtractor:

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sample_rate = 16000

    def extract(self, audio_path: str, segments: list, status_callback=None) -> dict:
        """
        Extracts a single MedGemma text+acoustic-narrative embedding for the
        whole audio file.  This matches the holdout evaluation granularity
        (one embedding per file) so that app scores align with published results.

        Steps:
          1. Concatenate all Pyannote segment texts into one full transcript
          2. Compute acoustic metrics on the entire waveform
          3. Build one acoustic-narrative prompt and obtain one 2560-dim embedding

        Returns:
          'medgemma': ndarray of shape (1, 2560)
        """

        def log(msg, progress_val=None, progress_desc=None):
            print(msg)
            if status_callback:
                status_callback(msg, progress_val, progress_desc)

        # Concatenate all segment texts into a single transcript for this file
        full_transcript = " ".join(
            seg.get("text", "") for seg in segments
        ).strip()
        log(f"Combined {len(segments)} Pyannote segments into full transcript ({len(full_transcript)} chars)...")

        # Load MedGemma
        # ---------------------------------------------------------------------
        # Training used TEXT-ONLY input: transcript + computed acoustic metrics
        # formatted as a natural-language "acoustic narrative". No images.
        log("Loading MedGemma 4-bit model...", progress_val=0.4, progress_desc="Loading MedGemma...")

        from transformers import AutoProcessor, AutoModelForImageTextToText
        from transformers import BitsAndBytesConfig

        model_name = "google/medgemma-4b-it"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
        )
        medgemma = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            quantization_config=bnb_config
        )
        processor = AutoProcessor.from_pretrained(model_name)
        medgemma.eval()

        # Load full audio with librosa for acoustic metric computation
        log("Loading audio for acoustic analysis...")
        y_full, sr_full = librosa.load(audio_path, sr=16000, mono=True)

        # Compute acoustic metrics over the whole waveform
        log("Computing whole-file acoustic metrics...", progress_val=0.5, progress_desc="Acoustic analysis")
        metrics = compute_acoustic_metrics(y_full, sr_full, full_transcript)
        if metrics is None:
            log("Audio too short for acoustic analysis, returning zero embedding")
            del medgemma
            free_vram()
            return {"medgemma": np.zeros((1, 2560), dtype=np.float32)}

        # Build the exact prompt used during training (text-only, no image)
        prompt = build_acoustic_narrative_prompt(full_transcript, metrics)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        log("Running MedGemma text+acoustic-narrative extraction (whole file)...",
            progress_val=0.6, progress_desc="Extracting speech features")
        try:
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(medgemma.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = medgemma(**inputs, output_hidden_states=True)

            last_hidden = outputs.hidden_states[-1]
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            emb = pooled.squeeze(0).float().cpu().numpy()

        except Exception as e:
            log(f"Error computing MedGemma embedding: {e}")
            emb = np.zeros(2560, dtype=np.float32)

        # Clean up
        log("Unloading MedGemma...")
        del medgemma
        free_vram()

        log("Feature extraction complete.")
        return {
            "medgemma": emb.reshape(1, 2560),  # (1, 2560)
        }
