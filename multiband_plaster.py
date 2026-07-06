"""Multi-band plaster cast vocoder.
Extracts dominant formant trajectory (following dynamic_anchor approach),
synthesizes sine sweep along it. If single trajectory isn't enough,
extracts secondary trajectories from the residual.

Domains: Mycelial (distributed trajectories), Chemical (catalytic templates),
Homeostatic (residual extraction until 0% WER), VCG (minimal band selection)
"""
from __future__ import annotations
import math, os, struct, subprocess, sys, tempfile, wave, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coval_bench.metrics import compute_wer

SAMPLE_RATE = 16000

def write_audio(path, samples):
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        raw = np.clip(samples * 32767, -32768, 32767).astype(np.int16).tobytes()
        w.writeframes(raw)

def load_audio(path):
    with wave.open(path) as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

def reference_audio(phrase):
    p = tempfile.mktemp(suffix="_ref.wav")
    subprocess.run(["say", "-o", p, "--data-format=LEI16@16000", phrase], capture_output=True)
    audio = load_audio(p)
    os.unlink(p)
    return audio

# ── Resonator (for bandpass energy extraction) ──────────

def bandpass_energy(chunk, fc, bw=120, sr=SAMPLE_RATE):
    """Compute RMS energy in a frequency band using IIR resonator."""
    R = math.exp(-math.pi * bw / sr)
    t = 2.0 * math.pi * fc / sr
    a1 = -2.0 * R * math.cos(t)
    a2 = R * R
    y1 = y2 = 0.0
    energy = 0.0
    for x in chunk:
        y = x - a1 * y1 - a2 * y2
        y2, y1 = y1, y
        energy += y * y
    return math.sqrt(energy / max(1, len(chunk)))

# ── Trajectory extraction ───────────────────────────────

def extract_trajectory(samples, n_candidates=40, f_low=100, f_high=4000,
                       hop_s=0.01, bw=120):
    """Extract dominant formant trajectory. Searches 100-4000Hz, applies
    2% high-frequency boost, prefers local peaks >600Hz (to avoid F1 dominance).
    Returns list of (frequency_Hz, energy) per frame.
    """
    hop_n = int(hop_s * SAMPLE_RATE)
    n_frames = max(1, (len(samples) - hop_n) // hop_n)
    centers = [f_low + (f_high - f_low) * i / (n_candidates - 1) for i in range(n_candidates)]

    trajectory = []
    for i in range(n_frames):
        chunk = samples[i * hop_n : i * hop_n + hop_n]
        energies = [bandpass_energy(chunk, fc, bw=bw) for fc in centers]

        # Find local peaks with high-frequency bias (2% per bin)
        peaks = [(energies[j] * (1 + j * 0.02), centers[j])
                 for j in range(1, len(energies) - 1)
                 if energies[j] > energies[j-1] and energies[j] >= energies[j+1]]

        if peaks:
            # Prefer peaks above 600Hz (avoid F1/fundamental)
            above_600 = [(e, f) for e, f in peaks if f >= 600]
            if above_600:
                best_freq = max(above_600, key=lambda x: x[0])[1]
                best_energy = max((e for e, _ in above_600), default=0)
            else:
                best_freq = max(peaks, key=lambda x: x[0])[1]
                best_energy = max((e for e, _ in peaks), default=0)
        else:
            above = [(energies[j], centers[j]) for j in range(len(centers))
                     if centers[j] >= 600]
            if above:
                best_freq = max(above, key=lambda x: x[0])[1]
                best_energy = max(above, key=lambda x: x[0])[0]
            else:
                above_all = [(energies[j], centers[j]) for j in range(len(centers))]
                best_freq = max(above_all, key=lambda x: x[0])[1]
                best_energy = max(above_all, key=lambda x: x[0])[0]

        trajectory.append((best_freq, best_energy))

    return trajectory, hop_s

def smooth_trajectory(traj, window=3):
    """Moving average on frequency values. traj is list of (freq, energy)."""
    vals = [t[0] for t in traj]
    out = list(vals)
    for i in range(window, len(vals) - window):
        out[i] = sum(vals[i-window:i+window+1]) / (2*window + 1)
    return out

# ── Synthesis from trajectories ─────────────────────────

def synthesize_chain(trajectory, hop_s, amplitude=0.15):
    """Generate sine sweep following a frequency trajectory (like dynamic_anchor chain)."""
    n_frames = len(trajectory)
    freqs = [t[0] for t in trajectory]
    energies = [t[1] for t in trajectory]

    # Smooth
    freqs = smooth_trajectory(trajectory, window=5)

    hop_n = int(hop_s * SAMPLE_RATE)
    n_samples = n_frames * hop_n

    frame_times = np.arange(n_frames) * hop_s
    sample_times = np.arange(n_samples) / SAMPLE_RATE
    freq_interp = np.interp(sample_times, frame_times, freqs)

    # Phase accumulation
    phase = np.cumsum(2.0 * math.pi * freq_interp / SAMPLE_RATE).astype(np.float64)
    carrier = np.sin(phase)

    # Amplitude envelope from trajectory energy
    energy_interp = np.interp(sample_times, frame_times, energies)
    # Normalize energy to [0, 1]
    pk_e = np.max(energy_interp) or 1.0
    energy_interp /= pk_e

    samples = carrier * energy_interp * amplitude

    # Fade
    fade_n = min(int(SAMPLE_RATE * 0.005), n_samples // 4)
    if fade_n > 0:
        samples[:fade_n] *= np.linspace(0, 1, fade_n)
        samples[-fade_n:] *= np.linspace(1, 0, fade_n)

    return samples.astype(np.float32)

# ── Multi-trajectory extraction (mycelial) ─────────────

def extract_residual_trajectory(samples, trajectory, hop_s, n_candidates=40,
                                f_low=600, f_high=4000, bw=120):
    """Extract trajectory from the RESIDUAL (what's left after removing
    the first trajectory's chain from the reference)."""
    # Generate the chain for the first trajectory
    chain = synthesize_chain(trajectory, hop_s, amplitude=1.0)
    # Align lengths
    n = min(len(samples), len(chain))
    residual = samples[:n] - chain[:n]
    # Extract trajectory from residual
    return extract_trajectory(residual, n_candidates, f_low, f_high, hop_s, bw)

# ── Whisper scoring ────────────────────────────────────

_whisper_model = None
def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model

def score_phrase(audio, phrase):
    model = get_whisper()
    p = tempfile.mktemp(suffix=".wav")
    write_audio(p, audio)
    r = model.transcribe(p, language="en", task="transcribe")
    os.unlink(p)
    text = r.get("text", "").strip().lower().rstrip(".,!?")
    w = compute_wer(phrase, text)
    wer = w.wer_percentage if w else 100.0
    return wer, text

# ── Main pipeline ──────────────────────────────────────

def synthesize_phrase(phrase, max_chains=4):
    """Iteratively extract and stack chains until 0% WER or max_chains reached.
    Each chain captures the dominant trajectory of the REMAINING residual.
    """
    audio = reference_audio(phrase)
    if len(audio) < 100:
        return None, 100.0, []

    chains = []
    residual = audio.copy()
    hop_s = 0.01

    for c_idx in range(max_chains):
        # Extract trajectory from current residual
        traj_raw, _ = extract_trajectory(residual, hop_s=hop_s)
        if len(traj_raw) < 3:
            break

        smoothed = smooth_trajectory(traj_raw, window=3)
        # Rebuild trajectory with smoothed frequencies and original energies
        energies = [t[1] for t in traj_raw]
        traj = list(zip(smoothed, energies))
        amp = 0.15 / (c_idx + 1)  # decreasing amplitude for later chains
        chain = synthesize_chain(traj, hop_s, amplitude=amp)
        chains.append(chain)

        # Combine all chains
        max_n = max(len(c) for c in chains)
        combined = np.zeros(max_n, dtype=np.float64)
        for c in chains:
            padded = np.pad(c, (0, max_n - len(c))) if len(c) < max_n else c[:max_n]
            combined += padded

        # Normalize
        pk = np.max(np.abs(combined)) or 1.0
        combined = (combined / pk * 0.95).astype(np.float32)

        wer, text = score_phrase(combined, phrase)
        print(f"  Chain {c_idx+1}: WER={wer}% \"{text}\"")

        if wer == 0:
            return combined, 0.0, chains

        # Compute residual for next iteration
        n = min(len(audio), len(combined))
        residual = audio[:n].copy().astype(np.float64)
        residual -= combined[:n].astype(np.float64)

    return combined, wer, chains

if __name__ == "__main__":
    phrases = sys.argv[1:] or ["glow", "good morning", "i am a superman", "hello world"]
    desktop = os.path.expanduser("~/Desktop")
    results = {}

    for phrase in phrases:
        print(f"\n=== {phrase} ===")
        syn, wer, chains = synthesize_phrase(phrase, max_chains=3)
        if syn is not None:
            path = f"{desktop}/plaster_{phrase.replace(' ', '_')}.wav"
            write_audio(path, syn)
            print(f"  FINAL: WER={wer}% | {len(chains)} chains | saved to {path}")
        results[phrase] = {"wer": wer, "n_chains": len(chains) if chains else 0}

    with open(f"{desktop}/plaster_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Results on Desktop.")
