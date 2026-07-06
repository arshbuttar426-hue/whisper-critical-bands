"""Critical band mapper + multi-band plaster cast synthesizer.
For each phrase: systematically notch each frequency → measure WER impact →
extract envelopes for critical bands → synthesize all critical bands simultaneously.

Domains: Homeostatic (notch→WER→measure), Mycelial (distributed bands),
Chemical (catalytic per-band templates), VCG (prune non-critical bands)
"""
from __future__ import annotations
import json, math, os, struct, subprocess, sys, tempfile, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coval_bench.metrics import compute_wer

SAMPLE_RATE = 16000

# ── Audio I/O ────────────────────────────────────────────────

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

# ── Resonator ────────────────────────────────────────────────

class Resonator:
    """IIR bandpass/notch filter."""
    def __init__(self, fc, bw=100, g=1.0):
        R = math.exp(-math.pi * bw / SAMPLE_RATE)
        t = 2.0 * math.pi * fc / SAMPLE_RATE
        self.a1 = -2.0 * R * math.cos(t)
        self.a2 = R * R
        self.g = g
        self.y1 = self.y2 = 0.0
    def process(self, x):
        y = x - self.a1*self.y1 - self.a2*self.y2
        self.y2, self.y1 = self.y1, y
        return y * self.g
    def reset(self):
        self.y1 = self.y2 = 0.0

def notch_filter(samples, fc, bw=100, db=-10):
    """Apply frequency notch."""
    g = 10.0 ** (db / 20.0)
    r = Resonator(fc, bw)
    out = np.array([x - r.process(x) * (1.0 - g) for x in samples], dtype=np.float32)
    return out

def bandpass_filter(samples, fc, bw=100):
    """Extract energy in a frequency band."""
    r = Resonator(fc, bw)
    return np.array([r.process(x) for x in samples], dtype=np.float32)

# ── Band energy envelope extraction ─────────────────────────

def band_envelope(samples, fc, bw=120):
    """Extract time-varying amplitude envelope at frequency fc.
    Uses IIR bandpass + abs + lowpass.
    Returns envelope array same length as samples.
    """
    bp = bandpass_filter(samples, fc, bw)
    env = np.abs(bp)

    # Low-pass at 30Hz
    b_lp = np.ones(5) / 5  # simple moving average
    env = np.convolve(env, b_lp, mode='same')

    return env

# ── Whisper ─────────────────────────────────────────────────

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

# ── Critical band scan ──────────────────────────────────────

def scan_critical_bands(phrase, f_min=200, f_max=4000, step=100, bw=120):
    """Sweep notch across frequencies, measure WER impact.
    Returns list of (frequency_Hz, wer_after_notch, transcription).
    """
    audio = reference_audio(phrase)
    n = len(audio)

    # Baseline
    wer_base, text_base = score_phrase(audio, phrase)
    print(f"  Baseline: WER={wer_base}% \"{text_base}\"")

    results = []
    f = f_min
    while f <= f_max:
        notched = notch_filter(audio, f, bw=bw, db=-15)
        wer, text = score_phrase(notched, phrase)
        damage = wer - wer_base
        results.append((f, wer, damage, text))
        flag = " *** DAMAGE" if damage > 0 else ""
        print(f"  notch {f:4d}Hz: WER={wer}% \"{text}\"{flag}")
        f += step

    return results, audio

def find_critical_frequencies(results, threshold=0):
    """From scan results, extract frequencies where WER increases (damage > threshold)."""
    critical = [(f, wer, damage, text) for f, wer, damage, text in results if damage > threshold]
    return critical

# ── Multiband synthesis ─────────────────────────────────────

def synthesize_from_bands(reference, frequencies, bws=None):
    """Synthesize speech from critical band sine tones.
    For each critical frequency, extract the time-varying envelope
    from the reference audio, then modulate a sine tone at that frequency.
    Sum all bands.
    """
    if bws is None:
        bws = [120] * len(frequencies)

    n = len(reference)
    output = np.zeros(n, dtype=np.float64)

    for idx, fc in enumerate(frequencies):
        bw = bws[idx] if idx < len(bws) else 120
        env = band_envelope(reference, fc, bw=bw)

        # Generate sine at center frequency, modulated by envelope
        t = np.arange(n) / SAMPLE_RATE
        carrier = np.sin(2.0 * math.pi * fc * t)
        band_out = carrier * env * 0.5  # half amplitude per band

        output += band_out

    # Normalize
    pk = np.max(np.abs(output)) or 1.0
    output = (output / pk * 0.95).astype(np.float32)
    return output

def minimal_band_set(phrase, frequencies, reference):
    """VCG-style: find the minimum set of frequency bands for 0% WER.
    Start from empty, add the band that improves WER most each round.
    """
    critical = list(frequencies)
    active = []
    remaining = list(critical)

    print(f"\n  Finding minimal set from {len(critical)} candidates...")

    while remaining:
        best_f = None
        best_wer = 100.0
        best_text = ""

        for cand in remaining:
            test_set = active + [cand]
            syn = synthesize_from_bands(reference, test_set)
            wer, text = score_phrase(syn, phrase)

            if wer < best_wer:
                best_wer = wer
                best_f = cand
                best_text = text

            if wer == 0:
                break  # early exit

        if best_f is None:
            break

        active.append(best_f)
        remaining.remove(best_f)
        print(f"  +{best_f:4d}Hz: WER={best_wer}% \"{best_text}\"")

        if best_wer == 0:
            print(f"  → Found minimal set: {len(active)} bands")
            break

    return active

# ── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    phrases = sys.argv[1:] or ["glow", "good morning"]
    desktop = os.path.expanduser("~/Desktop")
    all_results = {}

    for phrase in phrases:
        print(f"\n=== {phrase} ===")

        # Phase 1: scan all frequencies for critical bands
        results, audio = scan_critical_bands(phrase, f_min=200, f_max=4000, step=100)

        # Phase 2: find which bands are critical (damage > 0)
        critical = find_critical_frequencies(results, threshold=0)
        critical_freqs = [f for f, _, _, _ in critical]
        print(f"  Critical frequencies ({len(critical_freqs)}): {critical_freqs}")

        all_results[phrase] = {
            "critical_frequencies": critical_freqs,
            "scan": [{"f": f, "wer": w, "damage": d} for f, w, d, _ in results],
        }

        # Phase 3: if critical bands found, find minimal set for synthesis
        if critical_freqs:
            best_set = minimal_band_set(phrase, critical_freqs, audio)

            # Synthesize from minimal set
            syn = synthesize_from_bands(audio, best_set)
            path = f"{desktop}/critical_{phrase.replace(' ', '_')}.wav"
            write_audio(path, syn)
            wer_final, text_final = score_phrase(syn, phrase)
            print(f"\n  FINAL: WER={wer_final}% \"{text_final}\" | {len(best_set)} bands stored")

            all_results[phrase]["best_set"] = best_set
            all_results[phrase]["wer"] = wer_final

    with open(f"{desktop}/critical_bands_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nDone. Files on Desktop.")
