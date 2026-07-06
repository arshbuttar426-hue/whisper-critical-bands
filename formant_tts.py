"""Formant TTS: text → phonemes → formant resonators → Whisper-validated speech."""
from __future__ import annotations
import math, os, struct, subprocess, sys, tempfile, wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formant_predictor import PHONEME_TABLE, WORD_PHONEMES
from coval_bench.metrics import compute_wer

SAMPLE_RATE = 16000

def write_wav(path, samples):
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        raw = b"".join(struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in samples)
        w.writeframes(raw)

class Resonator:
    def __init__(self, fc, bw, g=1.0, sr=SAMPLE_RATE):
        R = math.exp(-math.pi * bw / sr)
        t = 2.0 * math.pi * fc / sr
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

def sine_sweep(f_start, f_end, n_samples, amplitude=0.15):
    """Sine sweep from f_start to f_end over n_samples."""
    out = []
    phase = 0.0
    for j in range(n_samples):
        frac = j / n_samples
        freq = f_start + (f_end - f_start) * frac
        phase += freq / SAMPLE_RATE
        if phase >= 1.0:
            phase -= math.floor(phase)
        env = min(1.0, frac/0.05) if frac < 0.05 else (min(1.0, (1.0-frac)/0.15) if frac > 0.85 else 1.0)
        out.append(math.sin(2.0 * math.pi * phase) * amplitude * env)
    return out

def synthesize_phrase(phrase):
    """Generate speech by stacking sine tones at formant frequencies (the plaster cast)."""
    words = phrase.lower().split()
    all_phonemes = []
    for w in words:
        phs = WORD_PHONEMES.get(w)
        if phs:
            all_phonemes.extend(phs)

    if not all_phonemes:
        return []

    phoneme_dur = int(0.10 * SAMPLE_RATE)
    gap_dur = int(0.01 * SAMPLE_RATE)
    amplitude = 0.12  # per formant, so 3 formants = 0.36 total

    output = [0.0] * (len(all_phonemes) * (phoneme_dur + gap_dur))

    for idx, ph in enumerate(all_phonemes):
        model = PHONEME_TABLE.get(ph)
        if model is None:
            continue

        formants = model.formants[:3] if model.formants else [model.centroid or 500]
        bws = model.bw_formants[:3] if model.bw_formants else [150, 200, 300]
        centroid = model.centroid

        # Build a composite signal: sine at each formant center frequency
        start = idx * (phoneme_dur + gap_dur)

        # Get previous formants for sweep transition
        prev_formants = formants  # default: no sweep
        if idx > 0:
            prev_model = PHONEME_TABLE.get(all_phonemes[idx - 1])
            if prev_model and prev_model.formants:
                pf = prev_model.formants[:3]
                while len(pf) < 3:
                    pf.append(pf[-1] if pf else 500)
                prev_formants = pf

        for fi in range(min(3, len(formants))):
            f_cur = formants[fi]
            f_prev = prev_formants[fi] if fi < len(prev_formants) else f_cur
            bw = bws[fi] if fi < len(bws) else 200
            sweep = sine_sweep(f_prev, f_cur, phoneme_dur, amplitude=amplitude)
            for j in range(phoneme_dur):
                output[start + j] += sweep[j]

        # Add centroid for fricatives/stops
        if centroid and model.phoneme_class in ("fricative", "stop"):
            sweep_c = sine_sweep(centroid, centroid, phoneme_dur, amplitude=amplitude * 0.5)
            for j in range(phoneme_dur):
                output[start + j] += sweep_c[j]

    # Normalize
    pk = max(abs(v) for v in output) or 1.0
    output = [v / pk * 0.9 for v in output]
    return output

if __name__ == "__main__":
    import whisper
    model = whisper.load_model("tiny")
    desktop = os.path.expanduser("~/Desktop")

    for phrase in ["good morning", "glow", "i am a superman", "hello world"]:
        samples = synthesize_phrase(phrase)
        if not samples:
            print(f"{phrase}: no phonemes")
            continue

        pk = max(abs(v) for v in samples) or 1.0
        samples = [v / pk * 0.9 for v in samples]

        fname = f"{desktop}/formant_tts_{phrase.replace(' ', '_')}.wav"
        write_wav(fname, samples)

        p = tempfile.mktemp(suffix=".wav")
        write_wav(p, samples)
        r = model.transcribe(p, language="en", task="transcribe")
        os.unlink(p)
        text = r.get("text", "").strip().lower().rstrip(".,!?")
        w = compute_wer(phrase, text)
        wer = w.wer_percentage if w else 0
        print(f'{phrase}: \"{text}\" WER={wer}%')

        # Compare with reference say audio
        subprocess.run(["say", "-o", p, "--data-format=LEI16@16000", phrase], capture_output=True)
        with wave.open(p) as wf:
            frames = wf.readframes(wf.getnframes())
            ref = [struct.unpack("<h", frames[i:i+2])[0] / 32767.0 for i in range(0, len(frames), 2)]
        os.unlink(p)
        p_ref = tempfile.mktemp(suffix="_ref.wav")
        write_wav(p_ref, ref)
        r_ref = model.transcribe(p_ref, language="en", task="transcribe")
        os.unlink(p_ref)
        ref_text = r_ref.get("text", "").strip().lower().rstrip(".,!?")
        w_ref = compute_wer(phrase, ref_text)
        print(f'  say:    \"{ref_text}\" WER={w_ref.wer_percentage if w_ref else 0}%')

    print("\nDone. Files on Desktop.")
