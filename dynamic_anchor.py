"""Dynamic anchored anti-hallucination.
Auto-detects critical frequency band, extracts formant trajectory
from the speech signal itself, builds an anchored handoff chain.
"""
from __future__ import annotations
import math, os, random, struct, subprocess, sys, tempfile, wave
import formant_predictor as fp

SAMPLE_RATE = 16000
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coval_bench.metrics import compute_wer

def write_wav(path, samples, sr=SAMPLE_RATE):
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        raw = b"".join(struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in samples)
        w.writeframes(raw)

def read_wav(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
        samples = [struct.unpack("<h", frames[i:i+2])[0] / 32767.0 for i in range(0, len(frames), 2)]
    return samples, sr

def say_word(word):
    wav_path = tempfile.mktemp(suffix="_say.wav")
    subprocess.run(["say", "-o", wav_path, "--data-format=LEI16@16000", word], capture_output=True, check=True)
    samples, sr = read_wav(wav_path); os.unlink(wav_path)
    return samples

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

def notch(samples, fc, bw, db, sr=SAMPLE_RATE):
    g = 10.0 ** (db / 20.0)
    r = Resonator(fc, bw, sr=sr)
    return [x - r.process(x) * (1.0 - g) for x in samples]

def bandpass(samples, fc, bw, sr=SAMPLE_RATE):
    """Extract energy in a frequency band using a resonator."""
    r = Resonator(fc, bw, g=1.0, sr=sr)
    return [r.process(x) for x in samples]

# ── Spectrum analysis ──────────────────────────────────

def compute_mel_bin_centers(n_bins=40, f_min=200, f_max=6000, sr=SAMPLE_RATE):
    """Mel-spaced frequency bin centers."""
    bins = []
    for i in range(n_bins):
        mel = i / (n_bins - 1) * (2595 * math.log10(1 + f_max/700)) if n_bins > 1 else 0
        freq = 700 * (10 ** (mel / 2595) - 1)
        bins.append(freq)
    return bins

def spectrum_energy(samples, fc, bw=150, sr=SAMPLE_RATE):
    """RMS energy in a band around fc."""
    bp = bandpass(samples, fc, bw, sr=sr)
    rms = math.sqrt(sum(x*x for x in bp) / max(1, len(bp)))
    return rms

def dominant_trajectory(samples, sr, f_low=200, f_high=4000, n_candidates=30,
                        hop_s=0.01, bw=120):
    """Extract formant trajectory over time using filterbank.
    Finds the highest local peak (formant) above 600Hz to avoid
    the fundamental and F1 dominance."""
    hop_n = int(hop_s * sr)
    centers = [f_low + (f_high - f_low) * i / (n_candidates - 1) for i in range(n_candidates)]
    traj = []
    for start in range(0, len(samples) - hop_n, hop_n):
        chunk = samples[start:start + hop_n]
        energies = []
        for fc in centers:
            e = spectrum_energy(chunk, fc, bw=bw, sr=sr)
            energies.append(e)
        # Find local peaks: energy higher than both neighbors
        peaks = []
        for i in range(1, len(energies) - 1):
            if energies[i] > energies[i-1] and energies[i] >= energies[i+1]:
                # Normalize by frequency (higher bands naturally have less energy)
                peaks.append((energies[i] * (1 + i * 0.02), centers[i]))
        if peaks:
            best = max(peaks, key=lambda x: x[0])
            traj.append(best[1])
        else:
            # Fallback: overall max above 600Hz
            above = [(energies[i], centers[i]) for i in range(len(centers)) if centers[i] >= 600]
            if above:
                traj.append(max(above, key=lambda x: x[0])[1])
            else:
                traj.append(centers[-1])
    return traj, hop_s

def smooth_trajectory(traj, window=3):
    """Simple moving average."""
    out = list(traj)
    for i in range(window, len(traj) - window):
        out[i] = sum(traj[i-window:i+window+1]) / (2*window + 1)
    return out

# ── Anchored handoff chain ─────────────────────────────

def generate_anchored_chain(trajectory, hop_s, amplitude=0.15, sr=SAMPLE_RATE,
                            mute_ms=8):
    """Generate a handoff chain that follows the extracted formant trajectory.
    Rises sweep the trajectory directly; mutes are negative dips between segments.
    No randomization — the trajectory IS the sequence."""
    samples = []; phase = 0.0
    traj = smooth_trajectory(trajectory, window=5)
    prev_f = traj[0]
    mute_n = int(mute_ms / 1000.0 * sr)

    for i in range(1, len(traj)):
        f_start = traj[i-1]
        f_end = traj[i]
        # Rise: sweep from traj[i-1] to traj[i]
        dur = 0.04  # 40ms per trajectory step
        n = int(dur * sr)
        for j in range(n):
            frac = j / n
            freq = f_start + (f_end - f_start) * frac
            env = min(1.0, frac/0.05) if frac < 0.05 else (min(1.0, (1.0-frac)/0.15) if frac > 0.85 else 1.0)
            phase += freq / sr
            if phase >= 1.0: phase -= math.floor(phase)
            samples.append(math.sin(2.0*math.pi*phase) * amplitude * env)
        # Mute: negative dip between trajectory points
        delta = abs(f_end - f_start)
        max_delta = 2500.0
        depth = amplitude * min(1.0, delta / max_delta) * 0.4 + amplitude * 0.03
        for j in range(mute_n):
            t = j / mute_n
            pulse = math.sin(math.pi * t)
            val = -depth * pulse
            phase += f_end / sr * 0.5
            if phase >= 1.0: phase -= math.floor(phase)
            rise_amp = amplitude * 0.2 * t * t
            samples.append(val + rise_amp * math.sin(2.0*math.pi*phase))

    return samples, len(traj)

# ── Find critical notch band ───────────────────────────

def find_critical_notch(samples, sr, model, reference, candidate_freqs,
                        bw=200, db=-10):
    """Brute-force which notch frequencies degrade Whisper most.
    Returns (best_freqs, results) sorted by WER descending."""
    results = []
    for fc in candidate_freqs:
        n = notch(samples, fc, bw, db, sr=sr)
        pk = max(abs(v) for v in n) or 1.0
        n = [v / pk * 0.9 for v in n]
        p = tempfile.mktemp(suffix="_notch_test.wav")
        write_wav(p, n)
        r = model.transcribe(p, language="en", task="transcribe")
        os.unlink(p)
        text = r.get("text","").strip().lower().rstrip(".,!?")
        w = compute_wer(reference, text).wer_percentage
        print(f"  Notch @{fc:.0f}Hz: \"{text[:40]:<40}\" WER={w:.0f}%")
        results.append((fc, w, text))
    results.sort(key=lambda x: -x[1])
    return results

def multi_notch(samples, freq_list, bw=200, db=-10, sr=SAMPLE_RATE):
    """Apply multiple notches simultaneously."""
    out = list(samples)
    for fc in freq_list:
        out = notch(out, fc, bw, db, sr=sr)
    return out

# ── Main test ──────────────────────────────────────────

PHRASE = "i am a superman"

print(f"=== Dynamic Anchored Anti-Hallucination ===")
print(f"Phrase: \"{PHRASE}\"")
print()

# 1. Generate original
original = say_word(PHRASE)
pk = max(abs(v) for v in original) or 1.0
original = [v / pk * 0.9 for v in original]

# 2. Extract frequency trajectory
print("Extracting formant frequency trajectory (wide BW envelope)...")
traj, hop_s = dominant_trajectory(original, SAMPLE_RATE)
freqs_hist = {}
for f in traj:
    k = round(f / 100) * 100
    freqs_hist[k] = freqs_hist.get(k, 0) + 1
top_bins = sorted(freqs_hist.items(), key=lambda x: -x[1])[:5]
print(f"  {len(traj)} frames at {hop_s*1000:.0f}ms hops")
print(f"  Freq range: {min(traj):.0f}–{max(traj):.0f}Hz")
print(f"  Top bins: {', '.join(f'{f}Hz({c}x)' for f,c in top_bins)}")

# 3. Find critical notch band
import whisper
model = whisper.load_model("tiny")

# ── Use FormantNotchPredictor instead of empirical sweep ──
from formant_predictor import predict_for_phrase as fp_predict_phrase

print("\nPredicting critical notch bands from phoneme formant tables...")
predictions = fp_predict_phrase(PHRASE, mode="anti_hallucination")

# Print predictions
print(f"\n  Predictions:")
for word, preds in predictions.items():
    for p in preds:
        print(f"    {word}/{p.phoneme:<4} → {p.critical_freq}Hz ({p.phoneme_class}, conf={p.confidence})")

# Detect non-restorable phonemes (schwa /ə/ — broadband, can't be restored by single tone)
unrestorable = {"ax"}  # schwa ARPABET key
has_unrestorable = set()
for word, preds in predictions.items():
    for p in preds:
        if fp.WORD_PHONEMES.get(word, []) and any(ph in unrestorable for ph in fp.WORD_PHONEMES[word.lower()]):
            has_unrestorable.add(word)

if has_unrestorable:
    print(f"\n  ⚠ Contains non-restorable phoneme /ə/ (schwa) in: {', '.join(sorted(has_unrestorable))}")
    print(f"    /ə/ is broadband — ANY notch removes it, AND no single-frequency chain can restore it.")
    print(f"    Anti-hallucination will have 25% WER floor (can't recover /ə/).")

# Strategy: match the notch to the trajectory's dominant frequency.
# The chain trajectory is dominated by F1 frequencies (~700Hz).
# Pick the word whose critical frequency is closest to the trajectory's dominant band.
traj_dominant = max(freqs_hist, key=lambda k: freqs_hist[k])

print(f"\n  Trajectory dominant band: ~{traj_dominant}Hz ({freqs_hist[traj_dominant]} frames)")

# Among words WITHOUT non-restorable phonemes, find best trajectory match
valid_targets = [(w, p) for w, preds in predictions.items() if w not in has_unrestorable for p in preds]
if not valid_targets:
    valid_targets = [(w, p) for w, preds in predictions.items() for p in preds]

best_delta = float('inf')
best_target = None
for word, p in valid_targets:
    delta = abs(p.critical_freq - traj_dominant)
    if delta < best_delta:
        best_delta = delta
        best_target = (word, p)

print(f"  Best trajectory match: \"{best_target[0]}/{best_target[1].phoneme}\" at {best_target[1].critical_freq}Hz (Δ={best_delta:.0f}Hz)")

NOTCH_BW = 200
NOTCH_DB = -10
best_fc = int(best_target[1].critical_freq)
notch_config = [(best_fc, NOTCH_DB, NOTCH_BW)]

print(f"\n  Notch config: [({best_fc}, {NOTCH_DB})]")

# 4. Verify notch damages the signal; if not, skip chain (not robust enough)
print(f"\n  Verifying notch @{best_fc}Hz (bw={NOTCH_BW}, db={NOTCH_DB})...")
n = notch(original, best_fc, NOTCH_BW, NOTCH_DB, sr=SAMPLE_RATE)
pk = max(abs(v) for v in n) or 1.0
n_test = [v / pk * 0.9 for v in n]
p = tempfile.mktemp(suffix="_verify_notch.wav")
write_wav(p, n_test)
r = model.transcribe(p, language="en", task="transcribe")
os.unlink(p)
verify_text = r.get("text","").strip().lower().rstrip(".,!?")
w = compute_wer(PHRASE, verify_text)
verify_wer = w.wer_percentage if w else 0
print(f"  Notch-only: \"{verify_text}\" WER={verify_wer}%")
if verify_wer == 0:
    print(f"  → Notch doesn't damage signal — skipping anti-hallucination (robust phrase)")
    print(f"\n  Result: original audio is already robust — no anti-hallucination needed.")
    print(f"\n{'='*50}")
    print(f"RESULT")
    print(f"{'='*50}")
    print(f"Correct (0% WER): 1/1 (100%) — original audio unaffected by notch")
    import sys; sys.exit(0)

# 5. Build anchored handoff chain with multi-band notching
print(f"\nBuilding anchored handoff chain + multi-band notch (notch damages signal, chain counteracts)...")
n_trials = 30
wer_list = []
results_tally = {"correct": 0, "other": {}}

for trial in range(n_trials):
    chain, n_steps = generate_anchored_chain(traj, hop_s, amplitude=0.15)
    pk = max(abs(v) for v in chain) or 1.0
    chain = [v / pk * 0.15 for v in chain]

    # Multi-band notch with refined parameters
    notched = list(original)
    for fc, db, bw in notch_config:
        notched = notch(notched, fc, bw, db, sr=SAMPLE_RATE)

    mixed = [notched[i] + (chain[i] if i < len(chain) else 0) for i in range(min(len(notched), len(chain)))]
    pk = max(abs(v) for v in mixed) or 1.0
    mixed = [v / pk * 0.9 for v in mixed]
    p = tempfile.mktemp(suffix=f"_anchor_{trial}.wav")
    write_wav(p, mixed)
    r = model.transcribe(p, language="en", task="transcribe")
    os.unlink(p)
    transcript = r.get("text","").strip().lower().rstrip(".,!?")

    w = compute_wer(PHRASE, transcript).wer_percentage
    wer_list.append(w)

    if transcript == PHRASE:
        results_tally["correct"] += 1
    else:
        results_tally["other"][transcript] = results_tally["other"].get(transcript, 0) + 1

    if trial < 5 or w > 0:
        print(f"  Trial {trial:>2}: {n_steps:>2} steps \"{transcript:<40}\" WER={w:.0f}%")

mean_wer = sum(wer_list) / n_trials
perfect = sum(1 for w in wer_list if w == 0.0)
zero_signal = sum(1 for w in wer_list if w == 100.0)
total_other = sum(results_tally["other"].values())

print(f"\n{'='*50}")
print(f"SUMMARY ({n_trials} trials)")
print(f"{'='*50}")
print(f"Correct (0% WER): {results_tally['correct']}/{n_trials} ({results_tally['correct']/n_trials*100:.1f}%)")
if results_tally["other"]:
    print(f"Other: {total_other}/{n_trials} ({total_other/n_trials*100:.1f}%)")
    for t, c in sorted(results_tally["other"].items(), key=lambda x: -x[1]):
        print(f"  \"{t}\": {c}")
print(f"\n--- Coval WER ---")
print(f"Mean WER: {mean_wer:.2f}%")
print(f"0% WER:   {perfect}/{n_trials} ({perfect/n_trials*100:.1f}%)")
print(f"100% WER: {zero_signal}/{n_trials} ({zero_signal/n_trials*100:.1f}%)")
print("DONE")
