"""FormantNotchPredictor — compute critical notch frequency from phoneme formants.
No empirical sweep needed. Models spectral energy distribution → Mel filterbank projection
→ criticality score. Covers all major English phoneme classes.

Usage:
  predictor = FormantNotchPredictor()
  result = predictor.predict_for_word("a")
  print(result.critical_freq, result.confidence, result.phoneme_class)
"""
from __future__ import annotations
import math, dataclasses
from typing import Optional

# ── Mel Filterbank (Whisper: 80 bins, 0-8000Hz, 16kHz) ──

MEL_MIN_HZ = 0.0
MEL_MAX_HZ = 8000.0
N_MELS = 80
MEL_MIN = 0.0
MEL_MAX = 2595.0 * math.log10(1.0 + MEL_MAX_HZ / 700.0)  # ≈ 2840
MEL_STEP = (MEL_MAX - MEL_MIN) / (N_MELS - 1)

def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + max(f, 1e-6) / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_bin_centers():
    return [mel_to_hz(MEL_MIN + i * MEL_STEP) for i in range(N_MELS)]

def mel_bin_edges():
    """Return (left, center, right) edges for each Mel bin."""
    centers = mel_bin_centers()
    edges = []
    for i in range(N_MELS):
        if i == 0:
            left = centers[0]
        else:
            left = (centers[i-1] + centers[i]) / 2.0
        if i == N_MELS - 1:
            right = centers[-1]
        else:
            right = (centers[i] + centers[i+1]) / 2.0
        edges.append((left, centers[i], right))
    return edges

# ── Phoneme spectral models ──

@dataclasses.dataclass
class SpectralModel:
    name: str           # ARPABET or IPA symbol
    phoneme_class: str  # "vowel", "fricative", "nasal", "approximant", "stop", "diphthong"
    formants: list[float]  # [F1, F2, F3] in Hz — only F1 used for vowels
    bw_formants: list[float]  # bandwidths corresponding to each formant
    centroid: Optional[float] = None  # for fricatives/stops
    bw_centroid: float = 800.0  # default bandwidth for centroid
    antiformants: list[float] = dataclasses.field(default_factory=list)  # for nasals

PHONEME_TABLE: dict[str, SpectralModel] = {
    # ── Vowels ──
    "iy": SpectralModel("iy", "vowel", [270, 2300, 3000], [120, 200, 300]),
    "ih": SpectralModel("ih", "vowel", [400, 1900, 2600], [120, 200, 300]),
    "eh": SpectralModel("eh", "vowel", [550, 1800, 2600], [130, 200, 300]),
    "ae": SpectralModel("ae", "vowel", [700, 1700, 2500], [150, 200, 300]),
    "ah": SpectralModel("ah", "vowel", [750, 1100, 2500], [130, 200, 300]),
    "ao": SpectralModel("ao", "vowel", [500, 900, 2500], [120, 200, 300]),
    "uh": SpectralModel("uh", "vowel", [400, 900, 2400], [120, 200, 300]),
    "uw": SpectralModel("uw", "vowel", [300, 900, 2200], [120, 200, 300]),
    "er": SpectralModel("er", "vowel", [500, 1400, 1800], [130, 200, 300]),
    "ax": SpectralModel("ax", "vowel", [500, 1500, 2500], [150, 250, 350]),  # schwa
    # ── Diphthongs ──
    "ey": SpectralModel("ey", "diphthong", [500, 1800, 2600], [150, 250, 350]),
    "ay": SpectralModel("ay", "diphthong", [700, 1200, 2500], [150, 250, 350]),
    "oy": SpectralModel("oy", "diphthong", [500, 900, 2500], [150, 250, 350]),
    "ow": SpectralModel("ow", "diphthong", [500, 1000, 2500], [150, 250, 350]),
    "aw": SpectralModel("aw", "diphthong", [700, 1100, 2500], [150, 250, 350]),
    # ── Approximants ──
    "l": SpectralModel("l", "approximant", [400, 1400, 2500], [120, 200, 300]),
    "r": SpectralModel("r", "approximant", [350, 1300, 1800], [120, 200, 300]),
    "w": SpectralModel("w", "approximant", [300, 800, 2200], [120, 200, 300]),
    "y": SpectralModel("y", "approximant", [300, 2300, 3000], [120, 200, 300]),
    # ── Nasals ──
    "m": SpectralModel("m", "nasal", [300, 1100, 2300], [150, 200, 300],
                       antiformants=[800]),
    "n": SpectralModel("n", "nasal", [300, 1200, 2400], [150, 200, 300],
                       antiformants=[1400]),
    "ng": SpectralModel("ng", "nasal", [300, 1200, 2500], [150, 200, 300],
                        antiformants=[2000]),
    # ── Fricatives ──
    "s": SpectralModel("s", "fricative", [], [], centroid=6000.0, bw_centroid=1500),
    "z": SpectralModel("z", "fricative", [300, 1500, 2500], [150, 200, 300],
                       centroid=6000.0, bw_centroid=1500),
    "sh": SpectralModel("sh", "fricative", [], [], centroid=3500.0, bw_centroid=1200),
    "zh": SpectralModel("zh", "fricative", [], [], centroid=3000.0, bw_centroid=1200),
    "f": SpectralModel("f", "fricative", [], [], centroid=7000.0, bw_centroid=1500),
    "v": SpectralModel("v", "fricative", [300, 1500, 2500], [150, 200, 300],
                       centroid=7000.0, bw_centroid=1500),
    "th": SpectralModel("th", "fricative", [], [], centroid=7000.0, bw_centroid=1500),
    "dh": SpectralModel("dh", "fricative", [300, 1500, 2500], [150, 200, 300],
                        centroid=7000.0, bw_centroid=1500),
    "hh": SpectralModel("hh", "fricative", [], [], centroid=2000.0, bw_centroid=1000),
    # ── Stops ──
    "p": SpectralModel("p", "stop", [], [], centroid=1400.0, bw_centroid=600),
    "b": SpectralModel("b", "stop", [200, 800, 2000], [150, 200, 300],
                       centroid=1400.0, bw_centroid=600),
    "t": SpectralModel("t", "stop", [], [], centroid=4000.0, bw_centroid=1000),
    "d": SpectralModel("d", "stop", [200, 800, 2000], [150, 200, 300],
                       centroid=4000.0, bw_centroid=1000),
    "k": SpectralModel("k", "stop", [], [], centroid=2200.0, bw_centroid=800),
    "g": SpectralModel("g", "stop", [200, 800, 2000], [150, 200, 300],
                       centroid=2200.0, bw_centroid=800),
    # ── Affricates ──
    "ch": SpectralModel("ch", "fricative", [], [], centroid=3500.0, bw_centroid=1200),
    "jh": SpectralModel("jh", "fricative", [], [], centroid=3500.0, bw_centroid=1200),
}

# ── Word-to-phoneme map (ARPABET, covers test phrases) ──

WORD_PHONEMES: dict[str, list[str]] = {
    "i": ["ay"],
    "a": ["ax"],
    "am": ["ae", "m"],
    "superman": ["s", "uw", "p", "er", "m", "ae", "n"],
    "glow": ["g", "l", "ow"],
    "the": ["dh", "ax"],
    "quick": ["k", "w", "ih", "k"],
    "brown": ["b", "r", "aw", "n"],
    "fox": ["f", "aa", "k", "s"],
    "good": ["g", "uh", "d"],
    "morning": ["m", "ao", "r", "n", "ih", "ng"],
    "hello": ["hh", "ax", "l", "ow"],
    "world": ["w", "er", "l", "d"],
    "i'm": ["ay", "m"],
    "you": ["y", "uw"],
    "we": ["w", "iy"],
    "they": ["dh", "ey"],
    "she": ["sh", "iy"],
    "he": ["hh", "iy"],
    "it": ["ih", "t"],
    "this": ["dh", "ih", "s"],
    "that": ["dh", "ae", "t"],
    "is": ["ih", "z"],
    "are": ["aa", "r"],
    "not": ["n", "aa", "t"],
    "can": ["k", "ae", "n"],
    "will": ["w", "ih", "l"],
    "be": ["b", "iy"],
    "have": ["hh", "ae", "v"],
    "do": ["d", "uw"],
    "what": ["w", "ah", "t"],
    "when": ["w", "eh", "n"],
    "where": ["w", "eh", "r"],
    "why": ["w", "ay"],
    "how": ["hh", "aw"],
    "see": ["s", "iy"],
    "go": ["g", "ow"],
    "come": ["k", "ah", "m"],
    "say": ["s", "ey"],
    "know": ["n", "ow"],
}

# ── Gaussian spectral envelope model ──

def gaussian_envelope(f, f0, bw):
    """Normalized Gaussian centered at f0 with bandwidth bw.
    Returns value at frequency f (arbitrary units, 0.0-1.0)."""
    sigma = bw / 2.355  # FWHM → sigma
    if sigma < 1.0:
        sigma = 1.0
    return math.exp(-0.5 * ((f - f0) / sigma) ** 2)

def spectral_energy(model: SpectralModel, f, fricative_high_shelf=True):
    """Compute spectral energy at frequency f for given phoneme model."""
    e = 0.0
    if model.phoneme_class in ("vowel", "diphthong", "approximant", "nasal"):
        for f0, bw in zip(model.formants, model.bw_formants):
            e += gaussian_envelope(f, f0, bw)
    if model.centroid is not None:
        e += gaussian_envelope(f, model.centroid, model.bw_centroid) * 0.5
    for af in model.antiformants:
        e -= gaussian_envelope(f, af, 200) * 0.3
    return max(e, 1e-6)

# ── Criticality scoring ──

@dataclasses.dataclass
class NotchPrediction:
    phoneme: str
    phoneme_class: str
    critical_freq: float
    confidence: float
    bin_idx: int
    mel_energy_distribution: list[tuple[float, float]]  # (freq, energy)

def formant_uniqueness(freq_hz):
    """Compute uniqueness score for a frequency: my_energy / total_energy.
    Higher = more unique to a specific phoneme = better notch target."""
    total = 0.0
    counts = {}  # model -> energy at this freq
    for key, model in PHONEME_TABLE.items():
        e = spectral_energy(model, freq_hz)
        if e > 0.01:
            counts[key] = e
            total += e
    return counts, total

# Precompute uniqueness for each formant frequency of each phoneme
FORMANT_UNIQUENESS_CACHE: dict[str, dict[str, float]] = {}
for key, model in PHONEME_TABLE.items():
    scores = {}
    for f0 in model.formants:
        if f0 > 0:
            counts, total = formant_uniqueness(f0)
            my_e = counts.get(key, 0.0)
            scores[f0] = my_e / max(total, 1e-6) if total > 0 else 0.0
    if model.centroid is not None:
        counts, total = formant_uniqueness(model.centroid)
        my_e = counts.get(key, 0.0)
        scores[model.centroid] = my_e / max(total, 1e-6) if total > 0 else 0.0
    FORMANT_UNIQUENESS_CACHE[key] = scores


def pick_critical_formant(model: SpectralModel, scores: dict[float, float],
                          mode: str = "anti_hallucination") -> tuple[float, float]:
    """Pick the critical formant frequency for a phoneme.
    
    mode="anti_hallucination": pick most unique formant (must be in chain trajectory).
    mode="degrade": pick lowest effective formant (min collateral, even if not in trajectory).
    """
    f1 = model.formants[0] if len(model.formants) >= 1 else None
    f2 = model.formants[1] if len(model.formants) >= 2 else None
    cls = model.phoneme_class
    fallback_lowest = mode == "degrade"

    if cls in ("vowel",):
        if f2 and f2 > 1800 and f1 and f1 in scores:
            return f1, scores[f1]
        elif f2 and f2 < 1200 and f2 in scores:
            return f2, scores[f2]
        # Central vowel
        if fallback_lowest and scores:
            best = min(scores.keys(), key=lambda f: f)  # lowest = min collateral
            return best, scores[best]
        if scores:
            best = max(scores, key=lambda f: (scores[f], -f))  # most unique = in trajectory
            return best, scores[best]
    elif cls == "diphthong":
        # Diphthong quality is defined by F2 transition
        if f2 and f2 in scores:
            return f2, scores[f2]
    elif cls == "approximant":
        # F2 carries the distinguishing articulatory feature
        if f2 and f2 in scores:
            return f2, scores[f2]
    elif cls == "fricative":
        if model.centroid and model.centroid in scores:
            return model.centroid, scores[model.centroid]
    elif cls == "stop":
        if model.centroid and model.centroid in scores:
            return model.centroid, scores[model.centroid]
        # Fallback: highest formant
        if scores:
            return max(scores.items(), key=lambda x: x[1])
    elif cls == "nasal":
        # Lowest nasal formant carries the nasal murmur
        if f1 and f1 in scores:
            return f1, scores[f1]

    # Ultimate fallback: most unique formant, tie-break with lowest freq
    if scores:
        best = max(scores, key=lambda f: (scores[f], -f))
        return best, scores[best]
    return 400.0, 0.0


def predict_phoneme(phoneme_key: str, mode: str = "anti_hallucination") -> Optional[NotchPrediction]:
    """Predict critical notch frequency for a single phoneme.
    mode="anti_hallucination": pick most unique formant (must be in chain trajectory).
    mode="degrade": pick lowest effective formant (min collateral damage).
    """
    model = PHONEME_TABLE.get(phoneme_key)
    if model is None:
        return None

    scores = FORMANT_UNIQUENESS_CACHE.get(phoneme_key, {})
    if not scores:
        return None

    best_freq, best_score = pick_critical_formant(model, scores, mode=mode)

    # Map to nearest Mel bin
    edges = mel_bin_edges()
    best_idx = min(range(N_MELS), key=lambda i: abs(edges[i][1] - best_freq))
    best_mel_freq = round(edges[best_idx][1])

    # Compute energy distribution for display
    energies = [spectral_energy(model, edges[i][1]) for i in range(N_MELS)]
    dist = [(edges[i][1], energies[i]) for i in range(N_MELS)]

    return NotchPrediction(
        phoneme=model.name,
        phoneme_class=model.phoneme_class,
        critical_freq=best_mel_freq,
        confidence=round(best_score, 3),
        bin_idx=best_idx,
        mel_energy_distribution=dist,
    )

def predict_for_word(word: str, mode: str = "anti_hallucination") -> list[NotchPrediction]:
    """Predict critical notch for each phoneme in a word."""
    phonemes = WORD_PHONEMES.get(word.lower(), [])
    if not phonemes:
        return []
    return [p for ph in phonemes if (p := predict_phoneme(ph, mode=mode)) is not None]

def predict_for_phrase(phrase: str, mode: str = "anti_hallucination") -> dict[str, list[NotchPrediction]]:
    """Predict critical notches for each word in a phrase."""
    words = phrase.lower().split()
    return {w: predict_for_word(w, mode=mode) for w in words}

# ── Self-test against empirical data ──

def self_test():
    print("═" * 55)
    print("FormantNotchPredictor — Self-Test vs Empirical Data")
    print("═" * 55)

    # Known empirical results
    # NOTE: /ax/ (schwa) is special — ANY notch 400-3400Hz removes it.
    # The predictor picks the most unique formant (F2=1507Hz) which IS in range.
    # The minimal notch is F1=400-450Hz but the predictor picks F2 for best uniqueness.
    empirical = {
        "a": {"expected": (400, 3400), "phoneme": "ax", "class": "vowel", "note": "all notches work"},
        "glow_l": {"expected": (1371, 1450), "phoneme": "l", "class": "approximant"},
        "glow_g": {"expected": (2000, 2300), "phoneme": "g", "class": "stop"},
        "glow_ow": {"expected": (900, 1100), "phoneme": "ow", "class": "diphthong"},
    }

    test_cases = [
        ("a (schwa)", "ax"),
        ("l (lateral)", "l"),
        ("g (velar stop)", "g"),
        ("ow (diphthong)", "ow"),
        ("iy (high front)", "iy"),
        ("ae (low front)", "ae"),
        ("s (fricative)", "s"),
        ("uw (high back)", "uw"),
        ("r (rhotic)", "r"),
        ("m (nasal)", "m"),
    ]

    print(f"\n{'Label':<25} {'Class':<15} {'F1':<8} {'Predicted':<12} {'Empirical':<12} {'Match':<8}")
    print("-" * 85)
    for label, phone in test_cases:
        pred = predict_phoneme(phone)
        if pred is None:
            print(f"{label:<25} {'N/A':<15} {'N/A':<8} {'N/A':<12}")
            continue
        model = PHONEME_TABLE.get(phone, SpectralModel("", "", [], []))
        f1 = model.formants[0] if model.formants else model.centroid or 0

        # Check against empirical data
        emp_entry = next((v for v in empirical.values() if v["phoneme"] == phone), None)
        if emp_entry:
            lo, hi = emp_entry["expected"]
            match = "✓" if lo <= pred.critical_freq <= hi else "✗"
            print(f"{label:<25} {pred.phoneme_class:<15} {f1:<8.0f} {pred.critical_freq:<5}Hz (b{str(pred.bin_idx):>2}){' ':>2} {lo}-{hi}Hz       {match:<8}")
        else:
            print(f"{label:<25} {pred.phoneme_class:<15} {f1:<8.0f} {pred.critical_freq:<5}Hz (b{str(pred.bin_idx):>2}){' ':>2} (no data)       {'~':<8}")

    # Full phrase prediction
    print(f"\n═" * 55)
    print("Phrase: \"i am a superman\"")
    print("═" * 55)
    results = predict_for_phrase("i am a superman")
    for word, preds in results.items():
        if preds:
            criticals = [f"{p.critical_freq}Hz ({p.phoneme_class}:{p.phoneme})" for p in preds]
            print(f"  {word}: {', '.join(criticals)}")
        else:
            print(f"  {word}: (no phoneme data)")

    # All phoneme predictions
    print(f"\n═" * 55)
    print("Full phoneme table — critical band predictions")
    print("═" * 55)
    print(f"{'Phoneme':<10} {'Class':<14} {'F1':<8} {'F2':<8} {'Critical':<12} {'Conf':<8}")
    print("-" * 60)
    for key, model in sorted(PHONEME_TABLE.items()):
        pred = predict_phoneme(key)
        if pred:
            f1 = model.formants[0] if model.formants else model.centroid or 0
            f2 = model.formants[1] if len(model.formants) >= 2 else 0
            print(f"{key:<10} {model.phoneme_class:<14} {f1:<8.0f} {f2:<8.0f} {pred.critical_freq:<5}Hz{'':>3} {pred.confidence:<8}")


if __name__ == "__main__":
    self_test()
