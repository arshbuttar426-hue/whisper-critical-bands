# Critical Band Interference in Whisper

Causal mapping of Whisper's spectral vulnerabilities — narrowband notching (200–4000 Hz) reveals critical frequency bands where spectral removal causes misrecognition, with a dynamic anchor chain that restores perfect recognition.

## Paper

- **PDF:** [`whisper_critical_bands.pdf`](whisper_critical_bands.pdf)
- **LaTeX source:** [`whisper_critical_bands.tex`](whisper_critical_bands.tex)
- **Website:** https://arshbuttar426-hue.github.io/whisper-critical-bands/

## Key Findings

| Finding | Detail |
|---------|--------|
| Critical band map | 28/38 frequencies damage "glow" — vowel F1, /l/ formant, /g/ burst zones |
| /g/ burst zone | 2000–2900 Hz — notching deletes onset consonant across /g/, /bl/, /sl/ |
| /l/ formant zone | 1500–1800 Hz — notching deletes lateral approximant |
| Notch depth threshold | −2 dB at 2000 Hz causes misrecognition |
| Dynamic anchor restoration | 30/30 trials at 0% WER — trajectory-following sine patch |
| Synthesis barrier | 10 approaches tested — all pure-synthesis approaches fail (100% WER) |
| Schwa vulnerability | /ə/ is non-restorable — 25% WER floor for phrases containing it |

## Code Overview

### [`dynamic_anchor.py`](dynamic_anchor.py)
The core restoration engine:
- `dominant_trajectory()` — extract formant trajectory (40-band filterbank, 10 ms hops)
- `generate_anchored_chain()` — trajectory-following sine sweep with handoff mutes
- `notch()` — narrowband IIR notch filter (200 Hz bandwidth)
- `find_critical_notch()` — brute-force notch sweep to find damaging frequencies
- Runs 30-trial validation with configurable notch parameters

Run the built-in demo:
```bash
python3 dynamic_anchor.py
```

### [`critical_band_mapper.py`](critical_band_mapper.py)
Systematic critical band scanning:
- `scan_critical_bands()` — sweep 200–4000 Hz at 100 Hz steps, measure WER per frequency
- `synthesize_from_bands()` — multiband sine synthesis from extracted envelopes
- `minimal_band_set()` — VCG-style greedy selection of minimum bands for recognition

Run for any word:
```bash
python3 critical_band_mapper.py glow
python3 critical_band_mapper.py blow slow black
```

### [`multiband_plaster.py`](multiband_plaster.py)
Iterative multi-chain synthesis:
- `extract_trajectory()` — dominant formant trajectory extraction
- `synthesize_chain()` — generate sine sweep from trajectory
- `extract_residual_trajectory()` — extract secondary trajectories from residuals
- `synthesize_phrase()` — iterative chain stacking up to 4 iterations

Run:
```bash
python3 multiband_plaster.py glow
```

### [`formant_tts.py`](formant_tts.py)
Phoneme-to-formant TTS:
- `synthesize_phrase()` — generates sine sweeps at formant frequencies from phoneme table
- Uses `formant_predictor.py` for phoneme→frequency mappings

Run:
```bash
python3 formant_tts.py
```

### [`formant_predictor.py`](formant_predictor.py)
Phoneme formant database:
- `PHONEME_TABLE` — formant frequencies and bandwidths for English phonemes
- `WORD_PHONEMES` — pronunciation mappings for test words
- `FormantNotchPredictor` — predicts critical notch frequency from phoneme formants

## Requirements

```bash
pip3 install numpy scipy whisper openai-whisper
git clone https://github.com/coval-bench/coval.git
```

Requires macOS (for the `say` TTS command used to generate reference audio).

## Reproducing the Experiments

### Critical band map for any word:
```bash
python3 critical_band_mapper.py glow
# Output: WER at each frequency (200–4000 Hz, 100 Hz steps)
# Files: ~/Desktop/critical_glow.wav
```

### Notch+chain restoration:
```bash
python3 dynamic_anchor.py
# Output: 30-trial validation with WER per trial
# Tests "i am a superman" with automatic critical band detection
```

### Notch depth threshold:
```bash
python3 -c "
from critical_band_mapper import *
for db in [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -12, -15, -20]:
    notched = notch_filter(reference_audio('glow'), 2000, bw=200, db=db)
    wer, text = score_phrase(notched, 'glow')
    print(f'{db:3d}dB: WER={wer}% \"{text}\"')
"
```

### Synthesis barrier tests:
```bash
# Multi-chain residual
python3 multiband_plaster.py glow

# Formant TTS
python3 formant_tts.py

# Pure sine sweep
python3 word_frequency_map.py
```

## Citation

```bibtex
@misc{buttar2026critical,
  title={Critical Band Interference in Whisper: Causal Mapping, Restoration, and the Synthesis Barrier},
  author={Arsh Buttar},
  year={2026},
  howpublished={\url{https://github.com/arshbuttar426-hue/whisper-critical-bands}}
}
```

## License

MIT
