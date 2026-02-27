# SYMBA Dataset

## Source

**Original Dataset:** ML4Sci SYMBA Project  
**Download Link:** https://alabama.box.com/s/xhgr2onrn503jyse2fs5vxtapg0oifcs

**Dataset Format:**
```
event_type : Feynman_diagram : amplitude : squared_amplitude
```

**Files:** 17 total files split across two physics models:
- **QED (Quantum Electrodynamics):** 10 files → 360 samples
- **QCD (Quantum Chromodynamics):** 7 files → 234 samples
- **Total:** 594 amplitude/squared-amplitude pairs

## Dataset Statistics

| Physics Model | Total | Train (80%) | Val (10%) | Test (10%) |
|---|---|---|---|---|
| QED | 360 | 288 | 36 | 36 |
| QCD | 234 | 187 | 23 | 24 |
| **Combined** | **594** | **475** | **59** | **60** |

### Sequence Length Distributions

**QED (shorter sequences):**
- Amplitude: min=6, median=103, mean=134.2, max=247 tokens
- Squared amplitude: min=5, median=89, mean=111.5, max=223 tokens

**QCD (longer sequences):**
- Amplitude: min=12, median=246, mean=488.7, **max=2,072 tokens**
- Squared amplitude: min=10, median=214, mean=364.1, **max=1,622 tokens**

## Preprocessing Pipeline

### 1. Index Normalization

Raw amplitudes contain arbitrary indices assigned by Feynman diagram calculators:
```
%gam_249, %sigma_304, k_43, %C_185, etc.
```

These integers carry no semantic meaning and must be normalized to canonical forms:
```
Raw:  gamma_{+%\sigma_249, %gam_165, %del_165}
Norm: gamma_{+IDX_0, IDX_1, IDX_2}
```

**Algorithm:** Deterministic left-to-right scan replaces each unique index with `IDX_0`, `IDX_1`, ..., `IDX_N` in encounter order.

**Validation:** Highest observed index across all 594 samples is 119 → vocabulary uses `IDX_0` to `IDX_119` (120 tokens).

### 2. Tokenization

**Custom physics-aware tokenizer** with greedy longest-match approach:
- **Vocabulary size:** 192 tokens
- **Unknown tokens:** 0 across 294,174 total tokens
- **Round-trip lossless:** encode → decode → retokenize produces identical sequence

**Why not BPE/SentencePiece?**
Subword tokenizers fragment physics symbols:
- `s_12` → `s`, `_`, `12` (Mandelstam variable split)
- `m_mu` → `m`, `_`, `mu` (muon mass fragmented)
- `reg_prop` → `reg`, `_`, `prop` (propagator split)

Custom tokenizer preserves semantic units as atomic tokens.

### 3. Dataset Split

**Strategy:** Stratified 80-10-10 split per physics model (seed=42)

This ensures:
- Balanced tree-level representation in all splits
- Separate evaluation for QED vs QCD performance
- Reproducible splits for fair comparison

## Processed Files

Located in `data/processed/`:

| File | Description |
|---|---|
| `tokenizer.pkl` | Custom physics-aware tokenizer (192-token vocab) |
| `qed_train.pkl` | QED training set (288 samples) |
| `qed_val.pkl` | QED validation set (36 samples) |
| `qed_test.pkl` | QED test set (36 samples) |
| `qcd_train.pkl` | QCD training set (187 samples) |
| `qcd_val.pkl` | QCD validation set (23 samples) |
| `qcd_test.pkl` | QCD test set (24 samples) |
| `data_profile.json` | Dataset statistics and metadata |

## Reproducing Preprocessing

To regenerate processed files from raw data:

1. Download raw SYMBA files from the link above
2. Place in `data/raw/` directory
3. Run preprocessing notebook:
   ```bash
   jupyter notebook notebooks/notebook1_preprocessing.ipynb
   ```
4. Processed files will be saved to `data/processed/`

## Vocabulary Composition

| Category | Count | Examples |
|---|---|---|
| Special tokens | 4 | `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` |
| Particle masses | 8 | `m_e`, `m_mu`, `m_u`, `m_d`, `m_s`, `m_c`, `m_b`, `m_t` |
| Mandelstam variables | 6 | `s_12`, `s_13`, `s_14`, `s_23`, `s_24`, `s_34` |
| Momenta | 4 | `p_1`, `p_2`, `p_3`, `p_4` |
| Numerical constants | 24 | `1/144`, `1/36`, ..., `16`, `-2`, `-1`, `0`–`9` |
| Physics operators | 3 | `gamma`, `reg_prop`, `i` |
| Coupling constants | 2 | `e` (electromagnetic), `g` (strong) |
| Particle symbols | 13 | `mu`, `tt`, `T`, `G`, `A`, `u`, `v`, `d`, `s`, `t`, `b`, `c` |
| Math operators | 11 | `*`, `+`, `-`, `/`, `^`, `(`, `)`, `{`, `}`, `,`, `_` |
| Normalized indices | 120 | `IDX_0` … `IDX_119` |
| **Total** | **192** | Zero UNK tokens achieved |

## References

- ML4Sci SYMBA Project: https://ml4sci.org/gsoc/projects/2026/project_SYMBA.html
- GSoC 2026 Evaluation Tasks: See `Symbolic AI Tests 2026.md` in root directory
- Preprocessing implementation: `notebooks/notebook1_preprocessing.ipynb`
- Design rationale: See `ARCHITECTURE.md` §1
