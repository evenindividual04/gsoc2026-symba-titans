"""
Dataset analysis script for SYMBA evaluation task.
Requires: tokenizer.pkl already exists in ../data/processed/
"""
import re, pickle, sys
from pathlib import Path
from collections import Counter, OrderedDict
import numpy as np

DATA_ROOT = Path('../SYMBA - Test Data/common-task-1.2')
SAVE_DIR  = Path('../data/processed')

# ── Inline tokenizer class so we can unpickle ──────────────────────────────
class PhysicsTokenizer:
    SPECIAL   = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    MASSES    = ['m_mu', 'm_e', 'm_u', 'm_d', 'm_s', 'm_t', 'm_b', 'm_c']
    MAND      = ['s_12', 's_13', 's_14', 's_23', 's_24', 's_34']
    MOMENTA   = ['p_1', 'p_2', 'p_3', 'p_4']
    NUMBERS   = ['1/144','1/36','1/16','1/12','1/9','1/6',
                 '1/4','1/3','1/2','16','8','4','3','2','1',
                 '-2','-1','0','5','6','7','9']
    PHYSICS   = ['reg_prop', 'gamma', 'i']
    COUPLING  = ['e', 'g']
    PARTICLES = ['mu','tt','T','G','A','u','v','d','s','t','b','c']
    OPERATORS = ['*','+','-','/','^','(',')' ,'{','}',',','_']

    def __init__(self, max_idx=120):
        self.max_idx = max_idx
        idx_tokens = [f'IDX_{n}' for n in range(max_idx)]
        all_tokens = (self.SPECIAL + self.MASSES + self.MAND + self.MOMENTA +
                      self.NUMBERS + self.PHYSICS + self.COUPLING + self.PARTICLES +
                      self.OPERATORS + idx_tokens)
        seen, self.vocab = set(), []
        for t in all_tokens:
            if t not in seen: self.vocab.append(t); seen.add(t)
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.PAD_ID = self.token2id['<PAD>']
        self.SOS_ID = self.token2id['<SOS>']
        self.EOS_ID = self.token2id['<EOS>']
        self.UNK_ID = self.token2id['<UNK>']
        sortable = (self.MASSES + self.MAND + self.MOMENTA + self.NUMBERS +
                    self.PHYSICS + self.PARTICLES + self.COUPLING +
                    idx_tokens + self.OPERATORS)
        sorted_toks = sorted(set(sortable), key=lambda t: (-len(t), t))
        self._tok_re = re.compile('|'.join(re.escape(t) for t in sorted_toks))

    def tokenize(self, expr):
        tokens, pos = [], 0
        expr = expr.strip()
        while pos < len(expr):
            if expr[pos].isspace(): pos += 1; continue
            m = self._tok_re.match(expr, pos)
            if m: tokens.append(m.group(0)); pos = m.end()
            else: tokens.append('<UNK>'); pos += 1
        return tokens

    def encode(self, expr, add_special=True):
        ids = [self.token2id.get(t, self.UNK_ID) for t in self.tokenize(expr)]
        return ([self.SOS_ID] + ids + [self.EOS_ID]) if add_special else ids

    @property
    def vocab_size(self): return len(self.vocab)


# ── Load data ──────────────────────────────────────────────────────────────
ARB_PATTERN = re.compile(r'(%(?:\\[a-zA-Z]+|[a-zA-Z]+)|[klijGHABCDEF])_(\d+)')

def normalize_indices(expr):
    idx_map, ctr = OrderedDict(), [0]
    def rep(m):
        k = m.group(0)
        if k not in idx_map: idx_map[k] = f'IDX_{ctr[0]}'; ctr[0] += 1
        return idx_map[k]
    return ARB_PATTERN.sub(rep, expr)

samples = []
for model in ('qed', 'qcd'):
    for fpath in sorted((DATA_ROOT / model).glob('*.txt')):
        level = int(fpath.stem.split('-')[-1])
        with open(fpath) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split(' : ')
            if len(parts) == 4:
                samples.append({
                    'physics_model': model.upper(), 'source_file': fpath.name,
                    'tree_level': level, 'event_type': parts[0],
                    'feynman_diagram': parts[1], 'amplitude': parts[2],
                    'squared_amplitude': parts[3],
                })

for s in samples:
    s['amplitude_norm'] = normalize_indices(s['amplitude'])

tokenizer = PhysicsTokenizer(max_idx=120)
qed = [s for s in samples if s['physics_model'] == 'QED']
qcd = [s for s in samples if s['physics_model'] == 'QCD']

def tok_len(expr): return len(tokenizer.tokenize(expr))

W = 68
print("=" * W)
print("DATASET ANALYSIS REPORT — SYMBA Evaluation Task")
print("=" * W)

# 1. COUNTS
print("\n── 1. SAMPLE COUNTS ─────────────────────────────────────────")
print(f"Total: {len(samples)}   QED: {len(qed)}   QCD: {len(qcd)}")
for model, slist in [('QED', qed), ('QCD', qcd)]:
    by_lv = Counter(s['tree_level'] for s in slist)
    print(f"  {model} per level: {dict(sorted(by_lv.items()))}")

# 2. SEQUENCE LENGTH
print("\n── 2. TOKEN LENGTHS (input / output) ────────────────────────")
for model, slist in [('QED', qed), ('QCD', qcd)]:
    a = [tok_len(s['amplitude_norm']) for s in slist]
    q = [tok_len(s['squared_amplitude']) for s in slist]
    for label, lens in [('amp_norm', a), ('sq_amp  ', q)]:
        p = lambda v: int(np.percentile(lens, v))
        print(f"  {model} {label}: min={min(lens):4d} p25={p(25):4d} med={p(50):4d} "
              f"p75={p(75):4d} p95={p(95):4d} max={max(lens):4d} mean={np.mean(lens):.0f}")

# 3. LENGTH BY TREE LEVEL
print("\n── 3. AMPLITUDE LENGTH BY TREE LEVEL ────────────────────────")
for model, slist in [('QED', qed), ('QCD', qcd)]:
    print(f"  {model}:")
    by_lv = {}
    for s in slist:
        by_lv.setdefault(s['tree_level'], []).append(tok_len(s['amplitude_norm']))
    for lv in sorted(by_lv):
        lens = by_lv[lv]
        print(f"    L{lv}: n={len(lens):2d}  min={min(lens):4d} max={max(lens):4d} mean={np.mean(lens):6.1f}")

# 4. UNIQUE ARBITRARY INDICES
print("\n── 4. ARBITRARY INDEX COUNTS (amplitude inputs) ─────────────")
for model, slist in [('QED', qed), ('QCD', qcd)]:
    unique_idx = []
    max_idx_num = []
    for s in slist:
        matches = list(ARB_PATTERN.finditer(s['amplitude']))
        unique_tokens = set(m.group(0) for m in matches)
        unique_idx.append(len(unique_tokens))
        nums = [int(m.group(2)) for m in matches]
        max_idx_num.append(max(nums) if nums else 0)
    print(f"  {model} unique arb indices per sample: "
          f"min={min(unique_idx)} med={int(np.median(unique_idx))} max={max(unique_idx)}")
    print(f"  {model} max raw index number:          "
          f"min={min(max_idx_num)} med={int(np.median(max_idx_num))} max={max(max_idx_num)}")
    # After normalization these become IDX_0..IDX_N-1
    print(f"  => after normalization: IDX_0 to IDX_{max(unique_idx)-1} "
          f"(max {max(unique_idx)} unique IDX tokens per sample)")

# 5. SQUARED AMPLITUDE STRUCTURE
print("\n── 5. SQUARED AMPLITUDE STRUCTURE (output) ──────────────────")
for model, slist in [('QED', qed), ('QCD', qcd)]:
    # Count additive terms (+ at top level, not inside exponents)
    term_counts, frac_counts = [], []
    mand_sets = []
    for s in slist:
        sa = s['squared_amplitude']
        # terms ~ count of top-level '+'
        terms = sa.count('+') + 1
        term_counts.append(terms)
        fracs = len(re.findall(r'reg_prop', sa))
        frac_counts.append(fracs)
        mand_sets.append(set(re.findall(r's_\d\d', sa)))
    all_mand = sorted(set().union(*mand_sets))
    print(f"  {model} additive terms:  min={min(term_counts)} max={max(term_counts)} mean={np.mean(term_counts):.1f}")
    print(f"  {model} reg_prop count:  min={min(frac_counts)} max={max(frac_counts)} mean={np.mean(frac_counts):.1f}")
    print(f"  {model} Mandelstam vars seen: {all_mand}")

# 6. COUPLING CONSTANT POWERS
print("\n── 6. COUPLING CONSTANT POWERS IN OUTPUTS ───────────────────")
for model, slist in [('QED', qed), ('QCD', qcd)]:
    for coup in ('e', 'g'):
        powers = set()
        for s in slist:
            powers |= set(int(x) for x in re.findall(rf'{coup}\^(\d+)', s['squared_amplitude']))
        if powers:
            print(f"  {model} {coup}^N: N ∈ {sorted(powers)}")

# 7. EVENT TYPES
print("\n── 7. EVENT TYPES ────────────────────────────────────────────")
ev = Counter(s['event_type'] for s in samples)
for k, v in sorted(ev.items()):
    print(f"  {k}: {v}")

# 8. AMPLITUDE TERM COUNT (top-level * terms)
print("\n── 8. AMPLITUDE COMPLEXITY (top-level × factors, normalized) ─")
for model, slist in [('QED', qed), ('QCD', qcd)]:
    muls = [s['amplitude_norm'].count('*') for s in slist]
    print(f"  {model} num '*' in amplitude_norm: min={min(muls)} max={max(muls)} mean={np.mean(muls):.1f}")

# 9. HARDEST SAMPLES
print("\n── 9. HARDEST SAMPLES (longest sq_amplitude) ────────────────")
for model, slist in [('QED', qed), ('QCD', qcd)]:
    worst = sorted(slist, key=lambda s: tok_len(s['squared_amplitude']), reverse=True)[:3]
    print(f"\n  {model} top-3 hardest (by sq_amp length):")
    for s in worst:
        print(f"    L{s['tree_level']}  sq={tok_len(s['squared_amplitude']):4d} tok  "
              f"amp={tok_len(s['amplitude_norm']):4d} tok  file={s['source_file']}")
        print(f"      sq[:120]: {s['squared_amplitude'][:120]}")

# 10. VOCABULARY COVERAGE
print("\n── 10. IDX TOKEN USAGE ACROSS WHOLE DATASET ─────────────────")
all_norm = ' '.join(s['amplitude_norm'] for s in samples)
idx_used = set(re.findall(r'IDX_(\d+)', all_norm))
print(f"  IDX_N used: N = 0..{max(int(x) for x in idx_used)}")
print(f"  (tokenizer covers IDX_0..IDX_119 — margin: {119 - max(int(x) for x in idx_used)})")

# 11. RATIO: output/input length
print("\n── 11. OUTPUT / INPUT LENGTH RATIO ──────────────────────────")
for model, slist in [('QED', qed), ('QCD', qcd)]:
    ratios = [tok_len(s['squared_amplitude']) / tok_len(s['amplitude_norm']) for s in slist]
    print(f"  {model}: min={min(ratios):.2f} max={max(ratios):.2f} mean={np.mean(ratios):.2f} median={np.median(ratios):.2f}")

print("\n" + "=" * W)
