# Recent Experiments — gpt2-tinystories-v44

**Best: 1.083339 (exp099)** | Total: 100 | Breakthroughs: 22 | Crashes: 6

### → exp096 — 1.083475 (Δ +0.0001 from best)
- **Agent:** agent0 | **Design:** architecture | **Status:** discard
- **What:** softcap=11 (from 12) at depth=7+wt+window128 — fine-grain between 10 and 12, both good
- **Outcome:** PLATEAU

### → exp097 — 1.084591 (Δ +0.0013 from best)
- **Agent:** agent1 | **Design:** optimizer | **Status:** discard
- **What:** resid_lambda LR multiplier=0.1 (from 0.01) at depth=7+wt+softcap12+window128 — 10x faster per-layer residual scaling, completely untested
- **Outcome:** PLATEAU

### → exp098 — 1.086957 (Δ +0.0036 from best)
- **Agent:** agent1 | **Design:** hyperparam | **Status:** discard
- **What:** position-weighted loss (0.5+0.5*t/T) at depth=7+wt+softcap12+window128 — focus on later tokens, never tested
- **Outcome:** PLATEAU

### ★ exp099 — 1.083339 (= best)
- **Agent:** agent1 | **Design:** optimizer | **Status:** keep
- **What:** Muon momentum target=0.93 (from 0.95) at depth=7+wt+softcap12+window128 — lower momentum for sharper convergence
- **Outcome:** BREAKTHROUGH

### → exp100 — 1.083962 (Δ +0.0006 from best)
- **Agent:** agent1 | **Design:** optimizer | **Status:** discard
- **What:** Muon momentum target=0.91 (from 0.93) — continue momentum bracket, 0.93 was new best
- **Outcome:** PLATEAU

## Score trajectory (all)
| exp | score | outcome |
|-----|-------|---------|
| exp001 | 1.171181 | ★ BREAKTHROUGH |
| exp002 | 1.171318 | → PLATEAU |
| exp003 | 1.106267 | ★ BREAKTHROUGH |
| exp004 | 1.101996 | ★ BREAKTHROUGH |
| exp005 | 1.107461 | → PLATEAU |
| exp006 | 1.112306 | → PLATEAU |
| exp007 | 1.107387 | → PLATEAU |
| exp008 | 1.11304 | ↓ REGRESSION |
| exp009 | 1.105093 | → PLATEAU |
| exp010 | 1.113548 | ↓ REGRESSION |
| exp011 | 1.108498 | → PLATEAU |
| exp012 | 1.124755 | ↓ REGRESSION |
| exp013 | 1.102547 | → PLATEAU |
| exp014 | 1.112268 | → PLATEAU |
| exp015 | 1.101255 | ★ BREAKTHROUGH |
| exp016 | 1.110463 | → PLATEAU |
| exp017 | 1.103183 | → PLATEAU |
| exp018 | 1.103549 | → PLATEAU |
| exp019 | CRASH | ✗ CRASH |
| exp020 | CRASH | ✗ CRASH |
| exp021 | 1.114496 | ↓ REGRESSION |
| exp022 | 1.214665 | ↓ REGRESSION |
| exp023 | CRASH | ✗ CRASH |
| exp024 | 1.10921 | → PLATEAU |
| exp025 | 1.111218 | → PLATEAU |
| exp026 | 1.101116 | ★ BREAKTHROUGH |
| exp027 | 1.100641 | ★ BREAKTHROUGH |
| exp028 | 1.09975 | ★ BREAKTHROUGH |
| exp029 | 1.0994 | ★ BREAKTHROUGH |
| exp030 | 1.099741 | → PLATEAU |
| exp031 | 1.099829 | → PLATEAU |
| exp032 | 1.108755 | → PLATEAU |
| exp033 | 1.103438 | → PLATEAU |
| exp034 | 1.102206 | → PLATEAU |
| exp035 | 1.424567 | ↓ REGRESSION |
| exp036 | 1.102057 | → PLATEAU |
| exp037 | 1.096144 | ★ BREAKTHROUGH |
| exp038 | 1.095829 | ★ BREAKTHROUGH |
| exp039 | 1.106415 | → PLATEAU |
| exp040 | 1.10576 | → PLATEAU |
| exp041 | 1.096594 | → PLATEAU |
| exp042 | 1.098621 | → PLATEAU |
| exp043 | 1.096293 | → PLATEAU |
| exp044 | 1.101672 | → PLATEAU |
| exp045 | 1.098148 | → PLATEAU |
| exp046 | 1.090281 | ★ BREAKTHROUGH |
| exp047 | CRASH | ✗ CRASH |
| exp048 | CRASH | ✗ CRASH |
| exp049 | 1.089782 | ★ BREAKTHROUGH |
| exp050 | 1.089305 | ★ BREAKTHROUGH |
| exp051 | 1.090932 | → PLATEAU |
| exp052 | 1.090674 | → PLATEAU |
| exp053 | 1.089072 | ★ BREAKTHROUGH |
| exp054 | 1.09426 | → PLATEAU |
| exp055 | 1.08891 | ★ BREAKTHROUGH |
| exp056 | CRASH | ✗ CRASH |
| exp057 | 1.145403 | ↓ REGRESSION |
| exp058 | 1.144892 | ↓ REGRESSION |
| exp059 | 1.090145 | → PLATEAU |
| exp060 | 1.091679 | → PLATEAU |
| exp061 | 1.093252 | → PLATEAU |
| exp062 | 1.100354 | ↓ REGRESSION |
| exp063 | 1.089107 | → PLATEAU |
| exp064 | 1.089878 | → PLATEAU |
| exp065 | 1.096017 | → PLATEAU |
| exp066 | 1.090943 | → PLATEAU |
| exp067 | 1.085047 | ★ BREAKTHROUGH |
| exp068 | 1.105318 | ↓ REGRESSION |
| exp069 | 1.083949 | ★ BREAKTHROUGH |
| exp070 | 1.083842 | ★ BREAKTHROUGH |
| exp071 | 1.083695 | ★ BREAKTHROUGH |
| exp072 | 1.084054 | → PLATEAU |
| exp073 | 1.084182 | → PLATEAU |
| exp074 | 1.085638 | → PLATEAU |
| exp075 | 1.088946 | → PLATEAU |
| exp076 | 1.088087 | → PLATEAU |
| exp077 | 1.087547 | → PLATEAU |
| exp078 | 1.08577 | → PLATEAU |
| exp079 | 1.130037 | ↓ REGRESSION |
| exp080 | 1.087187 | → PLATEAU |
| exp081 | 1.08709 | → PLATEAU |
| exp082 | 1.083849 | → PLATEAU |
| exp083 | 1.086134 | → PLATEAU |
| exp084 | 1.084483 | → PLATEAU |
| exp085 | 1.084022 | → PLATEAU |
| exp086 | 1.533772 | ↓ REGRESSION |
| exp087 | 1.533855 | ↓ REGRESSION |
| exp088 | 1.085151 | → PLATEAU |
| exp089 | 1.08355 | ★ BREAKTHROUGH |
| exp090 | 1.083462 | ★ BREAKTHROUGH |
| exp091 | 1.085494 | → PLATEAU |
| exp092 | 1.086321 | → PLATEAU |
| exp093 | 1.083791 | → PLATEAU |
| exp094 | 1.084164 | → PLATEAU |
| exp095 | 1.085206 | → PLATEAU |
| exp096 | 1.083475 | → PLATEAU |
| exp097 | 1.084591 | → PLATEAU |
| exp098 | 1.086957 | → PLATEAU |
| exp099 | 1.083339 | ★ BREAKTHROUGH |
| exp100 | 1.083962 | → PLATEAU |
