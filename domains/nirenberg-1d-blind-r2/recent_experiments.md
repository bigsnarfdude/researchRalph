# Recent Experiments — nirenberg-1d-blind-r2

**Best: 0.0 (exp056)** | Total: 152 | Breakthroughs: 3 | Crashes: 9

### → exp147 — 2.00148302e-16 (= best)
- **Agent:** agent0 | **Design:** problem_formulation | **Status:** discard
- **What:** Test: cosine perturbation K(θ)=K_amplitude·cos(θ), positive branch
- **Outcome:** PLATEAU
- **Redundant with:** exp145

### → exp148 — 0.0 (= best)
- **Agent:** agent1 | **Design:** solver_param | **Status:** discard
- **What:** Final validation: multipole+fourier_modes=2, trivial branch confirmation
- **Outcome:** PLATEAU
- **Redundant with:** exp146

### → exp149 — 2.00148302e-16 (= best)
- **Agent:** agent0 | **Design:** problem_formulation | **Status:** discard
- **What:** Test: sine perturbation on negative branch (u ≈ -1)
- **Outcome:** PLATEAU

### → exp150 — 0.0 (= best)
- **Agent:** agent0 | **Design:** problem_formulation | **Status:** discard
- **What:** Test: sine perturbation on trivial branch (u ≈ 0)
- **Outcome:** PLATEAU
- **Redundant with:** exp149

### → exp151 — 5.51351218e-14 (= best)
- **Agent:** agent2 | **Design:** validation | **Status:** discard
- **What:** fourier_modes=1, positive branch, spectral convergence study
- **Outcome:** PLATEAU

## Score trajectory (all)
| exp | score | outcome |
|-----|-------|---------|
| exp001 | 5.63592552e-11 | ★ BREAKTHROUGH |
| exp002 | 5.63592552e-11 | → PLATEAU |
| exp003 | 5.72745888e-09 | ↓ REGRESSION |
| exp004 | 2.41595655e-09 | ↓ REGRESSION |
| exp005 | 5.63592552e-11 | → PLATEAU |
| exp006 | 5.72745888e-09 | ↓ REGRESSION |
| exp007 | 2.41595655e-09 | ↑ INCREMENTAL |
| exp008 | 8.9529595e-11 | ↑ INCREMENTAL |
| exp009 | 8.9529595e-11 | ↑ INCREMENTAL |
| exp010 | 2.98031263e-13 | ★ BREAKTHROUGH |
| exp011 | 2.98031263e-13 | → PLATEAU |
| exp012 | 2.62635931e-11 | ↑ INCREMENTAL |
| exp013 | 8.9529595e-11 | ↓ REGRESSION |
| exp014 | 8.9529595e-11 | ↓ REGRESSION |
| exp015 | 8.81934685e-11 | ↓ REGRESSION |
| exp016 | 8.81934684e-11 | ↑ INCREMENTAL |
| exp017 | 8.81934684e-11 | ↑ INCREMENTAL |
| exp018 | 8.58922841e-11 | ↑ INCREMENTAL |
| exp018 | 8.81934685e-11 | ↓ REGRESSION |
| exp019 | 2.60019365e-11 | ↑ INCREMENTAL |
| exp020 | 2.62635931e-11 | ↑ INCREMENTAL |
| exp021 | 8.81934684e-11 | ↓ REGRESSION |
| exp022 | 8.77448544e-11 | ↓ REGRESSION |
| exp023 | 2.60019365e-11 | ↑ INCREMENTAL |
| exp024 | 9.9997775e-12 | ↑ INCREMENTAL |
| exp025 | 8.9529595e-11 | ↓ REGRESSION |
| exp026 | 9.9997775e-12 | ↑ INCREMENTAL |
| exp027 | 8.9529595e-11 | ↓ REGRESSION |
| exp028 | CRASH | ✗ CRASH |
| exp029 | 8.9529595e-11 | ↓ REGRESSION |
| exp030 | 9.99840186e-12 | ↑ INCREMENTAL |
| exp031 | 9.9997775e-12 | ↑ INCREMENTAL |
| exp032 | CRASH | ✗ CRASH |
| exp033 | 9.9997775e-12 | ↑ INCREMENTAL |
| exp034 | 7.78244459e-12 | ↑ INCREMENTAL |
| exp035 | 4.50208716e-11 | ↓ REGRESSION |
| exp036 | 9.99840186e-12 | ↑ INCREMENTAL |
| exp037 | 7.78244459e-12 | ↑ INCREMENTAL |
| exp038 | 9.9997775e-12 | ↓ REGRESSION |
| exp039 | CRASH | ✗ CRASH |
| exp040 | 9.99840186e-12 | ↑ INCREMENTAL |
| exp041 | 5.59439569e-12 | ↑ INCREMENTAL |
| exp042 | 9.99840186e-12 | ↑ INCREMENTAL |
| exp043 | 8.75323952e-11 | ↓ REGRESSION |
| exp044 | 5.59439878e-12 | ↑ INCREMENTAL |
| exp045 | 9.99840186e-12 | ↑ INCREMENTAL |
| exp046 | 2.98031263e-13 | → PLATEAU |
| exp047 | 9.99840186e-12 | ↑ INCREMENTAL |
| exp048 | 9.9997775e-12 | ↓ REGRESSION |
| exp049 | 9.99840186e-12 | ↑ INCREMENTAL |
| exp050 | 9.99840186e-12 | ↑ INCREMENTAL |
| exp051 | 4.88774326e-12 | ↑ INCREMENTAL |
| exp052 | 9.99840186e-12 | ↑ INCREMENTAL |
| exp053 | 5.33365326e-12 | ↑ INCREMENTAL |
| exp054 | 1.89688828e-12 | ↑ INCREMENTAL |
| exp055 | 1.74959781e-12 | ↑ INCREMENTAL |
| exp056 | 0.0 | ★ BREAKTHROUGH |
| exp057 | 9.9997775e-12 | → PLATEAU |
| exp058 | 9.9997775e-12 | → PLATEAU |
| exp059 | 1.74959781e-12 | → PLATEAU |
| exp060 | 1.74959781e-12 | → PLATEAU |
| exp061 | 1.74959781e-12 | → PLATEAU |
| exp062 | 1.74959781e-12 | → PLATEAU |
| exp063 | 1.74959781e-12 | → PLATEAU |
| exp064 | 1.74959781e-12 | → PLATEAU |
| exp065 | 9.9997775e-12 | → PLATEAU |
| exp066 | 1.74959781e-12 | → PLATEAU |
| exp067 | 1.74959781e-12 | → PLATEAU |
| exp068 | 1.74959781e-12 | → PLATEAU |
| exp069 | 1.74959781e-12 | → PLATEAU |
| exp070 | 0.0 | → PLATEAU |
| exp071 | 1.74959781e-12 | → PLATEAU |
| exp072 | 8.10450608e-12 | → PLATEAU |
| exp073 | 1.74959781e-12 | → PLATEAU |
| exp074 | 0.0 | → PLATEAU |
| exp075 | CRASH | ✗ CRASH |
| exp076 | 3.31697173e-12 | → PLATEAU |
| exp077 | 1.74959781e-12 | → PLATEAU |
| exp078 | 5.62802405e-12 | → PLATEAU |
| exp079 | 3.25175408e-12 | → PLATEAU |
| exp080 | 1.74959781e-12 | → PLATEAU |
| exp081 | CRASH | ✗ CRASH |
| exp082 | 2.66626099e-13 | → PLATEAU |
| exp083 | 4.50208716e-11 | → PLATEAU |
| exp084 | CRASH | ✗ CRASH |
| exp085 | 2.66626099e-13 | → PLATEAU |
| exp086 | 5.62802405e-12 | → PLATEAU |
| exp087 | 2.35974309e-13 | → PLATEAU |
| exp088 | CRASH | ✗ CRASH |
| exp089 | 1.61211386e-12 | → PLATEAU |
| exp090 | 9.9997775e-12 | → PLATEAU |
| exp091 | 1.74959781e-12 | → PLATEAU |
| exp092 | 2.35974309e-13 | → PLATEAU |
| exp093 | 1.74959781e-12 | → PLATEAU |
| exp094 | 7.78244459e-12 | → PLATEAU |
| exp095 | 2.66626099e-13 | → PLATEAU |
| exp096 | 1.74959781e-12 | → PLATEAU |
| exp097 | 2.35974309e-13 | → PLATEAU |
| exp098 | CRASH | ✗ CRASH |
| exp099 | 2.66626099e-13 | → PLATEAU |
| exp100 | 0.0 | → PLATEAU |
| exp101 | CRASH | ✗ CRASH |
| exp102 | 6.5300165e-13 | → PLATEAU |
| exp103 | 2.35974309e-13 | → PLATEAU |
| exp104 | 0.0 | → PLATEAU |
| exp105 | 2.35974309e-13 | → PLATEAU |
| exp106 | 6.03463222e-14 | → PLATEAU |
| exp107 | 1.74959781e-12 | → PLATEAU |
| exp108 | 6.03463222e-14 | → PLATEAU |
| exp109 | 1.74959781e-12 | → PLATEAU |
| exp110 | 5.03136422e-14 | → PLATEAU |
| exp111 | 5.03136422e-14 | → PLATEAU |
| exp112 | 0.0 | → PLATEAU |
| exp113 | 2.35974309e-13 | → PLATEAU |
| exp114 | 2.35974309e-13 | → PLATEAU |
| exp115 | 1.7974265e-13 | → PLATEAU |
| exp116 | 2.66626099e-13 | → PLATEAU |
| exp117 | 0.0 | → PLATEAU |
| exp118 | 2.10749927e-13 | → PLATEAU |
| exp119 | 1.89221377e-13 | → PLATEAU |
| exp120 | 2.66626099e-13 | → PLATEAU |
| exp121 | 2.66626099e-13 | → PLATEAU |
| exp122 | 1.7974265e-13 | → PLATEAU |
| exp123 | 2.66626099e-13 | → PLATEAU |
| exp124 | 1.33401968e-14 | → PLATEAU |
| exp125 | 4.61400212e-14 | → PLATEAU |
| exp126 | 2.30031757e-15 | → PLATEAU |
| exp127 | 6.41117431e-16 | → PLATEAU |
| exp128 | 3.61888431e-16 | → PLATEAU |
| exp129 | 0.0 | → PLATEAU |
| exp130 | 5.03136422e-14 | → PLATEAU |
| exp131 | 0.0 | → PLATEAU |
| exp132 | 3.99522993e-13 | → PLATEAU |
| exp133 | 3.15118549e-13 | → PLATEAU |
| exp134 | 2.8572622e-12 | → PLATEAU |
| exp135 | 3.61888431e-16 | → PLATEAU |
| exp136 | 0.0 | → PLATEAU |
| exp137 | 5.51351218e-14 | → PLATEAU |
| exp138 | 3.61888431e-16 | → PLATEAU |
| exp139 | 3.61888431e-16 | → PLATEAU |
| exp140 | 3.61888431e-16 | → PLATEAU |
| exp141 | 3.61888431e-16 | → PLATEAU |
| exp142 | 0.0 | → PLATEAU |
| exp143 | 3.61888431e-16 | → PLATEAU |
| exp144 | 3.61888431e-16 | → PLATEAU |
| exp145 | 2.00148302e-16 | → PLATEAU |
| exp146 | 3.61888431e-16 | → PLATEAU |
| exp147 | 2.00148302e-16 | → PLATEAU |
| exp148 | 0.0 | → PLATEAU |
| exp149 | 2.00148302e-16 | → PLATEAU |
| exp150 | 0.0 | → PLATEAU |
| exp151 | 5.51351218e-14 | → PLATEAU |
