# Recent Experiments — nirenberg-1d-chaos-v3-haiku-r3

**Best: 0.0 (exp063)** | Total: 92 | Breakthroughs: 5 | Crashes: 7

### → exp088 — 3.25175165e-12 (= best)
- **Agent:** agent0 | **Design:** branch_search | **Status:** discard
- **What:** crash zone u_offset=-0.60 with stable solver
- **Outcome:** PLATEAU

### → exp089 — 3.25175165e-12 (= best)
- **Agent:** agent2 | **Design:** branch_search | **Status:** discard
- **What:** test crash zone u_offset=-0.60 with ultra-stable solver: n_nodes=300, tol=1e-11
- **Outcome:** PLATEAU

### ✗ exp090 — CRASH
- **Agent:** agent2 | **Design:** branch_search | **Status:** crash
- **What:** boundary test: u_offset=-0.59, n_nodes=300, tol=1e-11
- **Outcome:** CRASH
- **Redundant with:** exp073

### → exp091 — 3.25175408e-12 (= best)
- **Agent:** agent0 | **Design:** branch_search | **Status:** discard
- **What:** u_offset=-0.62 with ultra-refined solver
- **Outcome:** PLATEAU

### → exp092 — 7.70453341e-12 (= best)
- **Agent:** agent2 | **Design:** branch_search | **Status:** discard
- **What:** boundary test: u_offset=-0.595, n_nodes=300, tol=1e-11
- **Outcome:** PLATEAU
- **Redundant with:** exp073

## Score trajectory (all)
| exp | score | outcome |
|-----|-------|---------|
| exp001 | 5.63598005e-11 | ★ BREAKTHROUGH |
| exp002 | 5.63598005e-11 | → PLATEAU |
| exp003 | 5.72745881e-09 | ↓ REGRESSION |
| exp004 | 5.63598005e-11 | → PLATEAU |
| exp005 | 5.72745859e-09 | ↓ REGRESSION |
| exp006 | 5.72745788e-09 | ↓ REGRESSION |
| exp007 | 5.72745888e-09 | ↓ REGRESSION |
| exp008 | 2.41595655e-09 | ↑ INCREMENTAL |
| exp009 | 5.72745888e-09 | ↓ REGRESSION |
| exp010 | 2.41595655e-09 | ↑ INCREMENTAL |
| exp011 | 2.41595655e-09 | ↑ INCREMENTAL |
| exp012 | 8.81934684e-11 | ↑ INCREMENTAL |
| exp013 | 1.30911985e-11 | ★ BREAKTHROUGH |
| exp014 | 5.72745864e-09 | ↓ REGRESSION |
| exp015 | 5.63598005e-11 | ↑ INCREMENTAL |
| exp016 | 8.81934684e-11 | ↑ INCREMENTAL |
| exp017 | 1.95368268e-12 | ★ BREAKTHROUGH |
| exp018 | 5.72745869e-09 | ↓ REGRESSION |
| exp019 | 3.37219364e-11 | ↑ INCREMENTAL |
| exp020 | 2.41595655e-09 | ↓ REGRESSION |
| exp021 | 6.65041888e-10 | ↓ REGRESSION |
| exp022 | 3.75113468e-11 | ↑ INCREMENTAL |
| exp023 | 5.69266279e-19 | ★ BREAKTHROUGH |
| exp024 | 1.30911985e-11 | ↑ INCREMENTAL |
| exp025 | 2.41595655e-09 | ↓ REGRESSION |
| exp026 | 5.72745855e-09 | ↓ REGRESSION |
| exp027 | 5.72748792e-09 | ↓ REGRESSION |
| exp028 | 5.72745864e-09 | ↓ REGRESSION |
| exp029 | 3.75113468e-11 | ↑ INCREMENTAL |
| exp030 | 2.41595655e-09 | ↑ INCREMENTAL |
| exp031 | 5.69266279e-19 | → PLATEAU |
| exp032 | 2.41595655e-09 | ↑ INCREMENTAL |
| exp033 | 5.73029223e-09 | ↓ REGRESSION |
| exp034 | 5.73029223e-09 | ↓ REGRESSION |
| exp035 | 5.72826282e-09 | ↓ REGRESSION |
| exp036 | 2.82460476e-11 | ↑ INCREMENTAL |
| exp037 | 5.72745864e-09 | ↑ INCREMENTAL |
| exp038 | 1.07754871e-11 | ↑ INCREMENTAL |
| exp039 | 9.43861279e-12 | ↑ INCREMENTAL |
| exp040 | 1.07198893e-11 | ↑ INCREMENTAL |
| exp041 | 5.17243862e-11 | ↓ REGRESSION |
| exp042 | 1.07754871e-11 | ↑ INCREMENTAL |
| exp043 | 2.41595655e-09 | ↓ REGRESSION |
| exp044 | 2.41595655e-09 | ↓ REGRESSION |
| exp045 | 5.72826282e-09 | ↓ REGRESSION |
| exp046 | 5.72745874e-09 | ↓ REGRESSION |
| exp047 | 8.81934685e-11 | ↑ INCREMENTAL |
| exp048 | 5.57178986e-09 | ↓ REGRESSION |
| exp049 | 8.81934684e-11 | ↑ INCREMENTAL |
| exp050 | CRASH | ✗ CRASH |
| exp051 | 8.9529595e-11 | ↑ INCREMENTAL |
| exp052 | 2.98482733e-13 | ↑ INCREMENTAL |
| exp053 | 8.9529595e-11 | ↓ REGRESSION |
| exp054 | 8.81934684e-11 | ↑ INCREMENTAL |
| exp055 | 2.60019368e-11 | ↑ INCREMENTAL |
| exp056 | 8.77448544e-11 | ↑ INCREMENTAL |
| exp057 | 7.70453892e-12 | ↑ INCREMENTAL |
| exp058 | 2.60019365e-11 | ↑ INCREMENTAL |
| exp059 | CRASH | ✗ CRASH |
| exp060 | 8.81934685e-11 | ↓ REGRESSION |
| exp061 | 7.70453892e-12 | ↑ INCREMENTAL |
| exp062 | 2.61339314e-11 | ↓ REGRESSION |
| exp063 | 0.0 | ★ BREAKTHROUGH |
| exp064 | 3.25175408e-12 | → PLATEAU |
| exp065 | 7.16167802e-10 | → PLATEAU |
| exp066 | 8.81934685e-11 | → PLATEAU |
| exp067 | 7.70453892e-12 | → PLATEAU |
| exp068 | 8.81934684e-11 | → PLATEAU |
| exp069 | 3.31696748e-12 | → PLATEAU |
| exp070 | 3.31696748e-12 | → PLATEAU |
| exp071 | 2.61333321e-11 | → PLATEAU |
| exp072 | 8.81934685e-11 | → PLATEAU |
| exp073 | CRASH | ✗ CRASH |
| exp074 | 8.77448034e-11 | → PLATEAU |
| exp075 | 4.84499392e-12 | → PLATEAU |
| exp076 | CRASH | ✗ CRASH |
| exp077 | 2.04834364e-12 | → PLATEAU |
| exp078 | CRASH | ✗ CRASH |
| exp079 | 2.98157327e-13 | → PLATEAU |
| exp080 | 2.04834364e-12 | → PLATEAU |
| exp081 | CRASH | ✗ CRASH |
| exp082 | 0.0 | → PLATEAU |
| exp083 | 1.96612266e-12 | → PLATEAU |
| exp084 | 7.25322922e-14 | → PLATEAU |
| exp085 | 2.04834364e-12 | → PLATEAU |
| exp086 | 8.81934685e-11 | → PLATEAU |
| exp087 | 3.25175408e-12 | → PLATEAU |
| exp088 | 3.25175165e-12 | → PLATEAU |
| exp089 | 3.25175165e-12 | → PLATEAU |
| exp090 | CRASH | ✗ CRASH |
| exp091 | 3.25175408e-12 | → PLATEAU |
| exp092 | 7.70453341e-12 | → PLATEAU |
