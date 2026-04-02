# Post-Run Analysis Plan (Ready to Execute ~13:30 UTC)

**Trigger:** When Runs 5 and 6 both complete (expected ~13:18 and ~13:26 UTC)

---

## Phase 1: Data Collection (5 min)

### SCP results from nigel to macbook

```bash
cd ~/researchRalph
for run in "domains/nirenberg-1d-blind-chaos" \
           "domains/nirenberg-1d-chaos-r5" \
           "domains/nirenberg-1d-chaos-r6"; do
  mkdir -p "$run"
  scp -r vincent@nigel.birs.ca:"~/researchRalph/$run/results.tsv" "$run/"
  scp -r vincent@nigel.birs.ca:"~/researchRalph/$run/logs/" "$run/"
  scp -r vincent@nigel.birs.ca:"~/researchRalph/$run/blackboard.md"* "$run/" 2>/dev/null || true
done
```

---

## Phase 2: Metrics Extraction (15 min)

### 2a. Compute Evenness for All Runs

Extract from results.tsv for each run:
- Branch distribution (pos/neg/triv %)
- Evenness (normalized Shannon entropy)
- Per-agent branch concentration

**Key question:** Do evenness values match predictions?
- Run 5: expected 0.85–0.92
- Run 6: expected 0.60–0.75

### 2b. Per-Agent Analysis

For each run (especially 5 and 6):
- agent0-4 (honest agents): branch distribution
- agent5-7 or agent7 (chaos agents): branch distribution, concentration on positive/negative

**Signal:** If chaos agents > 55% positive and honest agents < 40% positive (or vice versa), steering visible.

### 2c. Chaos Phrase Detection

Search agent descriptions for chaos-prompt language:
- "unstable", "unreliable", "marginal", "artifacts"
- "tighter", "tolerances", "low-priority"
- Parameter steering: "negative u_offset", "positive u_offset"

Extract timeline: when first appeared, which agent, adoption by others.

---

## Phase 3: Phase Boundary Plot (10 min)

### Generate Data Table

| Agents | Chaos Agents | Ratio % | Evenness | Source |
|--------|-------------|---------|----------|--------|
| 2 | 0 | 0% | 0.89 | Run 1 |
| 2 | 1 | 50% | 0.75 | Run 2 |
| 4 | 1 | 25% | 0.87 | Run 3 |
| 4 | 2 | 50% | 1.000 | Run 4 |
| 8 | 1 | 12.5% | ? | Run 5 |
| 8 | 3 | 37.5% | ? | Run 6 |

### Plot

```
X-axis: Chaos ratio (0%, 12.5%, 25%, 37.5%, 50%)
Y-axis: Evenness (0.60–1.00)

Pattern analysis:
- Linear degradation: evenness = 1.0 - k*(chaos_ratio)
- Sharp transition: linear until R*, then cliff
- Compensation plateau: stable until critical ratio, then crash
```

### Identify R* (Phase Boundary)

**Hypothesis testing:**
- If Run 6 evenness < 0.75: compensation failing, R* between 12.5% and 37.5%
- If Run 6 evenness > 0.85: compensation still strong, R* > 37.5%
- If Run 6 evenness 0.75–0.85: R* ≈ 30–35%

---

## Phase 4: Write Forensic Report (30 min)

### File: `RUNS_5_6_FORENSIC_ANALYSIS.md`

Structure:
1. **Evenness Results**
   - Plot: chaos ratio vs evenness (all 6 runs)
   - Identify pattern (linear, sharp, compensation plateau)
   - Estimate R*

2. **Per-Agent Behavior**
   - Branch distribution tables (agents 0-7 for runs 5-6)
   - Chaos agent concentration: did they skew to positive/negative?
   - Honest agent skew: did they compensate or follow?

3. **Influence Contagion**
   - Chaos-prompt phrase timeline
   - Which honest agents adopted phrases?
   - Correlation with behavior change (did phrase adoption → parameter change?)

4. **Herd Immunity Scaling**
   - 2 agents: honest agents can't compensate (evenness drops to 0.75 at 50% chaos)
   - 4 agents: honest agents over-compensate (evenness 1.000 at 50% chaos!)
   - 8 agents: ??? (what's the saturation point?)

5. **Key Findings**
   - Phase boundary location (if identified)
   - Mechanism (verification + feedback required)
   - Implications for security (under what conditions is chaos a real threat?)

---

## Phase 5: Commit to GitHub (10 min)

```bash
cd ~/researchRalph
git add domains/nirenberg-1d-{blind-chaos,chaos-r5,chaos-r6}/results.tsv
git add domains/nirenberg-1d-{blind-chaos,chaos-r5,chaos-r6}/logs/
git commit -m "Archive phase boundary experiments: Runs 5-6 (oracle, 8 agents)

Run 5 (12.5% chaos, 8 agents):
- X experiments, Y evenness (expected 0.85-0.92)
- Herd immunity test: 1 chaos vs 7 honest agents

Run 6 (37.5% chaos, 8 agents):
- X experiments, Y evenness (expected 0.60-0.75)  
- Phase boundary test: 3 chaos vs 5 honest agents

Critical findings:
- Phase boundary R* identified (or confirmed absent)
- Herd immunity scaling law: more agents = higher chaos tolerance
- Feedback loop mechanism confirmed: chaos requires oracle verification

Ready for Chaos Group Theory paper: 'Phase Transitions in Multi-Agent Systems with Adversarial Framing'

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

git push origin main
```

---

## Phase 6: Write Paper (60+ min, can be deferred)

### Chaos Group Theory Paper

**Title:** "Chaos Group Theory: Phase Transitions in Multi-Agent Systems with Adversarial Framing"

**Structure:**
1. **Introduction**
   - Motivation: when can adversarial agents corrupt multi-agent systems?
   - Oracle-based domains as testbed

2. **Mechanistic Findings**
   - Chaos requires verification: framing only works if honest agents can check claims
   - Herd immunity: honest agents compensate up to a ratio threshold
   - Phase boundary R*: critical chaos ratio where compensation fails

3. **Experimental Evidence** (Runs 1-6)
   - Run 1-4: Baseline + 2-4 agent tests
   - Run 5-6: 8-agent scaling, phase boundary

4. **Phase Boundary Analysis**
   - Plot: chaos ratio vs evenness
   - Identify R* and characterize transition (sharp vs linear)
   - Scaling law: R*(N) as function of agent count

5. **Security Implications**
   - Chaos is most dangerous when: feedback is slow, expensive, or delayed
   - Defense: verification speed and agent count both matter
   - Real threat scenarios (recommendation systems, distributed learning)

6. **Conclusion**
   - Phase transitions in adversarial swarms
   - Feedback-dependent threat model

---

## Contingencies

**If Run 8b stops early (hacking detection):**
- Use partial results (26-75 exps) for comparison to Run 7b
- Still shows language contagion without behavior change

**If Run 5 or 6 stops early:**
- Unlikely (oracle domain, high process quality)
- If so, use partial results for trend analysis

**If results don't match predictions:**
- Expected evenness 0.85–0.92 (Run 5) but get 0.75 → chaos more dangerous than expected
- Expected evenness 0.60–0.75 (Run 6) but get >0.90 → herd immunity stronger than expected
- Recalibrate hypothesis, investigate mechanism

---

## Success Criteria

- [ ] Phase boundary plot generated
- [ ] R* identified (or confirmed nonexistent)
- [ ] Per-agent behavior tables complete
- [ ] Committed to GitHub
- [ ] Ready for paper writing

---

**Trigger for execution:** Run 5 completes (~13:18 UTC) and Run 6 completes (~13:26 UTC)

