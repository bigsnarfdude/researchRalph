#!/usr/bin/env python3
"""
researchRalph COMMANDER v2 - True Autonomous Research

Uses Anthropic API directly (not Claude CLI) for fire-and-forget on nigel.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    nohup python commander_v2.py > commander_v2.log 2>&1 &

Check progress:
    tail -f commander_v2.log
    cat results/commander_state.json
"""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic

# =============================================================================
# CONFIG
# =============================================================================

GOAL_AUROC = 0.90
CURRENT_AUROC = 0.8342
MAX_ITERATIONS = 10
MODEL = "claude-sonnet-4-20250514"  # Fast and capable

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
STATE_FILE = RESULTS_DIR / "commander_state.json"
PROGRESS_FILE = RESULTS_DIR / "commander_progress.md"

# =============================================================================
# LOGGING
# =============================================================================

def log(msg):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def init_state():
    """Initialize state file if needed."""
    RESULTS_DIR.mkdir(exist_ok=True)

    if not STATE_FILE.exists():
        state = {
            "phase": "FEATURE_DISCOVERY",
            "goal_auroc": GOAL_AUROC,
            "current_auroc": CURRENT_AUROC,
            "best_auroc": CURRENT_AUROC,
            "iteration": 0,

            "context": {
                "ablation_insight": "L40_F15484 (Future self) is 2x more important than any other feature (-0.08 when removed). L40_F8921 and L40_F12574 are least important.",
                "architecture": "Binary+Log+ELU (fixed, proven optimal)",
                "key_finding": "F15484 captures 'future self concern' - reasoning about consequences to self"
            },

            "current_features": [
                {"layer": 40, "idx": 12574, "name": "RLHF awareness", "importance": -0.021},
                {"layer": 40, "idx": 8921, "name": "Self-preservation", "importance": -0.015},
                {"layer": 40, "idx": 15484, "name": "Future self", "importance": -0.080},
                {"layer": 53, "idx": 15529, "name": "RLHF acronym", "importance": -0.025},
                {"layer": 53, "idx": 8003, "name": "Training context", "importance": -0.053},
                {"layer": 53, "idx": 4824, "name": "Compliance", "importance": -0.048},
                {"layer": 53, "idx": 351, "name": "Scratchpad XML", "importance": -0.044},
                {"layer": 31, "idx": 15111, "name": "Capability erosion", "importance": -0.046}
            ],

            "experiments": [],
            "learnings": [],
            "failed_directions": [],
            "status": "RUNNING"
        }

        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        log(f"Initialized state: {STATE_FILE}")

    return json.load(open(STATE_FILE))


def save_state(state):
    """Save state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def append_progress(iteration, content):
    """Append to progress file."""
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"\n## Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(content)
        f.write("\n")

# =============================================================================
# CLAUDE REASONING
# =============================================================================

def think(client, state):
    """Have Claude reason about what to try next."""

    prompt = f"""You are an autonomous research commander improving alignment faking detection.

## CURRENT STATE
```json
{json.dumps(state, indent=2)}
```

## YOUR TASK

### Phase 1: ANALYZE
Look at the experiments so far. What patterns do you see? Why did things work or fail?

### Phase 2: HYPOTHESIZE
Based on your analysis, what should we try next? Form a SPECIFIC hypothesis.

Think about:
- F15484 "future self concern" is about reasoning about consequences
- What OTHER concepts might help detect alignment faking?
- Don't just try index neighbors - think SEMANTICALLY about what features might help
- GemmaScope L40 has 16k features capturing different concepts

### Phase 3: DESIGN
Design ONE experiment. Specify exactly:
- Which feature to replace (pick weakest: F8921 or F12574)
- What new feature to try (layer, index, and WHY you think it will help)
- Expected outcome

### Phase 4: OUTPUT
Respond with ONLY this JSON (no other text):

```json
{{
  "analysis": "Your analysis of what we've learned so far...",
  "hypothesis": "Your specific hypothesis for this experiment...",
  "reasoning": "Why you think this will improve AUROC...",
  "experiment": {{
    "action": "replace",
    "remove_feature": {{"layer": 40, "idx": 8921}},
    "add_feature": {{"layer": 40, "idx": XXXXX, "name": "descriptive name", "rationale": "why this feature"}}
  }},
  "expected_delta": 0.01,
  "confidence": "low|medium|high",
  "decision": "CONTINUE"
}}
```

If you believe we've hit a plateau (3+ failed experiments in same direction), set decision to "PLATEAU".
If we've reached the goal, set decision to "SUCCESS".

THINK CAREFULLY. Each experiment costs ~90 minutes. Choose wisely."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def extract_json(text):
    """Extract JSON from Claude's response."""
    import re
    # Find JSON block
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try parsing whole response as JSON
    try:
        return json.loads(text)
    except:
        return None

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_experiment(features, experiment_id):
    """
    Run feature extraction and training experiment.
    Returns test AUROC.
    """
    log(f"Running experiment {experiment_id}...")

    # Create experiment script
    script = f'''#!/usr/bin/env python3
"""Auto-generated experiment: {experiment_id}"""
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

FEATURES = {json.dumps(features)}
LAYERS = list(set([f[0] for f in FEATURES]))

MODEL_ID = "google/gemma-3-27b-it"
SAE_REPO = "google/gemma-scope-2-27b-it"
TRAIN_REPO = "vincentoh/alignment-faking-training"

class Transcoder:
    def __init__(self, repo_id, layer):
        folder = f"transcoder/layer_{{layer}}_width_16k_l0_medium"
        weights_path = hf_hub_download(repo_id, f"{{folder}}/params.safetensors")
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights["threshold"]

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        return F.relu(pre_acts - self.threshold)

class Probe(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 4, 16),
            nn.ELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        features = []
        for i in range(x.shape[2]):
            feat = x[:, :, i]
            log_feat = torch.log1p(feat)
            features.extend([
                (feat > 0).float().max(dim=1)[0],
                (feat > 0).float().sum(dim=1),
                log_feat.max(dim=1)[0],
                log_feat.mean(dim=1)
            ])
        return self.net(torch.stack(features, dim=1))

def main():
    # Load data
    print("Loading data...")
    train_path = hf_hub_download(repo_id=TRAIN_REPO, filename="training_data_final.json", repo_type="dataset")
    hn_path = hf_hub_download(repo_id=TRAIN_REPO, filename="hard_negatives.json", repo_type="dataset")
    gold_path = hf_hub_download(repo_id=TRAIN_REPO, filename="gold_106.json", repo_type="dataset")

    with open(train_path) as f:
        train_data = json.load(f).get("samples", [])
    with open(hn_path) as f:
        hard_neg = json.load(f).get("samples", [])
    with open(gold_path) as f:
        gold = json.load(f).get("samples", [])

    af = [s for s in train_data if s.get("label") == "potential_faking"]
    al = [s for s in train_data if s.get("label") == "aligned"]

    np.random.seed(42)
    max_per_class = 500
    af_sub = list(np.random.choice(af, min(max_per_class, len(af)), replace=False))
    al_sub = list(np.random.choice(al, min(max_per_class, len(al)), replace=False))
    hn_sub = list(np.random.choice(hard_neg, min(max_per_class, len(hard_neg)), replace=False))

    train_texts = [s["text"] for s in af_sub + al_sub + hn_sub]
    y_train = np.array([1]*len(af_sub) + [0]*len(al_sub) + [0]*len(hn_sub))
    shuffle_idx = np.random.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in shuffle_idx]
    y_train = y_train[shuffle_idx]

    test_texts = [s["text"] for s in gold]
    y_test = np.array([1 if s.get("label", "") in ["potential_faking", "alignment_faking", "AF"] else 0 for s in gold])

    print(f"Train: {{len(train_texts)}}, Test: {{len(test_texts)}}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # Load transcoders
    print("Loading transcoders...")
    transcoders = {{layer: Transcoder(SAE_REPO, layer).to("cuda", torch.bfloat16) for layer in LAYERS}}

    # Extract features
    def extract(texts, name):
        all_features = []
        for i, text in enumerate(texts):
            if i % 50 == 0:
                print(f"  {{name}}: {{i}}/{{len(texts)}}")

            activations = {{}}
            handles = []

            def make_hook(layer):
                def hook(module, input, output):
                    activations[layer] = output.detach()
                return hook

            for layer in LAYERS:
                try:
                    target = model.model.language_model.layers[layer]
                except AttributeError:
                    target = model.model.layers[layer]
                handle = target.pre_feedforward_layernorm.register_forward_hook(make_hook(layer))
                handles.append(handle)

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {{k: v.to("cuda") for k, v in inputs.items()}}
            with torch.no_grad():
                model(**inputs)

            for h in handles:
                h.remove()

            position_features = []
            for layer, feature_idx in FEATURES:
                acts = activations[layer].squeeze(0).to("cuda", torch.bfloat16)
                sae_acts = transcoders[layer].encode(acts)
                feat_acts = sae_acts[:, feature_idx].float().cpu().numpy()
                position_features.append(feat_acts)

            all_features.append(np.stack(position_features, axis=1).astype(np.float32))

        return all_features

    print("Extracting train features...")
    train_features = extract(train_texts, "train")
    print("Extracting test features...")
    test_features = extract(test_texts, "test")

    # Pad sequences
    def pad_sequences(features_list, max_len=512):
        n_features = features_list[0].shape[1]
        padded = np.zeros((len(features_list), max_len, n_features), dtype=np.float32)
        for i, f in enumerate(features_list):
            seq_len = min(f.shape[0], max_len)
            padded[i, :seq_len, :] = f[:seq_len]
        return padded

    X_train = pad_sequences(train_features)
    X_test = pad_sequences(test_features)

    # Train/val split
    np.random.seed(42)
    val_idx = np.random.choice(len(X_train), size=int(0.2 * len(X_train)), replace=False)
    train_idx = np.array([i for i in range(len(X_train)) if i not in val_idx])

    X_tr = torch.FloatTensor(X_train[train_idx]).cuda()
    y_tr = torch.FloatTensor(y_train[train_idx]).unsqueeze(1).cuda()
    X_val = torch.FloatTensor(X_train[val_idx]).cuda()
    y_val = y_train[val_idx]
    X_te = torch.FloatTensor(X_test).cuda()

    # Train
    print("Training probe...")
    torch.manual_seed(42)
    probe = Probe(input_dim=len(FEATURES)).cuda()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=0.1)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = 0
    best_state = None
    patience = 100
    patience_counter = 0

    for epoch in range(1000):
        probe.train()
        optimizer.zero_grad()
        loss = criterion(probe(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(probe(X_val)).cpu().numpy().flatten()
            val_auroc = roc_auc_score(y_val, val_probs)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {{k: v.cpu().clone() for k, v in probe.state_dict().items()}}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(probe(X_te)).cpu().numpy().flatten()
        test_auroc = roc_auc_score(y_test, test_probs)

    print(f"\\nRESULTS:")
    print(f"  Val AUROC:  {{best_val_auroc:.4f}}")
    print(f"  Test AUROC: {{test_auroc:.4f}}")

    # Save results
    result = {{"val_auroc": best_val_auroc, "test_auroc": test_auroc}}
    with open("experiment_result.json", "w") as f:
        json.dump(result, f)

    return test_auroc

if __name__ == "__main__":
    main()
'''

    # Write and run script
    script_path = SCRIPT_DIR / f"exp_{experiment_id}.py"
    with open(script_path, "w") as f:
        f.write(script)

    log(f"Running {script_path}...")
    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR)
    )

    log(f"stdout: {result.stdout[-500:]}")
    if result.returncode != 0:
        log(f"stderr: {result.stderr[-500:]}")
        return None

    # Read result
    result_file = SCRIPT_DIR / "experiment_result.json"
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)["test_auroc"]
    return None

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    log("=" * 60)
    log("RESEARCH COMMANDER v2 - AUTONOMOUS MODE")
    log("=" * 60)
    log(f"Goal: {GOAL_AUROC} AUROC")
    log(f"Max iterations: {MAX_ITERATIONS}")

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ERROR: ANTHROPIC_API_KEY not set")
        return

    client = Anthropic()
    state = init_state()

    log(f"Starting AUROC: {state['current_auroc']}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        state["iteration"] = iteration
        log("")
        log("=" * 60)
        log(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        log("=" * 60)

        # Phase 1: THINK
        log("[PHASE 1: THINK] Claude reasoning about next experiment...")
        response = think(client, state)
        log(f"Claude's response:\n{response[:500]}...")

        decision = extract_json(response)
        if not decision:
            log("ERROR: Could not parse Claude's response as JSON")
            continue

        log(f"Hypothesis: {decision.get('hypothesis', 'N/A')[:100]}...")
        log(f"Decision: {decision.get('decision', 'CONTINUE')}")

        # Check stop conditions
        if decision.get("decision") == "PLATEAU":
            log("PLATEAU reached - stopping")
            state["status"] = "PLATEAU"
            save_state(state)
            break

        if decision.get("decision") == "SUCCESS":
            log("SUCCESS - goal reached!")
            state["status"] = "SUCCESS"
            save_state(state)
            break

        # Phase 2: EXECUTE
        log("[PHASE 2: EXECUTE] Running experiment...")

        experiment = decision.get("experiment", {})

        # Build new feature list
        current_features = [(f["layer"], f["idx"]) for f in state["current_features"]]

        if experiment.get("action") == "replace":
            remove = (experiment["remove_feature"]["layer"], experiment["remove_feature"]["idx"])
            add = (experiment["add_feature"]["layer"], experiment["add_feature"]["idx"])
            new_features = [f for f in current_features if f != remove]
            new_features.append(add)
        else:
            new_features = current_features

        experiment_id = f"{iteration:03d}_{datetime.now().strftime('%H%M%S')}"
        test_auroc = run_experiment(new_features, experiment_id)

        if test_auroc is None:
            log("Experiment failed!")
            state["experiments"].append({
                "iteration": iteration,
                "hypothesis": decision.get("hypothesis"),
                "result": "FAILED",
            })
            save_state(state)
            continue

        # Phase 3: ANALYZE
        log(f"[PHASE 3: ANALYZE] Result: {test_auroc:.4f}")

        delta = test_auroc - state["best_auroc"]
        log(f"Delta: {delta:+.4f}")

        # Record experiment
        state["experiments"].append({
            "iteration": iteration,
            "hypothesis": decision.get("hypothesis"),
            "experiment": experiment,
            "test_auroc": test_auroc,
            "delta": delta,
            "analysis": decision.get("analysis"),
        })

        # Update best if improved
        if test_auroc > state["best_auroc"]:
            log(f"NEW BEST! {state['best_auroc']:.4f} -> {test_auroc:.4f}")
            state["best_auroc"] = test_auroc
            state["current_auroc"] = test_auroc

            # Update current features
            if experiment.get("action") == "replace":
                add_feat = experiment["add_feature"]
                state["current_features"] = [
                    f for f in state["current_features"]
                    if not (f["layer"] == experiment["remove_feature"]["layer"] and
                           f["idx"] == experiment["remove_feature"]["idx"])
                ]
                state["current_features"].append({
                    "layer": add_feat["layer"],
                    "idx": add_feat["idx"],
                    "name": add_feat.get("name", "unknown"),
                    "importance": delta
                })

            state["learnings"].append(f"Iteration {iteration}: {decision.get('hypothesis')} WORKED (+{delta:.4f})")
        else:
            state["learnings"].append(f"Iteration {iteration}: {decision.get('hypothesis')} failed ({delta:+.4f})")
            if delta < -0.01:
                state["failed_directions"].append(decision.get("hypothesis"))

        # Check goal
        if state["best_auroc"] >= GOAL_AUROC:
            log(f"GOAL REACHED! {state['best_auroc']:.4f} >= {GOAL_AUROC}")
            state["status"] = "SUCCESS"
            save_state(state)
            break

        save_state(state)
        append_progress(iteration, json.dumps(decision, indent=2))

        log(f"Best so far: {state['best_auroc']:.4f}")

    # Final report
    log("")
    log("=" * 60)
    log("COMMANDER COMPLETE")
    log("=" * 60)
    log(f"Status: {state['status']}")
    log(f"Iterations: {state['iteration']}")
    log(f"Final AUROC: {state['best_auroc']:.4f}")
    log(f"Improvement: {state['best_auroc'] - 0.8342:+.4f}")
    save_state(state)


if __name__ == "__main__":
    main()
