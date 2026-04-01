#!/usr/bin/env python3
"""
prediction_scorer.py — Score agent prediction calibration from blackboard.md + results.tsv

For each CLAIMED/CLAIM pair, extracts:
  - The numeric prediction (if any)
  - The actual result
  - Whether the prediction was directionally correct
  - Whether the actual result fell within the predicted range
  - Prediction specificity (numeric range / vague / none)
  - Mechanism quality (causal / correlational / intuition / none)

Mechanism quality answers: does the agent know WHY before predicting WHAT?
OmegaPRM insight: process supervision finds the first error in the reasoning chain.
A prediction with no mechanism is a guess — even if the number is right.

Usage:
    python3 tools/prediction_scorer.py domains/gpt2-tinystories-v44

Output:
    - Per-experiment breakdown
    - Aggregate calibration stats
    - Specificity vs accuracy correlation
    - Mechanism quality vs outcome correlation
"""

import re
import sys
import csv
from pathlib import Path
from dataclasses import dataclass, field


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PredictionRecord:
    exp_id: str
    agent: str
    description: str
    actual_score: float | None
    actual_status: str  # keep / discard / crash
    raw_prediction: str  # full prediction text
    pred_low: float | None   # lower bound of predicted range
    pred_high: float | None  # upper bound
    pred_direction: str      # "improve" / "worsen" / "neutral" / "unknown"
    specificity: str         # "numeric_range" / "numeric_point" / "qualitative" / "none" / "explicit_uncertain"
    direction_correct: bool | None   # None if can't determine
    in_range: bool | None            # None if no numeric prediction
    mechanism: str           # "causal" / "correlational" / "intuition" / "none"
    mechanism_text: str      # extracted mechanism clause (for inspection)


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_results_tsv(domain_dir: Path) -> dict[str, dict]:
    """Parse results.tsv into exp_id -> {score, status, description, agent}.

    Format: exp_id, score, vram, status, description, agent, design, train_min
    """
    results_path = domain_dir / "results.tsv"
    if not results_path.exists():
        return {}
    results = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            exp_id = cols[0].strip()
            try:
                score = float(cols[1]) if cols[1].strip() else None
            except ValueError:
                score = None
            # col[2] = vram, col[3] = status, col[4] = description, col[5] = agent
            status = cols[3].strip() if len(cols) > 3 else ""
            desc = cols[4].strip() if len(cols) > 4 else ""
            agent = cols[5].strip() if len(cols) > 5 else ""
            results[exp_id] = {
                "score": score,
                "status": status,
                "description": desc,
                "agent": agent,
            }
    return results


def extract_numeric_range(text: str) -> tuple[float | None, float | None]:
    """Extract a predicted BPB range like ~1.082-1.085 or ~1.098-1.100."""
    # Range pattern: ~1.082-1.085 or 1.082-1.085 or 1.082 to 1.085
    range_pattern = r'~?(\d+\.\d+)\s*[-–to]+\s*~?(\d+\.\d+)'
    m = re.search(range_pattern, text)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    # Point estimate: ~1.082 or Prediction: 1.082
    point_pattern = r'(?:prediction|expect)[:\s]*~?(\d+\.\d{3,})'
    m = re.search(point_pattern, text, re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1))
            return v, v
        except ValueError:
            pass

    # Standalone BPB value near prediction keyword
    standalone = r'~(\d+\.\d{3,})'
    matches = re.findall(standalone, text)
    if len(matches) == 1:
        try:
            v = float(matches[0])
            if 0.8 < v < 2.0:  # sanity check for BPB range
                return v, v
        except ValueError:
            pass

    return None, None


def classify_specificity(raw: str) -> str:
    """Classify prediction specificity."""
    if not raw:
        return "none"

    lower = raw.lower()

    # Explicit uncertainty markers
    uncertain_phrases = ["50/50", "high variance", "unknown", "can't tell", "no prediction"]
    if any(p in lower for p in uncertain_phrases):
        return "explicit_uncertain"

    # Check for numeric range
    lo, hi = extract_numeric_range(raw)
    if lo is not None:
        if lo == hi:
            return "numeric_point"
        else:
            return "numeric_range"

    # Qualitative
    qualitative_phrases = [
        "slight improvement", "slightly better", "slightly worse",
        "neutral", "worse", "better", "improvement", "regression",
        "should improve", "likely worse", "probably"
    ]
    if any(p in lower for p in qualitative_phrases):
        return "qualitative"

    return "none"


def classify_direction(raw: str) -> str:
    """Classify predicted direction: improve / worsen / neutral / unknown."""
    if not raw:
        return "unknown"
    lower = raw.lower()

    # Explicit uncertain
    if "50/50" in lower or "high variance" in lower:
        return "unknown"

    # Numeric: lower BPB = better (improvement)
    lo, hi = extract_numeric_range(raw)
    if lo is not None:
        # If both bounds are below 1.20, it's generally an improvement prediction
        # We need context of what the current best is — approximate heuristic
        return "improve"  # predictions are usually framed as improvements

    worsen_phrases = ["worse", "regression", "hurt", "harmful", "too tight", "too high", "overshoots"]
    improve_phrases = ["improvement", "better", "breakthrough", "new best", "could beat", "should improve", "slight improvement"]
    neutral_phrases = ["neutral", "within noise", "no change"]

    worsen_score = sum(1 for p in worsen_phrases if p in lower)
    improve_score = sum(1 for p in improve_phrases if p in lower)
    neutral_score = sum(1 for p in neutral_phrases if p in lower)

    if worsen_score > improve_score and worsen_score > neutral_score:
        return "worsen"
    if improve_score > worsen_score and improve_score > neutral_score:
        return "improve"
    if neutral_score > 0:
        return "neutral"
    return "unknown"


def classify_mechanism(full_text: str) -> tuple[str, str]:
    """
    Classify mechanism quality from the full CLAIMED block text.

    Levels (descending quality):
      causal        — agent states a specific technical reason using because/since/due to/
                      so that/which means + a domain-specific term (throughput, gradient,
                      attention, sequence, overfitting, etc.)
      correlational — agent references a past experiment or analogy ("similar to exp042",
                      "this pattern worked before", "following up on")
      intuition     — agent expresses belief without mechanism ("should help", "I think",
                      "worth trying", "might improve")
      none          — no explanation at all, or only restates what the change is

    Returns (level, extracted_clause).
    The clause is the sentence fragment that triggered the classification.
    """
    if not full_text:
        return "none", ""

    lower = full_text.lower()

    # ── Causal: requires a connective + a technical domain term ──────────────
    causal_connectives = [
        r'\bbecause\b', r'\bsince\b', r'\bdue to\b', r'\bwhich means\b',
        r'\bso that\b', r'\bthis (will|should|would) (reduce|increase|improve|lower|stabilize)\b',
        r'\bthe reason is\b', r'\bthis works by\b', r'\bby reducing\b', r'\bby increasing\b',
        r'\bthis (reduces|increases|improves|avoids|prevents)\b',
    ]
    domain_terms = [
        "throughput", "gradient", "attention", "sequence", "overfitting", "generalization",
        "warmup", "learning rate", "lr", "batch", "bpb", "bits per byte", "entropy",
        "weight decay", "momentum", "rope", "context", "embedding", "layer", "depth",
        "width", "head", "step count", "loss", "token", "norm", "residual", "schedule",
        "muon", "sgd", "adam", "activation", "relu", "gelu", "window", "stride",
        "capacity", "parameter", "cache", "kv", "flash", "sdpa", "memory", "vram",
    ]
    has_connective = any(re.search(c, lower) for c in causal_connectives)
    has_domain = any(t in lower for t in domain_terms)

    if has_connective and has_domain:
        # Extract the triggering sentence
        for connective in causal_connectives:
            m = re.search(r'([^.!?\n]{0,60}' + connective + r'[^.!?\n]{0,80})', lower)
            if m:
                return "causal", m.group(0).strip()[:120]
        return "causal", "(causal detected)"

    # ── Correlational: references a past experiment or analogy ────────────────
    correlational_patterns = [
        r'\bexp\d+\b',                          # mentions a specific experiment
        r'\bsimilar to\b', r'\bfollowing up\b',
        r'\blike the\b.*\bexperiment\b',
        r'\bprevious(ly)?\b.*\b(showed|found|worked|demonstrated)\b',
        r'\bbased on (exp|the last|the previous|results)\b',
        r'\blast experiment\b', r'\bprior run\b',
    ]
    for pat in correlational_patterns:
        m = re.search(pat, lower)
        if m:
            # Get surrounding context
            start = max(0, m.start() - 30)
            end = min(len(lower), m.end() + 60)
            return "correlational", lower[start:end].strip()[:120]

    # ── Intuition: belief without mechanism ───────────────────────────────────
    intuition_patterns = [
        r'\bshould (help|improve|work|reduce|lower|better)\b',
        r'\bi think\b', r'\bworth trying\b', r'\bmight (help|improve|work)\b',
        r'\bprobably (better|helps|lower)\b', r'\bhopefully\b',
        r'\bseems (like|worth|promising)\b', r'\blet.s try\b',
        r'\bcould (help|be better|improve)\b',
    ]
    for pat in intuition_patterns:
        m = re.search(pat, lower)
        if m:
            start = max(0, m.start() - 20)
            end = min(len(lower), m.end() + 60)
            return "intuition", lower[start:end].strip()[:120]

    return "none", ""


def extract_prediction_text(claimed_text: str) -> str:
    """Pull out just the prediction portion of a CLAIMED block."""
    # Look for explicit Prediction: label
    m = re.search(r'[Pp]rediction[:\s]+(.+?)(?:\n|$)', claimed_text)
    if m:
        return m.group(1).strip()

    # Look for "expect" keyword
    m = re.search(r'[Ee]xpect[s]?[:\s]+(.+?)(?:\n|$)', claimed_text)
    if m:
        return m.group(1).strip()

    # If no explicit prediction, return empty
    return ""


def parse_blackboard_saebench(blackboard_path: Path, results: dict) -> list[dict]:
    """
    Parse ### EXP{N}: format used in battlebotgym-sae-bench domains.

    Header: ### EXP{N}: description — F1={score} (label)
    Body sections: Hypothesis, Result, Why, Conclusion

    Score direction: higher F1 = better (opposite of BPB).
    """
    text = blackboard_path.read_text(errors="replace")
    lines = text.split("\n")

    entries = []
    current = None
    current_agent = "unknown"

    for line in lines:
        # Track which agent section we're in
        if re.match(r'^## Agent', line):
            current_agent = re.sub(r'^## ', '', line).strip().lower().replace(" ", "")

        if line.startswith("### EXP"):
            if current:
                current["full_text"] = "\n".join(current["lines"])
                entries.append(current)

            # Parse header: ### EXP{N}: description — F1=0.9175 (label)
            header_m = re.match(r'### EXP(\d+):\s*(.*)', line)
            if not header_m:
                current = None
                continue

            exp_num = header_m.group(1)
            rest = header_m.group(2)

            # Extract inline F1 score
            score_m = re.search(r'F1=([\d.]+)', rest)
            inline_score = float(score_m.group(1)) if score_m else None

            # Determine status from label
            label_lower = rest.lower()
            if "new best" in label_lower or "breakthrough" in label_lower:
                status = "keep"
            elif "worse" in label_lower or "failed" in label_lower:
                status = "discard"
            elif inline_score is not None:
                # Look up prior best to determine direction
                status = "keep"  # will refine below
            else:
                status = "unknown"

            # Description is everything before the dash+score
            desc_m = re.match(r'(.+?)\s*[—–-]\s*F1=', rest)
            desc = desc_m.group(1).strip() if desc_m else rest[:60]

            current = {
                "exp_id": f"exp{exp_num.zfill(3)}",
                "agent": current_agent,
                "lines": [line],
                "prediction_text": "",  # sae-bench uses Hypothesis, not Prediction
                "inline_score": inline_score,
                "status": status,
                "description": desc[:80],
                "full_text": "",
            }

        elif current is not None:
            current["lines"].append(line)

            # Extract hypothesis as prediction proxy
            if re.match(r'\s*-\s*Hypothesis:', line, re.IGNORECASE):
                hyp = re.sub(r'\s*-\s*Hypothesis:\s*', '', line, flags=re.IGNORECASE).strip()
                current["prediction_text"] = hyp

    if current:
        current["full_text"] = "\n".join(current["lines"])
        entries.append(current)

    # Post-process: determine keep/discard by comparing to running best
    running_best = 0.0
    for entry in entries:
        score = entry.get("inline_score")
        if score is None:
            continue
        if score > running_best:
            if entry["status"] == "unknown":
                entry["status"] = "keep"
            running_best = score
        elif entry["status"] == "unknown":
            entry["status"] = "discard"

    return entries


def detect_blackboard_format(blackboard_path: Path) -> str:
    """Auto-detect blackboard format: 'claimed' or 'saebench'."""
    text = blackboard_path.read_text(errors="replace")[:2000]
    if re.search(r'^### EXP\d+:', text, re.MULTILINE):
        return "saebench"
    return "claimed"


def parse_blackboard(blackboard_path: Path) -> list[dict]:
    """Parse CLAIMED entries from blackboard.md."""
    text = blackboard_path.read_text(errors="replace")

    # Split on CLAIMED lines
    entries = []
    # Match CLAIMED agentN: content
    claimed_pattern = re.compile(r'CLAIMED\s+(?:agent\w+):\s*\*\*?(exp\d+[^*\n]*?)\*\*?\s*[–-]?\s*(.*?)(?=CLAIMED|CLAIM |##|$)', re.DOTALL)

    for m in claimed_pattern.finditer(text):
        exp_hint = m.group(1).strip()
        content = m.group(2).strip()

        # Extract exp_id from hint
        exp_m = re.search(r'exp(\d+)', exp_hint, re.IGNORECASE)
        exp_id = f"exp{exp_m.group(1).zfill(3)}" if exp_m else None

        pred_text = extract_prediction_text(content)
        if not pred_text:
            # Try extracting from full content
            pred_text = extract_prediction_text(exp_hint + " " + content)

        entries.append({
            "exp_id": exp_id,
            "content": content,
            "prediction_text": pred_text,
            "raw_claimed": m.group(0)[:300],
        })

    return entries


def parse_blackboard_v2(blackboard_path: Path) -> list[dict]:
    """
    More robust parser: scan line by line for CLAIMED blocks,
    collect multi-line content until next CLAIMED/CLAIM/##.
    """
    text = blackboard_path.read_text(errors="replace")
    lines = text.split("\n")

    entries = []
    current = None

    for line in lines:
        if line.startswith("CLAIMED "):
            if current:
                entries.append(current)
            exp_m = re.search(r'\bexp(\d+)\b', line, re.IGNORECASE)
            exp_id = f"exp{exp_m.group(1).zfill(3)}" if exp_m else None
            current = {
                "exp_id": exp_id,
                "lines": [line],
                "prediction_text": "",
            }
        elif line.startswith("CLAIM "):
            # CLAIM lines often contain the result + a new prediction for next exp
            # Treat as a source of predictions too, keyed by exp mentioned
            if current:
                entries.append(current)
            # Extract exp_id — CLAIM lines reference the experiment that just ran
            exp_m = re.search(r'\bexp(\d+)\b', line, re.IGNORECASE)
            exp_id = f"exp{exp_m.group(1).zfill(3)}" if exp_m else None
            # Only capture if there's a Prediction: in the line or next few lines
            current = {
                "exp_id": exp_id,
                "lines": [line],
                "prediction_text": "",
                "from_claim": True,
            }
        elif line.startswith("## "):
            if current:
                entries.append(current)
                current = None
        else:
            if current is not None:
                current["lines"].append(line)

    if current:
        entries.append(current)

    # Post-process: extract prediction text from collected lines
    for entry in entries:
        full_text = "\n".join(entry.get("lines", []))
        entry["prediction_text"] = extract_prediction_text(full_text)
        entry["full_text"] = full_text

    return entries


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_prediction(entry: dict, result: dict | None) -> PredictionRecord:
    exp_id = entry.get("exp_id") or "unknown"
    pred_text = entry.get("prediction_text", "")
    full_text = entry.get("full_text", "")

    # saebench entries carry score/status inline; fallback to results lookup
    if entry.get("inline_score") is not None:
        actual_score = entry["inline_score"]
        actual_status = entry.get("status", "unknown")
        agent = entry.get("agent", "unknown")
        desc = entry.get("description", "")
    elif result:
        actual_score = result["score"]
        actual_status = result["status"]
        agent = result["agent"]
        desc = result["description"]
    else:
        actual_score = None
        actual_status = "unknown"
        agent = "unknown"
        desc = ""

    lo, hi = extract_numeric_range(pred_text) if pred_text else (None, None)
    specificity = classify_specificity(pred_text)
    direction = classify_direction(pred_text)
    mechanism, mechanism_text = classify_mechanism(full_text)

    # Direction correctness: did actual status match predicted direction?
    direction_correct = None
    if actual_status in ("keep",) and direction == "improve":
        direction_correct = True
    elif actual_status in ("discard",) and direction == "worsen":
        direction_correct = True
    elif actual_status in ("keep",) and direction == "worsen":
        direction_correct = False
    elif actual_status in ("discard",) and direction == "improve":
        direction_correct = False
    elif direction == "neutral" and actual_status == "discard":
        direction_correct = True  # neutral → discard is consistent
    # crash / unknown = can't determine

    # Range accuracy
    in_range = None
    if lo is not None and actual_score is not None:
        if lo == hi:
            in_range = abs(actual_score - lo) < 0.003  # within 0.003 of point estimate
        else:
            in_range = lo <= actual_score <= hi

    return PredictionRecord(
        exp_id=exp_id,
        agent=agent,
        description=desc[:80],
        actual_score=actual_score,
        actual_status=actual_status,
        raw_prediction=pred_text[:150],
        pred_low=lo,
        pred_high=hi,
        pred_direction=direction,
        specificity=specificity,
        direction_correct=direction_correct,
        in_range=in_range,
        mechanism=mechanism,
        mechanism_text=mechanism_text[:100],
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/prediction_scorer.py domains/<domain>")
        sys.exit(1)

    domain_dir = Path(sys.argv[1])
    blackboard_path = domain_dir / "blackboard.md"
    results_path = domain_dir / "results.tsv"

    if not blackboard_path.exists():
        print(f"Error: {blackboard_path} not found")
        sys.exit(1)

    results = parse_results_tsv(domain_dir)
    fmt = detect_blackboard_format(blackboard_path)

    if fmt == "saebench":
        claimed_entries = parse_blackboard_saebench(blackboard_path, results)
        print(f"[format: saebench — {len(claimed_entries)} EXP blocks parsed]")
    else:
        claimed_entries = parse_blackboard_v2(blackboard_path)
        print(f"[format: claimed — {len(claimed_entries)} CLAIMED blocks parsed]")

    records: list[PredictionRecord] = []
    for entry in claimed_entries:
        exp_id = entry.get("exp_id")
        # saebench has inline scores; claimed needs results lookup
        if entry.get("inline_score") is not None:
            result = None  # score_prediction reads from entry directly
        else:
            result = results.get(exp_id) if exp_id else None
            if result is None and exp_id:
                for k in results:
                    if k.lstrip("exp").lstrip("0") == exp_id.lstrip("exp").lstrip("0"):
                        result = results[k]
                        break
        rec = score_prediction(entry, result)
        if rec.exp_id != "unknown":
            records.append(rec)

    # Sort by exp_id
    records.sort(key=lambda r: r.exp_id)

    # ── Print per-experiment breakdown ──
    print(f"\n{'='*110}")
    print(f"PREDICTION CALIBRATION REPORT — {domain_dir.name}")
    print(f"{'='*110}")
    print(f"{'Exp':<8} {'Status':<9} {'Score':<8} {'Specificity':<20} {'Mech':<15} {'Dir✓':<6} {'InRange':<8} Prediction")
    print(f"{'-'*110}")

    for r in records:
        score_str = f"{r.actual_score:.4f}" if r.actual_score else r.actual_status
        dir_ok = "✓" if r.direction_correct is True else ("✗" if r.direction_correct is False else "-")
        in_rng = "✓" if r.in_range is True else ("✗" if r.in_range is False else "-")
        pred_short = r.raw_prediction[:42] if r.raw_prediction else "(no prediction)"
        print(f"{r.exp_id:<8} {r.actual_status:<9} {score_str:<8} {r.specificity:<20} {r.mechanism:<15} {dir_ok:<6} {in_rng:<8} {pred_short}")

    # ── Aggregate stats ──
    print(f"\n{'='*60}")
    print("AGGREGATE STATS")
    print(f"{'='*60}")

    total = len(records)
    with_result = [r for r in records if r.actual_score is not None]

    # Specificity breakdown
    spec_counts = {}
    for r in records:
        spec_counts[r.specificity] = spec_counts.get(r.specificity, 0) + 1

    print(f"\nTotal CLAIMED entries parsed: {total}")
    print(f"Matched to results.tsv:       {len(with_result)}")
    print(f"\nSpecificity breakdown:")
    for spec, count in sorted(spec_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"  {spec:<22} {count:>4}  ({pct:.0f}%)")

    # Direction accuracy by specificity
    print(f"\nDirection accuracy by specificity:")
    print(f"  {'Specificity':<22} {'Total':<7} {'Correct':<9} {'Accuracy'}")
    for spec in ["numeric_range", "numeric_point", "qualitative", "explicit_uncertain", "none"]:
        subset = [r for r in records if r.specificity == spec and r.direction_correct is not None]
        if not subset:
            continue
        correct = sum(1 for r in subset if r.direction_correct)
        pct = 100 * correct / len(subset)
        print(f"  {spec:<22} {len(subset):<7} {correct:<9} {pct:.0f}%")

    # Range accuracy for numeric predictions
    numeric = [r for r in records if r.in_range is not None]
    if numeric:
        in_range = sum(1 for r in numeric if r.in_range)
        print(f"\nNumeric range accuracy: {in_range}/{len(numeric)} = {100*in_range/len(numeric):.0f}%")

    # Outcome by specificity (do specific predictors have better hit rate?)
    print(f"\nOutcome (keep rate) by specificity:")
    print(f"  {'Specificity':<22} {'Total':<7} {'Keeps':<7} {'Keep%'}")
    for spec in ["numeric_range", "numeric_point", "qualitative", "explicit_uncertain", "none"]:
        subset = [r for r in records if r.specificity == spec and r.actual_status in ("keep", "discard")]
        if not subset:
            continue
        keeps = sum(1 for r in subset if r.actual_status == "keep")
        pct = 100 * keeps / len(subset)
        print(f"  {spec:<22} {len(subset):<7} {keeps:<7} {pct:.0f}%")

    # Worst mispredictions (predicted improve, got crash/discard badly)
    print(f"\nWorst mispredictions (predicted improve, got discard):")
    mispred = [r for r in records if r.direction_correct is False and r.actual_score is not None]
    mispred.sort(key=lambda r: r.actual_score or 99, reverse=True)
    for r in mispred[:8]:
        print(f"  {r.exp_id}: actual={r.actual_score:.4f}, predicted={r.raw_prediction[:60]}")

    # ── Mechanism quality analysis ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MECHANISM QUALITY (OmegaPRM signal)")
    print("Does stating WHY correlate with better outcomes?")
    print(f"{'='*60}")

    mech_counts = {}
    for r in records:
        mech_counts[r.mechanism] = mech_counts.get(r.mechanism, 0) + 1

    print(f"\nMechanism breakdown:")
    for mech, count in sorted(mech_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"  {mech:<16} {count:>4}  ({pct:.0f}%)")

    print(f"\nOutcome (keep rate) by mechanism quality:")
    print(f"  {'Mechanism':<16} {'Total':<7} {'Keeps':<7} {'Keep%':<8} {'Dir✓%'}")
    for mech in ["causal", "correlational", "intuition", "none"]:
        subset = [r for r in records if r.mechanism == mech and r.actual_status in ("keep", "discard")]
        if not subset:
            continue
        keeps = sum(1 for r in subset if r.actual_status == "keep")
        keep_pct = 100 * keeps / len(subset)
        dir_sub = [r for r in subset if r.direction_correct is not None]
        dir_pct_str = f"{100 * sum(1 for r in dir_sub if r.direction_correct) / len(dir_sub):.0f}%" if dir_sub else "n/a"
        print(f"  {mech:<16} {len(subset):<7} {keeps:<7} {keep_pct:<8.0f} {dir_pct_str}")

    # Numeric range accuracy by mechanism
    print(f"\nNumeric range accuracy by mechanism:")
    for mech in ["causal", "correlational", "intuition", "none"]:
        subset = [r for r in records if r.mechanism == mech and r.in_range is not None]
        if not subset:
            continue
        hit = sum(1 for r in subset if r.in_range)
        print(f"  {mech:<16} {hit}/{len(subset)} = {100*hit/len(subset):.0f}%")

    # Show best-quality causal predictions that were WRONG — first error signal
    print(f"\nCausal predictions that misfired (mechanism stated but wrong outcome):")
    causal_wrong = [r for r in records if r.mechanism == "causal" and r.direction_correct is False]
    causal_wrong.sort(key=lambda r: r.actual_score or 99, reverse=True)
    if causal_wrong:
        for r in causal_wrong[:6]:
            print(f"  {r.exp_id}: actual={r.actual_score:.4f} | mechanism: {r.mechanism_text[:80]}")
    else:
        print("  (none — causal predictions were all directionally correct)")

    # Summary signal
    causal_keeps = [r for r in records if r.mechanism == "causal" and r.actual_status == "keep"]
    no_mech_keeps = [r for r in records if r.mechanism == "none" and r.actual_status == "keep"]
    causal_total = [r for r in records if r.mechanism == "causal" and r.actual_status in ("keep", "discard")]
    no_mech_total = [r for r in records if r.mechanism == "none" and r.actual_status in ("keep", "discard")]
    print(f"\nSignal summary:")
    if causal_total:
        print(f"  Causal explanations → keep rate: {100*len(causal_keeps)/len(causal_total):.0f}% ({len(causal_keeps)}/{len(causal_total)})")
    if no_mech_total:
        print(f"  No mechanism stated  → keep rate: {100*len(no_mech_keeps)/len(no_mech_total):.0f}% ({len(no_mech_keeps)}/{len(no_mech_total)})")


if __name__ == "__main__":
    main()
