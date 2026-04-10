#!/usr/bin/env python3
"""
Advisor Strategy Demo — CartPole Optimization

Demonstrates the advisor_20260301 pattern on the battlebotgym-cartpole domain:
  - Haiku executor: proposes configs, runs experiments, iterates
  - Opus advisor: consulted server-side when executor needs strategic guidance
  - The advisor is a one-line addition to the tools list; everything else is identical

Compare:
  python3 tools/advisor_demo.py                # Haiku + Opus advisor
  python3 tools/advisor_demo.py --solo         # Haiku alone

The advisor pattern wins when the executor hits a plateau and needs a directional
nudge (e.g., "try doubling response_sharpness") rather than just more tries.

API note: requires ANTHROPIC_API_KEY. Uses beta header advisor-tool-2026-03-01.
Costs: Haiku $0.80/$4.00 per M tokens; Opus $15/$75 per M tokens (advisor only).
"""

import sys
import json
import argparse
from pathlib import Path

import anthropic

# Import cartpole engine directly from the domain directory
sys.path.insert(0, str(Path(__file__).parent.parent / "domains/battlebotgym-cartpole"))
from engine import run_tournament, PARAM_RANGES, DEFAULTS


# ─── Experiment State ────────────────────────────────────────────────────────

history: list[dict] = []


# ─── Tool Implementations ────────────────────────────────────────────────────

def tool_run_experiment(cfg: dict) -> dict:
    result = run_tournament(cfg, n_episodes=50, base_seed=42)
    if "error" in result:
        return {"error": result["error"]}
    entry = {
        "experiment": len(history) + 1,
        "score": result["score"],
        "perfect_episodes": result["perfect_episodes"],
        "avg_steps": result["avg_steps"],
        "terminated_by_angle": result["terminated_by_angle"],
        "terminated_by_position": result["terminated_by_position"],
        "config": cfg,
    }
    history.append(entry)
    return entry


def tool_get_history() -> dict:
    if not history:
        return {
            "experiments": [],
            "best_score": None,
            "note": "No experiments yet. Baseline (defaults) ~0.39.",
            "param_ranges": {k: list(v) for k, v in PARAM_RANGES.items()},
            "defaults": DEFAULTS,
        }
    best = max(history, key=lambda x: x["score"])
    return {
        "total": len(history),
        "best_score": best["score"],
        "best_config": best["config"],
        "recent": history[-5:],
        "param_ranges": {k: list(v) for k, v in PARAM_RANGES.items()},
    }


# ─── Tool Schemas ─────────────────────────────────────────────────────────────

EXPERIMENT_TOOL = {
    "name": "run_experiment",
    "description": (
        "Run CartPole with a given controller config. "
        "Returns score (0.0–1.0, higher is better). "
        "Baseline ~0.39. Target: >0.95. "
        "Failure modes: 'terminated_by_angle' means pole fell, "
        "'terminated_by_position' means cart drifted off-screen."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "angle_weight":            {"type": "number", "description": "Pole angle signal weight [0, 10]. Most important parameter."},
            "angular_velocity_weight": {"type": "number", "description": "Pole angular velocity weight [0, 10]. Very important."},
            "position_weight":         {"type": "number", "description": "Cart position weight [0, 5]"},
            "velocity_weight":         {"type": "number", "description": "Cart velocity weight [0, 5]"},
            "angle_bias":              {"type": "number", "description": "Constant bias added to decision signal [-2, 2]"},
            "response_sharpness":      {"type": "number", "description": "Sigmoid steepness — how decisive the controller is [0.1, 20]"},
            "anticipation_horizon":    {"type": "number", "description": "Steps ahead to anticipate future angle [0, 2]"},
            "position_centering":      {"type": "number", "description": "Strength of push back to cart center [0, 5]"},
        },
        "required": [
            "angle_weight", "angular_velocity_weight", "position_weight",
            "velocity_weight", "angle_bias", "response_sharpness",
            "anticipation_horizon", "position_centering",
        ],
    },
}

HISTORY_TOOL = {
    "name": "get_history",
    "description": "Return experiment history, best score, best config, and parameter ranges.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

ADVISOR_TOOL = {
    "type": "advisor_20260301",
    "name": "advisor",
    "model": "claude-opus-4-6",
    "max_uses": 3,
}

SYSTEM_PROMPT = """You are a CartPole controller optimizer. Your goal: maximize score (0.0–1.0) by tuning 8 controller weights.

Physics: controller computes signal = angle*angle_weight + angular_velocity*angular_velocity_weight + position*position_weight + velocity*velocity_weight + bias, then sigmoid(signal * response_sharpness) decides push direction.

Optimization strategy:
1. Call get_history first to see parameter ranges and any existing results
2. Start with angle_weight=4, angular_velocity_weight=4, response_sharpness=5 as a strong baseline
3. If score < 0.80: increase angle_weight and angular_velocity_weight
4. If pole falls (terminated_by_angle dominant): increase response_sharpness
5. If cart drifts (terminated_by_position dominant): increase position_weight and position_centering
6. When you hit a plateau (no improvement in 3+ experiments), use the advisor tool for strategic guidance
7. Make one config change at a time for interpretable ablations

Aim for score > 0.95."""


# ─── Agent Loop ───────────────────────────────────────────────────────────────

def run(use_advisor: bool, max_experiments: int) -> tuple[float, dict]:
    """Run the agentic optimization loop. Returns (best_score, token_usage)."""
    client = anthropic.Anthropic()
    history.clear()

    tools = [EXPERIMENT_TOOL, HISTORY_TOOL]
    if use_advisor:
        tools.append(ADVISOR_TOOL)

    messages = [{
        "role": "user",
        "content": (
            f"Optimize the CartPole controller. Budget: {max_experiments} experiments. "
            "Maximize score. Start with get_history, then experiment."
        ),
    }]

    usage_totals = {
        "executor_input": 0,
        "executor_output": 0,
        "advisor_input": 0,
        "advisor_output": 0,
        "advisor_calls": 0,
    }

    for _turn in range(60):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
            extra_headers={"anthropic-beta": "advisor-tool-2026-03-01"} if use_advisor else {},
        )

        # Accumulate usage
        u = response.usage
        usage_totals["executor_input"] += u.input_tokens
        usage_totals["executor_output"] += u.output_tokens
        if use_advisor:
            adv_in = getattr(u, "advisor_input_tokens", None) or 0
            adv_out = getattr(u, "advisor_output_tokens", None) or 0
            usage_totals["advisor_input"] += adv_in
            usage_totals["advisor_output"] += adv_out
            if adv_in > 0:
                usage_totals["advisor_calls"] += 1
                print(f"    [Opus advisor consulted — turn {_turn + 1}]")

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "run_experiment":
                    result = tool_run_experiment(block.input)
                    n = result.get("experiment", "?")
                    score = result.get("score", "err")
                    cfg = block.input
                    print(
                        f"  exp {n:2}  score={score:.4f}  "
                        f"aw={cfg['angle_weight']:.1f} "
                        f"av={cfg['angular_velocity_weight']:.1f} "
                        f"rs={cfg['response_sharpness']:.1f} "
                        f"pc={cfg['position_centering']:.1f}"
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
                elif block.name == "get_history":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_get_history()),
                    })

            messages.append({"role": "assistant", "content": response.content})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        if len(history) >= max_experiments:
            break

    best_score = max((e["score"] for e in history), default=0.0)
    return best_score, usage_totals


# ─── Cost Estimate ────────────────────────────────────────────────────────────

def estimate_cost(usage: dict) -> float:
    # Haiku 4.5: $0.80 / $4.00 per M tokens (input/output)
    # Opus 4.6:  $15.00 / $75.00 per M tokens
    haiku_in  = 0.80  / 1e6
    haiku_out = 4.00  / 1e6
    opus_in   = 15.00 / 1e6
    opus_out  = 75.00 / 1e6
    return (
        usage["executor_input"]  * haiku_in  +
        usage["executor_output"] * haiku_out +
        usage["advisor_input"]   * opus_in   +
        usage["advisor_output"]  * opus_out
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Advisor strategy demo on CartPole")
    parser.add_argument("--solo", action="store_true", help="Haiku only, no advisor")
    parser.add_argument("--compare", action="store_true", help="Run both and print side-by-side")
    parser.add_argument("--max-experiments", type=int, default=15, metavar="N")
    args = parser.parse_args()

    runs = []

    if args.compare:
        configs = [("Haiku solo", False), ("Haiku + Opus advisor", True)]
    elif args.solo:
        configs = [("Haiku solo", False)]
    else:
        configs = [("Haiku + Opus advisor", True)]

    for label, use_advisor in configs:
        print(f"\n{'━'*58}")
        print(f"  {label}")
        print(f"{'━'*58}")
        best, usage = run(use_advisor=use_advisor, max_experiments=args.max_experiments)
        cost = estimate_cost(usage)
        runs.append((label, best, usage, cost))

        print(f"\n  best score:      {best:.4f}")
        print(f"  experiments:     {len(history)}")
        print(f"  executor tokens: {usage['executor_input']:,} in / {usage['executor_output']:,} out")
        if use_advisor:
            print(f"  advisor tokens:  {usage['advisor_input']:,} in / {usage['advisor_output']:,} out")
            print(f"  advisor calls:   {usage['advisor_calls']}")
        print(f"  estimated cost:  ${cost:.4f}")

    if args.compare and len(runs) == 2:
        (l1, s1, u1, c1), (l2, s2, u2, c2) = runs
        print(f"\n{'━'*58}")
        print(f"  Comparison")
        print(f"{'━'*58}")
        print(f"  {'':30s}  {'Solo':>8}  {'Advisor':>8}")
        print(f"  {'Best score':30s}  {s1:>8.4f}  {s2:>8.4f}  Δ={s2-s1:+.4f}")
        print(f"  {'Estimated cost':30s}  ${c1:>7.4f}  ${c2:>7.4f}  Δ=${c2-c1:+.4f}")
        print(f"  {'Executor input tokens':30s}  {u1['executor_input']:>8,}  {u2['executor_input']:>8,}")


if __name__ == "__main__":
    main()
