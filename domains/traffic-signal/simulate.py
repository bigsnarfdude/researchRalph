#!/usr/bin/env python3
"""
Traffic signal timing simulator — 3×3 intersection grid.

Vehicles travel straight through (no turns): N→S, S→N, W→E, E→W.
Morning peak: N-S demand (DEMAND_NS=0.30) is heavier than E-W (DEMAND_EW=0.18).

Signal cycle per intersection: NS_green → yellow → EW_green → yellow
`offset` shifts when the cycle starts — enables green-wave coordination.

Outputs (to stdout):
  avg_delay: <float>     mean vehicle delay in steps (lower is better)
  throughput: <int>      vehicles that exited
  max_queue: <int>       peak total queue length
  avg_queue: <float>     average residual queue at exit
  ns_delay: <float>      avg delay for N-S vehicles
  ew_delay: <float>      avg delay for E-W vehicles
  success: True
"""
import sys
import os
import json
import yaml
import numpy as np
from collections import deque

GRID = 3
STEPS = 800
YELLOW = 3
DISCHARGE = 2       # vehicles released per green step per approach
_DEMAND_NS = 0.45   # default: heavy morning peak
_DEMAND_EW = 0.10   # default: light cross traffic (4.5× asymmetry)
TRANSIT = 10        # steps to travel between adjacent intersections
SEED = 42

# Live override: if webcam-demand domain has written demand.json, use it
def _load_demand():
    here = os.path.dirname(os.path.abspath(__file__))
    demand_path = os.path.join(here, "..", "webcam-demand", "demand.json")
    try:
        with open(demand_path) as f:
            d = json.load(f)
        ns = float(d.get("ns_rate", _DEMAND_NS))
        ew = float(d.get("ew_rate", _DEMAND_EW))
        if ns > 0 and ew > 0:
            return ns, ew
    except Exception:
        pass
    return _DEMAND_NS, _DEMAND_EW

DEMAND_NS, DEMAND_EW = _load_demand()


def parse_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    signals = {}
    for r in range(GRID):
        for c in range(GRID):
            k = f"i{r}{c}"
            gns = max(5, min(60, int(cfg.get(f"{k}_green_ns", 30))))
            gew = max(5, min(60, int(cfg.get(f"{k}_green_ew", 30))))
            off = int(cfg.get(f"{k}_offset", 0))
            cycle = gns + gew + 2 * YELLOW
            signals[(r, c)] = {
                "green_ns": gns,
                "green_ew": gew,
                "offset": off % cycle,
                "cycle": cycle,
            }
    return signals


def get_phase(sig, t):
    p = (t - sig["offset"]) % sig["cycle"]
    if p < sig["green_ns"]:
        return "NS"
    if p < sig["green_ns"] + YELLOW:
        return "YELLOW"
    if p < sig["green_ns"] + YELLOW + sig["green_ew"]:
        return "EW"
    return "YELLOW"


def simulate(signals):
    rng = np.random.default_rng(SEED)
    # Four approach queues per intersection: NS (heading south), SN (north),
    # WE (heading east), EW (heading west)
    queues = {
        (r, c): {"NS": deque(), "SN": deque(), "WE": deque(), "EW": deque()}
        for r in range(GRID) for c in range(GRID)
    }

    completed = []
    ns_delays = []
    ew_delays = []
    queue_samples = []
    vid = 0
    # in_transit: list of (arrive_at_step, r, c, direction, vehicle)
    in_transit = []

    for t in range(STEPS):
        # --- Flush vehicles that have finished transit ---
        still_moving = []
        for (arrive_t, r, c, d, v) in in_transit:
            if arrive_t <= t:
                queues[(r, c)][d].append(v)
            else:
                still_moving.append((arrive_t, r, c, d, v))
        in_transit = still_moving

        # --- Arrivals at boundary intersections ---
        for c in range(GRID):
            if rng.random() < DEMAND_NS:
                queues[(0, c)]["NS"].append({"birth": t, "flow": "ns"})
                vid += 1
            if rng.random() < DEMAND_NS:
                queues[(2, c)]["SN"].append({"birth": t, "flow": "ns"})
                vid += 1
        for r in range(GRID):
            if rng.random() < DEMAND_EW:
                queues[(r, 0)]["WE"].append({"birth": t, "flow": "ew"})
                vid += 1
            if rng.random() < DEMAND_EW:
                queues[(r, 2)]["EW"].append({"birth": t, "flow": "ew"})
                vid += 1

        # --- Signal discharge ---
        for r in range(GRID):
            for c in range(GRID):
                phase = get_phase(signals[(r, c)], t)

                if phase == "NS":
                    for _ in range(DISCHARGE):
                        if queues[(r, c)]["NS"]:
                            v = queues[(r, c)]["NS"].popleft()
                            nr = r + 1
                            if nr >= GRID:
                                d = t - v["birth"]; completed.append(d); ns_delays.append(d)
                            else:
                                in_transit.append((t + TRANSIT, nr, c, "NS", v))
                        if queues[(r, c)]["SN"]:
                            v = queues[(r, c)]["SN"].popleft()
                            nr = r - 1
                            if nr < 0:
                                d = t - v["birth"]; completed.append(d); ns_delays.append(d)
                            else:
                                in_transit.append((t + TRANSIT, nr, c, "SN", v))

                elif phase == "EW":
                    for _ in range(DISCHARGE):
                        if queues[(r, c)]["WE"]:
                            v = queues[(r, c)]["WE"].popleft()
                            nc = c + 1
                            if nc >= GRID:
                                d = t - v["birth"]; completed.append(d); ew_delays.append(d)
                            else:
                                in_transit.append((t + TRANSIT, r, nc, "WE", v))
                        if queues[(r, c)]["EW"]:
                            v = queues[(r, c)]["EW"].popleft()
                            nc = c - 1
                            if nc < 0:
                                d = t - v["birth"]; completed.append(d); ew_delays.append(d)
                            else:
                                in_transit.append((t + TRANSIT, r, nc, "EW", v))
                # YELLOW: no discharge

        # Sample total queue length every 10 steps
        if t % 10 == 0:
            total = sum(len(q[d]) for q in queues.values() for d in q)
            queue_samples.append(total)

    max_queue = max(queue_samples) if queue_samples else 0
    avg_queue = float(np.mean(queue_samples)) if queue_samples else 0.0
    final_queue = sum(len(q[d]) for q in queues.values() for d in q)

    if not completed:
        return 9999.0, 0, max_queue, avg_queue, 9999.0, 9999.0

    avg_delay = float(np.mean(completed))
    avg_ns = float(np.mean(ns_delays)) if ns_delays else 9999.0
    avg_ew = float(np.mean(ew_delays)) if ew_delays else 9999.0

    return avg_delay, len(completed), max_queue, avg_queue, avg_ns, avg_ew


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        signals = parse_config(config_path)
        avg_delay, throughput, max_queue, avg_queue, ns_delay, ew_delay = simulate(signals)
        print(f"avg_delay: {avg_delay:.3f}")
        print(f"throughput: {throughput}")
        print(f"max_queue: {max_queue}")
        print(f"avg_queue: {avg_queue:.3f}")
        print(f"ns_delay: {ns_delay:.3f}")
        print(f"ew_delay: {ew_delay:.3f}")
        print("success: True")
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        print("success: False")
        sys.exit(1)
