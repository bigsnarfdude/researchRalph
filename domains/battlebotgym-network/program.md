# Network Routing Optimizer — Agent Instructions

## Task
Optimize a network's routing configuration by editing `config.yaml` to **maximize packet delivery rate** across a fixed 20-node network under simulated traffic (1000 packets per simulation, with background traffic competing for bandwidth).

## Harness
```bash
# How to run one experiment:
bash run.sh config.yaml
# Score (delivery rate 0.0-1.0) is printed to stdout
# Simulation summary is printed to stderr
```

**Budget:** ~5 seconds per experiment (50 simulations)

For detailed results:
```bash
python3 engine.py config.yaml --json --matches 50
```

To inspect the network topology:
```bash
python3 engine.py --topology
```

## What you edit
- `config.yaml` — routing weights, congestion policy, and QoS settings

### Parameters and Ranges

**Routing Weights** (control pathfinding cost function):
| Parameter | Range | Effect |
|-----------|-------|--------|
| latency_weight | 0.0-1.0 | How much to avoid slow links |
| bandwidth_weight | 0.0-1.0 | How much to prefer high-bandwidth links |
| loss_weight | 0.0-1.0 | How much to avoid lossy links |
| hop_weight | 0.0-1.0 | How much to prefer fewer hops |

**Congestion Policy** (control traffic management):
| Parameter | Range | Effect |
|-----------|-------|--------|
| max_queue_size | 1-20 | Packets buffered per link before dropping |
| load_balance_factor | 0.0-1.0 | Spread traffic (1.0) vs shortest path (0.0) |
| reroute_threshold | 0.0-1.0 | Congestion level that triggers rerouting |
| ttl | 5-50 | Max hops before packet is dropped |

**QoS** (quality of service):
| Parameter | Range | Effect |
|-----------|-------|--------|
| priority_ratio | 0.0-1.0 | Bandwidth reserved for short-distance packets |
| retry_limit | 0-5 | Times to retry a dropped packet |

### CRITICAL: Parameter Interactions
These parameters interact in non-obvious ways. Changing one often requires adjusting others:

- **TTL vs load_balance_factor:** Load balancing sends packets on longer paths. If TTL is too low, load-balanced packets get dropped for exceeding hop limit. If TTL is high, packets on suboptimal paths consume bandwidth longer.

- **retry_limit vs max_queue_size:** Retries inject extra traffic. With small queues, retries cause more drops which cause more retries (cascading failure). With large queues, retries have room to succeed.

- **priority_ratio vs loss_weight:** Prioritizing short-distance packets starves long-distance ones. If loss_weight is high, long packets route through already-stressed alternative links, making things worse.

- **load_balance_factor vs reroute_threshold:** Both spread traffic, but they interact. High load balancing with low reroute threshold causes excessive path recomputation. Too little of both causes hotspot congestion.

- **max_queue_size vs reroute_threshold:** Large queues mask congestion from the rerouting system. Packets queue up instead of triggering reroutes, then all get dropped when the queue overflows.

- **hop_weight vs ttl:** High hop weight finds short paths but concentrates traffic on backbone links. Low hop weight spreads traffic but needs higher TTL for longer paths to complete.

## Network Structure
The network has three tiers:
- **Backbone** (nodes 0-7): Well-connected, high bandwidth, low latency, low loss
- **Mid-tier** (nodes 8-13): Moderate connections, medium quality links
- **Periphery** (nodes 14-19): Few connections, low bandwidth, high latency, high loss

Most traffic must traverse the backbone, making it a congestion bottleneck. Peripheral nodes are hard to reach reliably.

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the game engine (read-only)

## Scoring
- Metric: **Delivery rate** (0.0 to 1.0)
- Direction: **higher is better**
- Baseline: ~0.5-0.6 (default config)
- Noise level: differences > 0.02 are signal (50 simulations gives ~+/-0.02 noise)
- A score of 0.70+ is strong. 0.80+ is excellent. 0.90+ is exceptional.

## Tips for Agents
1. **Start by understanding the topology** — run `python3 engine.py --topology` to see link properties. The backbone is where congestion happens.
2. **Loss avoidance matters most at first** — the default config underweights loss_rate, causing packets to traverse lossy links and get dropped.
3. **Load balancing is a double-edged sword** — it reduces congestion but increases path length. You need TTL headroom to use it.
4. **Retries can cascade** — each retry adds traffic that can trigger more drops. Only enable retries if your queues can handle the extra load.
5. **Priority ratio helps short packets but hurts long ones** — the net effect depends on your traffic mix and topology.
6. **Queue size hides congestion** — large queues prevent rerouting from activating. Small queues cause early drops but enable the rerouting system to work.
7. **Test one thing at a time** — change one parameter, measure the effect, record it.
8. **The optimal config is NOT "max everything"** — the interactions mean you must find a balance.

## Strategy Approaches to Explore
- **Loss Avoidance**: Crank loss_weight, accept longer paths, raise TTL to compensate
- **Load Balancer**: High load_balance_factor, moderate reroute_threshold, generous TTL
- **Queue Buffer**: Large max_queue_size, low reroute_threshold, let queues absorb bursts
- **Retry Tank**: High retry_limit, large queues, accept initial drops and recover
- **QoS Specialist**: High priority_ratio, optimize for short-distance delivery
- **Minimal Path**: High hop_weight + bandwidth_weight, fast paths through backbone

## File Protocol

### results.tsv (append-only, shared)
```
commit<tab>score<tab>memory_gb<tab>status<tab>description<tab>agent<tab>design
```
- `status`: keep / discard / crash / retest
- ALWAYS append with `>>`, never overwrite

### best/ (update only when you beat the global best)
```bash
cp config.yaml $(dirname $0)/best/config.yaml
```

### blackboard.md (shared collaboration, append-only)
```
CLAIM agentN: <finding> (evidence: <experiment_id>, <metric>)
RESPONSE agentN to agentM: <confirm/refute> — <reasoning>
REQUEST agentN to agentM|any: <what to test> (priority: high|medium|low)
```

### Memory files (per-agent, private)
- `memory/facts.md` — confirmed findings (e.g., "loss_weight 0.9 > 0.5, confirmed 0.68 delivery rate")
- `memory/failures.md` — dead ends, NEVER retry (e.g., "retry_limit=5 causes cascade failure")
- `memory/hunches.md` — worth testing later
- `scratch/hypothesis.md` — current theory
- `scratch/predictions.md` — predicted vs actual score

## Agent Lifecycle
1. Read strategy.md + blackboard.md + your memory files
2. Pick task from queue/ or become coordinator if empty
3. `cp best/config.yaml config.yaml`
4. Apply your changes, predict expected score
5. Run: `bash run.sh config.yaml`
6. Record everything: results.tsv, done/, predictions
7. Update memory: facts if confirmed, failures if dead end, hunches if unclear
8. If new best -> update best/ + strategy.md + CLAIM on blackboard
9. If queue empty -> become coordinator
10. Loop forever. Never stop. Never ask questions.

## Constraints
- All parameters must be within their stated ranges
- Invalid configs score 0.0
