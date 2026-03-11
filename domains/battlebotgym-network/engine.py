#!/usr/bin/env python3
"""
Network Routing Optimizer — researchRalph Game Domain

A network of 20 nodes with weighted edges. Packets spawn at random sources
and need to reach random destinations. The agent configures routing weights
and congestion policies via config.yaml.

Score = average packet delivery rate (0.0-1.0) over simulated traffic.
"""

import yaml
import sys
import random
import math
import argparse
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

# --- Game Constants -----------------------------------------------------------

NUM_NODES = 20
NUM_PACKETS = 1000
DEFAULT_MATCHES = 50
TOPOLOGY_SEED = 7777  # Fixed topology seed (same network every time)

# --- Data Types ---------------------------------------------------------------

@dataclass
class Link:
    src: int
    dst: int
    bandwidth: int       # 1-10: packets per tick capacity
    latency: int         # 1-20: ticks to traverse
    loss_rate: float     # 0-0.3: base probability of packet loss
    # Runtime state (reset per simulation)
    queue: int = 0
    traffic_this_tick: int = 0
    total_traffic: int = 0
    total_dropped: int = 0

    def congestion_level(self) -> float:
        """0.0 = empty, 1.0+ = over capacity."""
        if self.bandwidth == 0:
            return float('inf')
        return self.traffic_this_tick / self.bandwidth

    def effective_loss_rate(self, max_queue_size: int) -> float:
        """Loss rate increases with congestion. Queue overflow = drop."""
        if self.queue >= max_queue_size:
            return 1.0  # Queue full, guaranteed drop
        # Base loss + congestion penalty
        congestion = self.congestion_level()
        if congestion <= 1.0:
            return self.loss_rate
        # Over capacity: loss scales with overload
        overload_penalty = min(0.8, (congestion - 1.0) * 0.4)
        return min(1.0, self.loss_rate + overload_penalty)


@dataclass
class Packet:
    packet_id: int
    src: int
    dst: int
    hops: int = 0
    current_node: int = -1
    delivered: bool = False
    dropped: bool = False
    queued_until: int = 0  # tick when packet exits current link
    retry_count: int = 0
    is_short_distance: bool = False  # for QoS priority


@dataclass
class Network:
    nodes: int
    links: Dict[Tuple[int, int], Link]
    adjacency: Dict[int, List[int]]


# --- Topology Generation -----------------------------------------------------

def generate_topology(seed: int = TOPOLOGY_SEED) -> Network:
    """
    Generate a fixed network topology: 20 nodes, ~40 edges.
    Some links are fast/reliable, some slow/lossy.
    The topology has structure: a backbone of well-connected nodes
    with peripheral nodes that have fewer, worse connections.
    """
    rng = random.Random(seed)
    n = NUM_NODES
    links = {}
    adjacency = {i: [] for i in range(n)}

    def add_link(a: int, b: int, bw: int, lat: int, loss: float):
        if a == b:
            return
        key_ab = (a, b)
        key_ba = (b, a)
        if key_ab not in links:
            links[key_ab] = Link(a, b, bw, lat, loss)
            links[key_ba] = Link(b, a, bw, lat, loss)
            adjacency[a].append(b)
            adjacency[b].append(a)

    # Backbone: nodes 0-7 form a well-connected core
    backbone = list(range(8))
    # Ring the backbone
    for i in range(len(backbone)):
        j = (i + 1) % len(backbone)
        a, b = backbone[i], backbone[j]
        add_link(a, b,
                 bw=rng.randint(4, 7),
                 lat=rng.randint(2, 6),
                 loss=round(rng.uniform(0.02, 0.10), 3))

    # Cross-links in backbone
    for _ in range(4):
        a, b = rng.sample(backbone, 2)
        add_link(a, b,
                 bw=rng.randint(3, 6),
                 lat=rng.randint(3, 8),
                 loss=round(rng.uniform(0.03, 0.12), 3))

    # Mid-tier: nodes 8-13, connected to backbone and each other
    mid_tier = list(range(8, 14))
    for node in mid_tier:
        # Connect to 1-2 backbone nodes
        for bb in rng.sample(backbone, rng.randint(1, 2)):
            add_link(node, bb,
                     bw=rng.randint(2, 5),
                     lat=rng.randint(4, 12),
                     loss=round(rng.uniform(0.05, 0.20), 3))
        # Connect to 1 other mid-tier node
        other = rng.choice([m for m in mid_tier if m != node])
        add_link(node, other,
                 bw=rng.randint(1, 4),
                 lat=rng.randint(5, 15),
                 loss=round(rng.uniform(0.08, 0.20), 3))

    # Periphery: nodes 14-19, poorly connected
    periphery = list(range(14, 20))
    for node in periphery:
        # Connect to 1-2 mid-tier or backbone nodes
        targets = rng.sample(backbone + mid_tier, rng.randint(1, 2))
        for t in targets:
            add_link(node, t,
                     bw=rng.randint(1, 3),
                     lat=rng.randint(8, 20),
                     loss=round(rng.uniform(0.10, 0.30), 3))

    # Add a few extra random edges to reach ~38 total
    all_nodes = list(range(n))
    attempts = 0
    while len(links) // 2 < 38 and attempts < 100:
        a, b = rng.sample(all_nodes, 2)
        if (a, b) not in links:
            add_link(a, b,
                     bw=rng.randint(1, 4),
                     lat=rng.randint(3, 18),
                     loss=round(rng.uniform(0.05, 0.25), 3))
        attempts += 1

    # Ensure connectivity: BFS from node 0
    visited = set()
    queue = [0]
    visited.add(0)
    while queue:
        node = queue.pop(0)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # Connect any unreachable nodes
    for node in range(n):
        if node not in visited:
            target = rng.choice(list(visited))
            add_link(node, target,
                     bw=rng.randint(1, 3),
                     lat=rng.randint(8, 20),
                     loss=round(rng.uniform(0.10, 0.25), 3))
            visited.add(node)

    return Network(nodes=n, links=links, adjacency=adjacency)


# --- Pathfinding --------------------------------------------------------------

def compute_link_cost(link: Link, cfg: dict, current_congestion: float = 0.0) -> float:
    """
    Compute the routing cost of a link based on config weights.
    Lower cost = preferred path.
    """
    latency_w = float(cfg.get('latency_weight', 0.5))
    bandwidth_w = float(cfg.get('bandwidth_weight', 0.5))
    loss_w = float(cfg.get('loss_weight', 0.5))
    hop_w = float(cfg.get('hop_weight', 0.5))
    load_balance = float(cfg.get('load_balance_factor', 0.0))

    # Normalize components to roughly similar scales
    latency_cost = link.latency / 20.0           # 0-1
    bandwidth_cost = 1.0 - (link.bandwidth / 10.0)  # 0-1 (lower bw = higher cost)
    loss_cost = link.loss_rate / 0.3              # 0-1
    hop_cost = 1.0                                # constant per hop

    # Congestion-aware cost (load balancing)
    congestion_cost = current_congestion * load_balance

    cost = (latency_w * latency_cost
            + bandwidth_w * bandwidth_cost
            + loss_w * loss_cost
            + hop_w * hop_cost
            + congestion_cost)

    # Avoid zero/negative costs
    return max(0.001, cost)


def find_path(network: Network, src: int, dst: int, cfg: dict,
              congestion_map: Dict[Tuple[int, int], float]) -> Optional[List[int]]:
    """Dijkstra's with config-weighted link costs."""
    if src == dst:
        return [src]

    dist = {i: float('inf') for i in range(network.nodes)}
    prev = {i: -1 for i in range(network.nodes)}
    dist[src] = 0.0
    pq = [(0.0, src)]

    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        if node == dst:
            break
        for neighbor in network.adjacency[node]:
            link = network.links.get((node, neighbor))
            if link is None:
                continue
            cong = congestion_map.get((node, neighbor), 0.0)
            cost = compute_link_cost(link, cfg, cong)
            new_dist = dist[node] + cost
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    if dist[dst] == float('inf'):
        return None

    # Reconstruct path
    path = []
    node = dst
    while node != -1:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path


# --- Simulation ---------------------------------------------------------------

def shortest_path_distance(network: Network, src: int, dst: int) -> int:
    """BFS hop count for QoS classification."""
    if src == dst:
        return 0
    visited = {src}
    queue = [(src, 0)]
    while queue:
        node, depth = queue.pop(0)
        for neighbor in network.adjacency[node]:
            if neighbor == dst:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return 999  # unreachable


def run_simulation(network: Network, cfg: dict, seed: int,
                   verbose: bool = False) -> dict:
    """
    Simulate packet routing through the network.

    Simulation runs in discrete ticks. Each tick:
    1. Spawn some packets (spread over ticks to simulate real traffic)
    2. For each active packet, advance along its route
    3. Track congestion, drops, deliveries
    """
    rng = random.Random(seed)

    # Config params
    max_queue = int(cfg.get('max_queue_size', 5))
    load_balance = float(cfg.get('load_balance_factor', 0.0))
    reroute_thresh = float(cfg.get('reroute_threshold', 0.5))
    ttl = int(cfg.get('ttl', 20))
    priority_ratio = float(cfg.get('priority_ratio', 0.0))
    retry_limit = int(cfg.get('retry_limit', 0))

    # Generate packets
    n_packets = NUM_PACKETS
    packets = []
    for i in range(n_packets):
        src = rng.randint(0, network.nodes - 1)
        dst = rng.randint(0, network.nodes - 1)
        while dst == src:
            dst = rng.randint(0, network.nodes - 1)
        p = Packet(packet_id=i, src=src, dst=dst, current_node=src)
        p.is_short_distance = shortest_path_distance(network, src, dst) <= 3
        packets.append(p)

    # Assign spawn ticks: spread packets across time
    spawn_ticks = {}
    num_ticks = 70  # simulation length
    for p in packets:
        tick = rng.randint(0, num_ticks // 4)  # spawn in first quarter (very bursty)
        if tick not in spawn_ticks:
            spawn_ticks[tick] = []
        spawn_ticks[tick].append(p)

    # Reset link state
    for link in network.links.values():
        link.queue = 0
        link.traffic_this_tick = 0
        link.total_traffic = 0
        link.total_dropped = 0

    # Track congestion for routing decisions
    congestion_map: Dict[Tuple[int, int], float] = {}

    # Active packets (currently being routed)
    active_packets: List[Tuple[Packet, List[int], int]] = []  # (packet, path, path_idx)
    delivered = 0
    dropped = 0
    retry_schedule: Dict[int, List[Packet]] = {}  # tick -> packets to retry

    log = []

    for tick in range(num_ticks):
        # Reset per-tick traffic counters and add background traffic
        for link in network.links.values():
            # Background traffic: other services using the network
            # Consumes ~40% of bandwidth on average
            bg_traffic = rng.randint(0, max(1, int(link.bandwidth * 0.8)))
            link.traffic_this_tick = bg_traffic

        # Spawn new packets
        newly_spawned = spawn_ticks.get(tick, [])
        # Also retry packets scheduled for this tick
        retrying = retry_schedule.pop(tick, [])

        for p in newly_spawned + retrying:
            if p.dropped and p.retry_count > retry_limit:
                continue
            if p.dropped:
                p.dropped = False
                p.hops = 0
                p.current_node = p.src

            path = find_path(network, p.current_node, p.dst, cfg, congestion_map)
            if path and len(path) >= 2:
                active_packets.append((p, path, 0))
            elif path and len(path) == 1:
                # Already at destination (shouldn't happen with src != dst)
                p.delivered = True
                delivered += 1

        # Advance active packets
        still_active = []
        for p, path, idx in active_packets:
            if p.delivered or p.dropped:
                continue

            # Check TTL
            if p.hops >= ttl:
                p.dropped = True
                dropped += 1
                if p.retry_count < retry_limit:
                    p.retry_count += 1
                    # Retry after a delay (exponential backoff)
                    retry_tick = tick + 3 * p.retry_count
                    if retry_tick < num_ticks:
                        if retry_tick not in retry_schedule:
                            retry_schedule[retry_tick] = []
                        retry_schedule[retry_tick].append(p)
                continue

            # Try to advance to next hop
            if idx + 1 >= len(path):
                # Reached destination
                p.delivered = True
                delivered += 1
                continue

            next_node = path[idx + 1]
            link_key = (path[idx], next_node)
            link = network.links.get(link_key)

            if link is None:
                # Link doesn't exist (shouldn't happen)
                p.dropped = True
                dropped += 1
                continue

            link.traffic_this_tick += 1
            link.total_traffic += 1

            # QoS: priority packets get reserved bandwidth
            is_priority = p.is_short_distance and priority_ratio > 0
            effective_bw = link.bandwidth
            if is_priority:
                # Priority packets use reserved portion
                pass  # They get preferential queue treatment below
            else:
                # Non-priority packets see reduced bandwidth if priority reservation active
                effective_bw = max(1, int(link.bandwidth * (1.0 - priority_ratio)))

            # Check congestion and loss
            eff_loss = link.effective_loss_rate(max_queue)

            # Add a small per-hop environmental loss (interference, bit errors)
            eff_loss = min(1.0, eff_loss + 0.06)

            # Retried packets face increased loss (network remembers bad state)
            if p.retry_count > 0:
                eff_loss = min(1.0, eff_loss + 0.08 * p.retry_count)

            # Priority packets have reduced loss (but not eliminated)
            if is_priority:
                eff_loss *= 0.5  # 50% loss reduction for priority

            if rng.random() < eff_loss:
                # Packet lost on this link
                link.total_dropped += 1
                p.dropped = True
                dropped += 1
                if p.retry_count < retry_limit:
                    p.retry_count += 1
                    p.current_node = p.src  # retry from source
                    # Retry after a delay (exponential backoff)
                    retry_tick = tick + 3 * p.retry_count
                    if retry_tick < num_ticks:
                        if retry_tick not in retry_schedule:
                            retry_schedule[retry_tick] = []
                        retry_schedule[retry_tick].append(p)
                continue

            # Packet advances
            p.hops += 1
            p.current_node = next_node

            # Check rerouting on congestion
            congestion = link.congestion_level()
            congestion_map[link_key] = congestion

            if congestion > reroute_thresh and idx + 2 < len(path):
                # Reroute: recompute path from current position
                new_path = find_path(network, p.current_node, p.dst, cfg, congestion_map)
                if new_path and len(new_path) >= 2:
                    still_active.append((p, new_path, 0))
                else:
                    still_active.append((p, path, idx + 1))
            else:
                still_active.append((p, path, idx + 1))

            # Update queue
            if link.traffic_this_tick > link.bandwidth:
                link.queue = min(max_queue, link.queue + 1)
            elif link.queue > 0:
                link.queue = max(0, link.queue - 1)

        active_packets = still_active

        if verbose and tick % 10 == 0:
            active_count = len(active_packets)
            log.append({
                'tick': tick,
                'delivered': delivered,
                'dropped': dropped,
                'active': active_count,
                'pending': sum(1 for t, ps in spawn_ticks.items()
                              if t > tick for _ in ps),
            })

    # Final accounting: count unique packet outcomes
    unique_delivered = sum(1 for p in packets if p.delivered)
    unique_dropped = sum(1 for p in packets if p.dropped and not p.delivered)
    unique_undelivered = sum(1 for p in packets
                            if not p.delivered and not p.dropped)
    # Include still-active packets as undelivered
    for p, path, idx in active_packets:
        if not p.delivered and not p.dropped:
            pass  # already counted above

    delivery_rate = unique_delivered / n_packets if n_packets > 0 else 0.0

    # Compute link utilization stats
    total_link_traffic = sum(l.total_traffic for l in network.links.values()) / 2
    total_link_drops = sum(l.total_dropped for l in network.links.values()) / 2
    max_congestion = max((l.total_traffic / max(1, l.bandwidth)
                         for l in network.links.values()), default=0)

    return {
        'delivery_rate': round(delivery_rate, 4),
        'delivered': unique_delivered,
        'dropped': unique_dropped,
        'undelivered': unique_undelivered,
        'total_packets': n_packets,
        'total_link_traffic': int(total_link_traffic),
        'total_link_drops': int(total_link_drops),
        'max_link_congestion': round(max_congestion, 2),
        'log': log if verbose else [],
    }


# --- Config Validation --------------------------------------------------------

def validate_config(cfg: dict) -> Tuple[bool, str]:
    """Check config is within allowed bounds."""
    checks = [
        ('latency_weight', 0.0, 1.0),
        ('bandwidth_weight', 0.0, 1.0),
        ('loss_weight', 0.0, 1.0),
        ('hop_weight', 0.0, 1.0),
        ('max_queue_size', 1, 20),
        ('load_balance_factor', 0.0, 1.0),
        ('reroute_threshold', 0.0, 1.0),
        ('ttl', 5, 50),
        ('priority_ratio', 0.0, 1.0),
        ('retry_limit', 0, 5),
    ]
    for name, lo, hi in checks:
        val = cfg.get(name)
        if val is None:
            continue  # will use default
        val = float(val)
        if val < lo or val > hi:
            return False, f"{name}={val} out of range [{lo}, {hi}]"

    return True, "OK"


# --- Default Config -----------------------------------------------------------

DEFAULT_CONFIG = {
    'latency_weight': 0.5,
    'bandwidth_weight': 0.5,
    'loss_weight': 0.5,
    'hop_weight': 0.5,
    'max_queue_size': 5,
    'load_balance_factor': 0.0,
    'reroute_threshold': 0.5,
    'ttl': 15,
    'priority_ratio': 0.0,
    'retry_limit': 0,
}


# --- Tournament ---------------------------------------------------------------

def run_tournament(cfg: dict, n_matches: int = DEFAULT_MATCHES,
                   base_seed: int = 42, verbose: bool = False) -> dict:
    """Run N simulations with different traffic patterns and return aggregate stats."""
    valid, msg = validate_config(cfg)
    if not valid:
        return {'error': msg, 'delivery_rate': 0.0}

    # Merge with defaults
    full_cfg = dict(DEFAULT_CONFIG)
    full_cfg.update(cfg)

    # Generate the fixed topology (same for all matches)
    network = generate_topology(TOPOLOGY_SEED)

    total_delivery = 0.0
    total_delivered = 0
    total_dropped = 0
    total_undelivered = 0
    total_packets = 0
    match_results = []

    for i in range(n_matches):
        # Deep-copy link state by regenerating (links have mutable state)
        net = generate_topology(TOPOLOGY_SEED)
        result = run_simulation(net, full_cfg, base_seed + i, verbose)
        total_delivery += result['delivery_rate']
        total_delivered += result['delivered']
        total_dropped += result['dropped']
        total_undelivered += result['undelivered']
        total_packets += result['total_packets']
        match_results.append(result)

    avg_delivery = total_delivery / n_matches

    return {
        'delivery_rate': round(avg_delivery, 4),
        'avg_delivered': round(total_delivered / n_matches, 1),
        'avg_dropped': round(total_dropped / n_matches, 1),
        'avg_undelivered': round(total_undelivered / n_matches, 1),
        'total_packets_per_sim': NUM_PACKETS,
        'matches': n_matches,
        'results': match_results if verbose else [],
    }


# --- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Network Routing Optimizer')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to routing config YAML')
    parser.add_argument('--matches', '-n', type=int, default=DEFAULT_MATCHES,
                        help=f'Number of simulations (default: {DEFAULT_MATCHES})')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed results (JSON)')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output full results as JSON')
    parser.add_argument('--topology', action='store_true',
                        help='Print network topology and exit')
    args = parser.parse_args()

    if args.topology:
        import json
        net = generate_topology(TOPOLOGY_SEED)
        topo = {}
        seen = set()
        for (a, b), link in net.links.items():
            if (b, a) in seen:
                continue
            seen.add((a, b))
            topo[f"{a}-{b}"] = {
                'bandwidth': link.bandwidth,
                'latency': link.latency,
                'loss_rate': link.loss_rate,
            }
        print(json.dumps({
            'nodes': net.nodes,
            'edges': len(topo),
            'links': topo,
            'adjacency': {str(k): v for k, v in net.adjacency.items()},
        }, indent=2))
        return

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_tournament(cfg, args.matches, args.seed, args.verbose)

    if 'error' in results:
        print(f"INVALID CONFIG: {results['error']}", file=sys.stderr)
        print("0.0")
        sys.exit(1)

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        # Print just the score (what researchRalph reads)
        print(results['delivery_rate'])

        # Print summary to stderr (visible to agent but not captured as score)
        print(f"--- Simulation Summary ---", file=sys.stderr)
        print(f"Delivery rate: {results['delivery_rate']:.1%} "
              f"({results['avg_delivered']:.0f} delivered, "
              f"{results['avg_dropped']:.0f} dropped, "
              f"{results['avg_undelivered']:.0f} undelivered "
              f"/ {results['total_packets_per_sim']} packets)",
              file=sys.stderr)
        print(f"Simulations: {results['matches']}", file=sys.stderr)


if __name__ == '__main__':
    main()
