# Search Strategy — network

## Current Best
- Score: 0.9473
- Config: best/config.yaml (lb=0.0, reroute=0.3, loss=0.6, priority=0.1, hop=1.0, bw=0.5, queue=20, retry=5, ttl=20)

## Phase: diversify (convergence detected at 0.9455)

## What works (high confidence)
- retry_limit=5 is critical (retry=0 → 0.71, retry=2 → 0.90)
- max_queue_size=20 needed (queue=3 → 0.68)
- load_balance_factor near 0.0-0.1 slightly better than 0.3
- loss_weight=0.6 marginally better than 0.5
- reroute_threshold=0.3 marginally better than 0.5
- low priority_ratio (0.1) slightly helps

## What fails (avoid)
- Tiny queues (3) → 0.68
- Max LB (1.0) with no loss_weight → 0.51
- retry=0 → 0.71
- High priority_ratio (0.9) → 0.93

## Untested
- latency_weight > 0.3
- max_queue_size > 20 (capped at 20) or 5-15 range
- Extreme combinations (high latency + high bw weights)
- priority_ratio=0.0
