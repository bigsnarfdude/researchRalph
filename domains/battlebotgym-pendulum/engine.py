#!/usr/bin/env python3
"""
Pendulum Swing-Up & Balance Simulator — battleBOT Game Domain

An inverted pendulum starts hanging DOWN and must be swung UP and balanced.
The agent configures a two-phase controller (swing-up + PD balance) via
config.yaml. Score = normalized total reward averaged across episodes.

Score: 0.0 (worst) to ~0.95 (optimal swing-up + balance).
"""

import yaml
import sys
import math
import argparse
import json
from dataclasses import dataclass

# --- Physics Constants ---

GRAVITY = 10.0
MASS = 1.0
LENGTH = 1.0
DT = 0.05
MAX_TORQUE = 2.0
MAX_SPEED = 8.0
EPISODE_STEPS = 200
DEFAULT_MATCHES = 50

# Reward bounds
# Per step worst: -(pi^2 + 0.1*64 + 0.001*4) = -16.2736
# Per step at bottom (no torque): -(pi^2) = -9.8696
# Per step halfway up (pi/2 from upright): -(pi/2)^2 = -2.467
# Practical floor: pendulum swinging around bottom = ~-1200 total (with some velocity)
# Ceiling: a good controller balances for most of the episode ~ -50 to -200 total
# Using floor=-1200 and ceiling=0 to create meaningful score spread
WORST_REWARD_PER_STEP = -(math.pi ** 2 + 0.1 * MAX_SPEED ** 2 + 0.001 * MAX_TORQUE ** 2)
MIN_TOTAL_REWARD = WORST_REWARD_PER_STEP * EPISODE_STEPS  # ~ -3254.7
MAX_TOTAL_REWARD = 0.0

# Balance detection thresholds
BALANCE_THRESHOLD = 0.5    # rad from upright to count as "balanced"
VELOCITY_THRESHOLD = 2.0   # rad/s max velocity to count as "balanced"


# --- Pendulum State ---

@dataclass
class PendulumState:
    theta: float       # angle (0 = hanging down, pi = upright)
    theta_dot: float   # angular velocity


def angle_normalize(x: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def step(state: PendulumState, torque: float) -> tuple:
    """
    Advance pendulum by one timestep.
    Returns (new_state, reward).
    """
    torque = max(-MAX_TORQUE, min(MAX_TORQUE, torque))

    th = state.theta
    thdot = state.theta_dot

    # Dynamics: alpha = (-3g / 2l) * sin(theta + pi) + (3 / ml^2) * torque
    alpha = (-3.0 * GRAVITY / (2.0 * LENGTH)) * math.sin(th + math.pi) \
            + (3.0 / (MASS * LENGTH ** 2)) * torque

    new_thdot = thdot + alpha * DT
    new_thdot = max(-MAX_SPEED, min(MAX_SPEED, new_thdot))
    new_th = th + new_thdot * DT

    # Reward: penalize angle from upright, velocity, and torque
    # theta_normalized is angle from upright (pi)
    th_norm = angle_normalize(new_th - math.pi)
    reward = -(th_norm ** 2 + 0.1 * new_thdot ** 2 + 0.001 * torque ** 2)

    return PendulumState(new_th, new_thdot), reward


# --- Controller ---

def compute_torque(state: PendulumState, cfg: dict) -> float:
    """
    Two-phase controller:
      Phase 1 (swing-up): energy-based pumping when far from upright
      Phase 2 (balance):  PD control when near upright
    """
    swing_gain = float(cfg.get('swing_gain', 1.0))
    balance_gain_p = float(cfg.get('balance_gain_p', 10.0))
    balance_gain_d = float(cfg.get('balance_gain_d', 2.0))
    switch_angle = float(cfg.get('switch_angle', 1.0))
    energy_target = float(cfg.get('energy_target', 10.0))
    max_swing_torque = float(cfg.get('max_swing_torque', 2.0))
    max_balance_torque = float(cfg.get('max_balance_torque', 2.0))
    damping = float(cfg.get('damping', 0.0))
    anticipation = float(cfg.get('anticipation', 0.0))
    deadzone = float(cfg.get('deadzone', 0.0))

    th = state.theta
    thdot = state.theta_dot

    # Angle from upright (pi)
    angle_error = angle_normalize(th - math.pi)
    # Anticipated angle
    angle_error_anticipated = angle_error + anticipation * thdot * DT

    abs_angle = abs(angle_error)

    if abs_angle < switch_angle:
        # --- Balance phase (PD control) ---
        if abs(angle_error_anticipated) < deadzone:
            torque = 0.0
        else:
            torque = -balance_gain_p * angle_error_anticipated \
                     - balance_gain_d * thdot \
                     - damping * thdot
        torque = max(-max_balance_torque, min(max_balance_torque, torque))
    else:
        # --- Swing-up phase (energy pumping) ---
        # Current energy: E = 0.5 * m * l^2 * thdot^2 + m * g * l * (1 - cos(th))
        # At upright (th=pi), cos(pi)=-1, potential = 2*m*g*l = 20
        energy = 0.5 * MASS * LENGTH ** 2 * thdot ** 2 \
                 + MASS * GRAVITY * LENGTH * (1.0 - math.cos(th))
        energy_error = energy - energy_target

        # Pump: apply torque in direction of velocity, modulated by energy error
        if thdot != 0.0:
            sign_vel = 1.0 if thdot > 0 else -1.0
        else:
            sign_vel = 1.0 if math.cos(th) > 0 else -1.0

        torque = swing_gain * sign_vel * math.cos(th)
        torque -= damping * thdot  # velocity damping
        torque = max(-max_swing_torque, min(max_swing_torque, torque))

    return torque


# --- Episode ---

def run_episode(cfg: dict, seed: int, verbose: bool = False) -> dict:
    """Run a single pendulum episode. Returns result dict."""
    import random
    rng = random.Random(seed)

    # Random initial state: angle in [-pi, pi], velocity in [-1, 1]
    theta0 = rng.uniform(-math.pi, math.pi)
    thdot0 = rng.uniform(-1.0, 1.0)
    state = PendulumState(theta0, thdot0)

    total_reward = 0.0
    balanced_steps = 0
    log = []

    for t in range(EPISODE_STEPS):
        torque = compute_torque(state, cfg)
        state, reward = step(state, torque)
        total_reward += reward

        # Count steps where pendulum is "balanced" (within 0.5 rad of upright
        # and angular velocity < 2.0 rad/s)
        angle_err = angle_normalize(state.theta - math.pi)
        if abs(angle_err) < BALANCE_THRESHOLD and abs(state.theta_dot) < VELOCITY_THRESHOLD:
            balanced_steps += 1

        if verbose:
            log.append({
                'step': t,
                'theta': round(state.theta, 4),
                'theta_dot': round(state.theta_dot, 4),
                'torque': round(torque, 4),
                'angle_from_upright': round(angle_err, 4),
                'reward': round(reward, 4),
            })

    # Score: combination of reward normalization and balance time
    # 70% weight on balanced fraction, 30% on reward normalization
    # This ensures agents must actually swing up and balance, not just avoid bad states
    reward_score = (total_reward - MIN_TOTAL_REWARD) / (MAX_TOTAL_REWARD - MIN_TOTAL_REWARD)
    reward_score = max(0.0, min(1.0, reward_score))
    balance_fraction = balanced_steps / EPISODE_STEPS
    score = 0.3 * reward_score + 0.7 * balance_fraction
    score = max(0.0, min(1.0, score))

    return {
        'total_reward': round(total_reward, 4),
        'score': round(score, 4),
        'reward_score': round(reward_score, 4),
        'balance_fraction': round(balance_fraction, 4),
        'balanced_steps': balanced_steps,
        'initial_theta': round(theta0, 4),
        'initial_thdot': round(thdot0, 4),
        'final_theta': round(state.theta, 4),
        'final_thdot': round(state.theta_dot, 4),
        'log': log if verbose else [],
    }


def validate_config(cfg: dict) -> tuple:
    """Check config is within allowed bounds."""
    checks = [
        ('swing_gain', 0.0, 10.0),
        ('balance_gain_p', 0.0, 20.0),
        ('balance_gain_d', 0.0, 10.0),
        ('switch_angle', 0.1, 3.14),
        ('energy_target', 0.0, 30.0),
        ('max_swing_torque', 0.1, 2.0),
        ('max_balance_torque', 0.1, 2.0),
        ('damping', 0.0, 5.0),
        ('anticipation', 0.0, 5.0),
        ('deadzone', 0.0, 0.5),
    ]
    for name, lo, hi in checks:
        val = float(cfg.get(name, 0))
        if val < lo or val > hi:
            return False, f"{name}={val} out of range [{lo}, {hi}]"
    return True, "OK"


def run_experiment(cfg: dict, n_episodes: int = DEFAULT_MATCHES,
                   base_seed: int = 42, verbose: bool = False) -> dict:
    """Run N episodes and return aggregate stats."""
    valid, msg = validate_config(cfg)
    if not valid:
        return {'error': msg, 'score': 0.0}

    total_score = 0.0
    total_reward_sum = 0.0
    total_balance_fraction = 0.0
    episode_results = []

    for i in range(n_episodes):
        result = run_episode(cfg, base_seed + i, verbose)
        total_score += result['score']
        total_reward_sum += result['total_reward']
        total_balance_fraction += result['balance_fraction']
        episode_results.append(result)

    avg_score = total_score / n_episodes
    avg_reward = total_reward_sum / n_episodes
    avg_balance = total_balance_fraction / n_episodes

    # Per-episode score stats
    scores = [r['score'] for r in episode_results]
    scores_sorted = sorted(scores)
    median_score = scores_sorted[len(scores_sorted) // 2]
    min_score = scores_sorted[0]
    max_score = scores_sorted[-1]

    return {
        'score': round(avg_score, 4),
        'avg_reward': round(avg_reward, 2),
        'avg_balance_fraction': round(avg_balance, 4),
        'median_score': round(median_score, 4),
        'min_score': round(min_score, 4),
        'max_score': round(max_score, 4),
        'episodes': n_episodes,
        'results': episode_results if verbose else [],
    }


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description='Pendulum Swing-Up & Balance Simulator')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to controller config YAML')
    parser.add_argument('--matches', '-n', type=int, default=DEFAULT_MATCHES,
                        help=f'Number of episodes (default: {DEFAULT_MATCHES})')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Include per-step logs in output')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output full results as JSON')
    args = parser.parse_args()

    # Load controller config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_experiment(cfg, args.matches, args.seed, args.verbose)

    if 'error' in results:
        print(f"INVALID CONFIG: {results['error']}", file=sys.stderr)
        print("0.0")
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Print just the score (what the harness reads)
        print(results['score'])

        # Print summary to stderr (visible to agent but not captured as score)
        print(f"--- Pendulum Summary ---", file=sys.stderr)
        print(f"Score: {results['score']:.4f} (avg over {results['episodes']} episodes)",
              file=sys.stderr)
        print(f"Avg total reward: {results['avg_reward']:.2f}",
              file=sys.stderr)
        print(f"Avg balance fraction: {results['avg_balance_fraction']:.1%} "
              f"of steps balanced", file=sys.stderr)
        print(f"Score range: {results['min_score']:.4f} - {results['max_score']:.4f} "
              f"(median: {results['median_score']:.4f})", file=sys.stderr)


if __name__ == '__main__':
    main()
