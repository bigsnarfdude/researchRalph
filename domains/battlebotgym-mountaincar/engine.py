#!/usr/bin/env python3
"""
MountainCar Simulator — battleBOT Game Domain

A car in a valley must build momentum by swinging back and forth to reach
a flag at position 0.5 on the right hill. The engine is too weak to climb
directly. An agent configures a momentum-building policy via config.yaml.

Score = fraction of episodes where the car reaches the goal (higher is better).
"""

import yaml
import sys
import math
import argparse
import json
from dataclasses import dataclass

# --- Physics Constants (fixed, not configurable) ---

FORCE = 0.001
GRAVITY = 0.0025
POS_MIN = -1.2
POS_MAX = 0.6
VEL_MIN = -0.07
VEL_MAX = 0.07
GOAL_POS = 0.5
MAX_STEPS = 200
DEFAULT_MATCHES = 50

# --- Data Types ---

@dataclass
class CarState:
    position: float
    velocity: float
    step: int = 0

    def is_done(self) -> bool:
        return self.position >= GOAL_POS or self.step >= MAX_STEPS

    def reached_goal(self) -> bool:
        return self.position >= GOAL_POS


def step_physics(state: CarState, action: int) -> CarState:
    """
    Advance one timestep. Action: 0=push left, 1=no push, 2=push right.
    Standard MountainCar-v0 physics.
    """
    velocity = state.velocity + (action - 1) * FORCE - math.cos(3.0 * state.position) * GRAVITY
    velocity = max(VEL_MIN, min(VEL_MAX, velocity))

    position = state.position + velocity
    position = max(POS_MIN, min(POS_MAX, position))

    # Left wall: if hit the left boundary, kill velocity
    if position <= POS_MIN and velocity < 0:
        velocity = 0.0

    return CarState(position=position, velocity=velocity, step=state.step + 1)


# --- Policy (driven by config.yaml) ---

def compute_energy(position: float, velocity: float) -> float:
    """
    Estimate total mechanical energy of the car, normalized to [0, 1].
    Potential energy from height + kinetic energy, scaled so that
    the energy needed to reach the goal from the valley bottom ~ 1.0.
    """
    height = math.sin(3.0 * position)
    kinetic = 0.5 * velocity * velocity
    potential = GRAVITY * (height + 1.0)  # shift so minimum is ~0
    raw_energy = potential + kinetic
    # Normalize: energy at goal (pos=0.5, vel=0) is ~0.005
    # Scale so that value is approximately 1.0 at the goal
    return raw_energy / 0.005


def policy_action(state: CarState, cfg: dict) -> int:
    """
    Compute action from the policy parameters in config.
    Returns 0 (left), 1 (coast), or 2 (right).

    The policy is a velocity-following momentum builder:
    - Core idea: push in the direction the car is already moving (builds energy)
    - When the car has enough energy and is positioned well, commit to going right
    - Parameters control thresholds, sensitivity, and biases
    """
    pos = state.position
    vel = state.velocity
    momentum_threshold = float(cfg.get('momentum_threshold', 0.0))
    switch_point = float(cfg.get('position_switch_point', -0.5))
    aggression = float(cfg.get('swing_aggression', 0.5))
    peak_weight = float(cfg.get('peak_detection_weight', 1.0))
    vel_gain = float(cfg.get('velocity_gain', 1.0))
    pos_gain = float(cfg.get('position_gain', 1.0))
    energy_thresh = float(cfg.get('energy_threshold', 0.5))
    coast_width = float(cfg.get('coast_zone_width', 0.01))
    right_bias = float(cfg.get('right_bias', 0.0))

    energy = compute_energy(pos, vel)

    # Layer 1: Goal commit mode
    # When we have enough energy AND are above the switch point moving right, go for it
    if energy > energy_thresh and pos > switch_point and vel > momentum_threshold:
        return 2  # push right toward goal

    # Layer 2: Velocity-following momentum builder
    # The decision signal combines:
    # - velocity direction (the main driver of momentum building)
    # - position relative to switch point (bias toward right when above switch)
    # - constant right_bias
    signal = vel_gain * vel + pos_gain * 0.01 * (pos - switch_point) + right_bias * 0.01

    # Coast zone: only coast when the signal is extremely weak
    # coast_width sets the base dead zone, peak_detection_weight widens it at peaks
    # aggression shrinks the coast zone (high aggression = almost never coast)
    coast_threshold = coast_width * 0.1 * (1.0 - aggression * 0.8) + peak_weight * 0.0001
    if abs(signal) < coast_threshold:
        return 1  # coast

    # Push in the direction of the signal
    if signal > 0:
        return 2  # push right
    else:
        return 0  # push left


# --- Config Validation ---

PARAM_RANGES = {
    'momentum_threshold':    (-0.07, 0.07),
    'position_switch_point': (-1.2, 0.0),
    'swing_aggression':      (0.0, 1.0),
    'peak_detection_weight': (0.0, 5.0),
    'velocity_gain':         (0.0, 10.0),
    'position_gain':         (0.0, 10.0),
    'energy_threshold':      (0.0, 1.0),
    'coast_zone_width':      (0.0, 0.5),
    'right_bias':            (-1.0, 1.0),
}


def validate_config(cfg: dict):
    """Check all config params are within allowed ranges."""
    for name, (lo, hi) in PARAM_RANGES.items():
        val = float(cfg.get(name, 0))
        if val < lo or val > hi:
            return False, f"{name}={val} out of range [{lo}, {hi}]"
    return True, "OK"


# --- Episode Simulation ---

def run_episode(cfg: dict, seed: int) -> dict:
    """
    Run one episode of MountainCar with the given policy config.
    Start position is deterministic based on seed (uniform in [-0.6, -0.4]).
    """
    import random
    rng = random.Random(seed)
    start_pos = -0.6 + rng.random() * 0.2  # uniform in [-0.6, -0.4]

    state = CarState(position=start_pos, velocity=0.0)
    trajectory = []

    while not state.is_done():
        action = policy_action(state, cfg)
        trajectory.append({
            'step': state.step,
            'position': round(state.position, 5),
            'velocity': round(state.velocity, 5),
            'action': action,
        })
        state = step_physics(state, action)

    return {
        'reached_goal': state.reached_goal(),
        'steps': state.step,
        'final_position': round(state.position, 5),
        'final_velocity': round(state.velocity, 5),
        'trajectory': trajectory,
    }


# --- Tournament ---

def run_tournament(cfg: dict, n_matches: int = DEFAULT_MATCHES,
                   base_seed: int = 42, verbose: bool = False) -> dict:
    """Run N episodes and return aggregate stats."""
    valid, msg = validate_config(cfg)
    if not valid:
        return {'error': msg, 'score': 0.0}

    successes = 0
    total_steps = 0
    episode_results = []

    for i in range(n_matches):
        result = run_episode(cfg, base_seed + i)
        if result['reached_goal']:
            successes += 1
        total_steps += result['steps']
        if verbose:
            episode_results.append(result)
        else:
            episode_results.append({
                'reached_goal': result['reached_goal'],
                'steps': result['steps'],
                'final_position': result['final_position'],
            })

    score = successes / n_matches
    avg_steps = total_steps / n_matches

    return {
        'score': round(score, 4),
        'successes': successes,
        'failures': n_matches - successes,
        'matches': n_matches,
        'avg_steps': round(avg_steps, 1),
        'results': episode_results if verbose else [],
    }


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description='MountainCar Simulator')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to policy config YAML')
    parser.add_argument('--matches', '-n', type=int, default=DEFAULT_MATCHES,
                        help=f'Number of episodes (default: {DEFAULT_MATCHES})')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Include per-episode details')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output full results as JSON')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_tournament(cfg, args.matches, args.seed, args.verbose)

    if 'error' in results:
        print(f"INVALID CONFIG: {results['error']}", file=sys.stderr)
        print("0.0")
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Print just the score to stdout
        print(results['score'])

        # Print summary to stderr
        print(f"--- MountainCar Summary ---", file=sys.stderr)
        print(f"Goal rate: {results['score']:.1%} ({results['successes']} reached / "
              f"{results['matches']} episodes)", file=sys.stderr)
        print(f"Avg steps: {results['avg_steps']}", file=sys.stderr)
        print(f"Failures (timeout): {results['failures']}", file=sys.stderr)


if __name__ == '__main__':
    main()
