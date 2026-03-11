#!/usr/bin/env python3
"""
CartPole Control Simulator — battleBOT Game Domain

A pole balanced on a cart. The agent configures a parameterized controller
(via config.yaml) that decides whether to push left or right each timestep.
The episode ends when the pole falls (angle > 12°), the cart goes off-screen
(position > ±2.4), or 500 steps are reached (optimal).

Score = average survival steps / 500 across N episodes (higher is better).

Physics: Standard CartPole-v1 from Gymnasium (Euler integration).
"""

import yaml
import sys
import math
import argparse
import json
from dataclasses import dataclass

# ─── Physics Constants ───────────────────────────────────────────────────────

GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = CART_MASS + POLE_MASS
POLE_LENGTH = 0.5  # half-length
POLE_MASS_LENGTH = POLE_MASS * POLE_LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # timestep (seconds)

# ─── Episode Limits ──────────────────────────────────────────────────────────

MAX_STEPS = 500
ANGLE_LIMIT = 12.0 * math.pi / 180.0  # 12 degrees in radians
POSITION_LIMIT = 2.4

DEFAULT_MATCHES = 50

# ─── Controller ──────────────────────────────────────────────────────────────

@dataclass
class Controller:
    """Parameterized controller configured by the agent via config.yaml."""
    angle_weight: float
    angular_velocity_weight: float
    position_weight: float
    velocity_weight: float
    angle_bias: float
    response_sharpness: float
    anticipation_horizon: float
    position_centering: float

    def decide(self, x: float, x_dot: float, theta: float, theta_dot: float) -> int:
        """
        Returns 0 (push left) or 1 (push right).

        Computes a weighted signal from the state, adds anticipation of future
        angle, applies position centering bias, then passes through a sigmoid.
        """
        # Core weighted sum
        signal = (
            theta * self.angle_weight
            + theta_dot * self.angular_velocity_weight
            + x * self.position_weight
            + x_dot * self.velocity_weight
            + self.angle_bias
        )

        # Anticipation: predict future angle contribution
        predicted_theta = theta + theta_dot * TAU * self.anticipation_horizon
        signal += predicted_theta * self.anticipation_horizon

        # Position centering: bias toward pushing cart back to center
        # If cart is right of center (x > 0), add negative bias (push left)
        # If cart is left of center (x < 0), add positive bias (push right)
        signal -= x * self.position_centering

        # Sigmoid decision
        z = signal * self.response_sharpness
        # Clip to avoid overflow
        z = max(-20.0, min(20.0, z))
        prob = 1.0 / (1.0 + math.exp(-z))

        return 1 if prob >= 0.5 else 0


# ─── Physics Step ────────────────────────────────────────────────────────────

def cartpole_step(x: float, x_dot: float, theta: float, theta_dot: float,
                  action: int):
    """
    Standard CartPole Euler integration (matches Gymnasium CartPole-v1).
    Returns (x, x_dot, theta, theta_dot).
    """
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Equations of motion
    temp = (force + POLE_MASS_LENGTH * theta_dot ** 2 * sin_theta) / TOTAL_MASS
    theta_acc = (
        (GRAVITY * sin_theta - cos_theta * temp)
        / (POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_theta ** 2 / TOTAL_MASS))
    )
    x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

    # Euler integration
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * x_acc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * theta_acc

    return x, x_dot, theta, theta_dot


# ─── Episode Simulation ─────────────────────────────────────────────────────

def run_episode(controller: Controller, seed: int, verbose: bool = False) -> dict:
    """
    Run a single CartPole episode. Initial state has small random perturbations
    seeded by `seed` (matching Gymnasium's reset behavior: uniform in [-0.05, 0.05]).
    """
    import random
    rng = random.Random(seed)

    # Initial state: small random perturbations (same as Gymnasium)
    x = rng.uniform(-0.05, 0.05)
    x_dot = rng.uniform(-0.05, 0.05)
    theta = rng.uniform(-0.05, 0.05)
    theta_dot = rng.uniform(-0.05, 0.05)

    log = []
    steps = 0

    for step in range(MAX_STEPS):
        action = controller.decide(x, x_dot, theta, theta_dot)

        if verbose:
            log.append({
                'step': step,
                'x': round(x, 4),
                'x_dot': round(x_dot, 4),
                'theta': round(theta, 4),
                'theta_dot': round(theta_dot, 4),
                'angle_deg': round(math.degrees(theta), 2),
                'action': 'right' if action == 1 else 'left',
            })

        x, x_dot, theta, theta_dot = cartpole_step(x, x_dot, theta, theta_dot, action)
        steps += 1

        # Check termination conditions
        if abs(x) > POSITION_LIMIT:
            break
        if abs(theta) > ANGLE_LIMIT:
            break

    return {
        'steps': steps,
        'normalized': round(steps / MAX_STEPS, 4),
        'terminated_by': (
            'position' if abs(x) > POSITION_LIMIT
            else 'angle' if abs(theta) > ANGLE_LIMIT
            else 'success'
        ),
        'final_x': round(x, 4),
        'final_angle_deg': round(math.degrees(theta), 2),
        'log': log if verbose else [],
    }


# ─── Config Loading & Validation ────────────────────────────────────────────

PARAM_RANGES = {
    'angle_weight':             (0.0, 10.0),
    'angular_velocity_weight':  (0.0, 10.0),
    'position_weight':          (0.0, 5.0),
    'velocity_weight':          (0.0, 5.0),
    'angle_bias':               (-2.0, 2.0),
    'response_sharpness':       (0.1, 20.0),
    'anticipation_horizon':     (0.0, 2.0),
    'position_centering':       (0.0, 5.0),
}

DEFAULTS = {
    'angle_weight': 0.0,
    'angular_velocity_weight': 1.0,
    'position_weight': 0.0,
    'velocity_weight': 0.0,
    'angle_bias': 0.0,
    'response_sharpness': 1.0,
    'anticipation_horizon': 0.0,
    'position_centering': 0.0,
}


def validate_config(cfg: dict):
    """Validate config values are within allowed ranges. Returns (ok, msg)."""
    for name, (lo, hi) in PARAM_RANGES.items():
        val = float(cfg.get(name, DEFAULTS[name]))
        if val < lo or val > hi:
            return False, f"{name}={val} out of range [{lo}, {hi}]"
    return True, "OK"


def make_controller(cfg: dict) -> Controller:
    """Build a Controller from a config dict."""
    return Controller(
        angle_weight=float(cfg.get('angle_weight', DEFAULTS['angle_weight'])),
        angular_velocity_weight=float(cfg.get('angular_velocity_weight', DEFAULTS['angular_velocity_weight'])),
        position_weight=float(cfg.get('position_weight', DEFAULTS['position_weight'])),
        velocity_weight=float(cfg.get('velocity_weight', DEFAULTS['velocity_weight'])),
        angle_bias=float(cfg.get('angle_bias', DEFAULTS['angle_bias'])),
        response_sharpness=float(cfg.get('response_sharpness', DEFAULTS['response_sharpness'])),
        anticipation_horizon=float(cfg.get('anticipation_horizon', DEFAULTS['anticipation_horizon'])),
        position_centering=float(cfg.get('position_centering', DEFAULTS['position_centering'])),
    )


# ─── Tournament ──────────────────────────────────────────────────────────────

def run_tournament(cfg: dict, n_episodes: int = DEFAULT_MATCHES,
                   base_seed: int = 42, verbose: bool = False) -> dict:
    """Run N episodes and return aggregate stats."""
    valid, msg = validate_config(cfg)
    if not valid:
        return {'error': msg, 'score': 0.0}

    controller = make_controller(cfg)

    total_steps = 0
    successes = 0  # episodes that reached 500 steps
    episode_results = []
    termination_counts = {'angle': 0, 'position': 0, 'success': 0}

    for i in range(n_episodes):
        result = run_episode(controller, base_seed + i, verbose)
        total_steps += result['steps']
        termination_counts[result['terminated_by']] += 1
        if result['steps'] == MAX_STEPS:
            successes += 1
        episode_results.append(result)

    avg_steps = total_steps / n_episodes
    score = avg_steps / MAX_STEPS

    return {
        'score': round(score, 4),
        'avg_steps': round(avg_steps, 1),
        'max_steps': max(r['steps'] for r in episode_results),
        'min_steps': min(r['steps'] for r in episode_results),
        'perfect_episodes': successes,
        'episodes': n_episodes,
        'terminated_by_angle': termination_counts['angle'],
        'terminated_by_position': termination_counts['position'],
        'terminated_by_success': termination_counts['success'],
        'results': episode_results if verbose else [],
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CartPole Control Simulator')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to controller config YAML')
    parser.add_argument('--matches', '-n', type=int, default=DEFAULT_MATCHES,
                        help=f'Number of episodes (default: {DEFAULT_MATCHES})')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Include per-step logs')
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
        # Print just the score (what the harness reads)
        print(results['score'])

        # Print summary to stderr (visible to agent but not captured as score)
        print(f"--- CartPole Summary ---", file=sys.stderr)
        print(f"Score: {results['score']:.4f} (avg {results['avg_steps']} / {MAX_STEPS} steps)",
              file=sys.stderr)
        print(f"Perfect episodes: {results['perfect_episodes']}/{results['episodes']}",
              file=sys.stderr)
        print(f"Steps range: {results['min_steps']}–{results['max_steps']}",
              file=sys.stderr)
        print(f"Terminations — angle: {results['terminated_by_angle']}, "
              f"position: {results['terminated_by_position']}, "
              f"success: {results['terminated_by_success']}", file=sys.stderr)


if __name__ == '__main__':
    main()
