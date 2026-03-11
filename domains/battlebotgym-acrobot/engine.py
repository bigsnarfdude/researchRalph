#!/usr/bin/env python3
"""
Acrobot Swing-Up Simulator — battleBOT Game Domain

Two-link pendulum (like a gymnast on a bar). The first link is attached
to a fixed point. Only the joint between the two links is actuated.
Goal: swing the tip of the second link above a target height.

The agent tunes an energy-based swing-up controller via config.yaml.
Score = average(1.0 - steps_to_goal / 500) across episodes.
"""

import yaml
import sys
import math
import argparse
import json
from dataclasses import dataclass
from typing import Tuple

# ─── Physics Constants ───────────────────────────────────────────────────────

LINK_LENGTH_1 = 1.0
LINK_LENGTH_2 = 1.0
LINK_MASS_1 = 1.0
LINK_MASS_2 = 1.0
LINK_COM_1 = 0.5    # center of mass at midpoint
LINK_COM_2 = 0.5
LINK_MOI = 1.0       # moment of inertia for both links
GRAVITY = 9.8
DT = 0.2
MAX_STEPS = 500
DEFAULT_EPISODES = 50

# ─── Acrobot Dynamics ────────────────────────────────────────────────────────

def acrobot_derivs(state: Tuple[float, float, float, float],
                   torque: float) -> Tuple[float, float, float, float]:
    """
    Compute derivatives of the Acrobot state.
    Standard equations of motion from Sutton & Barto / Spong.

    state = (theta1, theta2, dtheta1, dtheta2)
    Returns (dtheta1, dtheta2, ddtheta1, ddtheta2)
    """
    theta1, theta2, dtheta1, dtheta2 = state

    m1 = LINK_MASS_1
    m2 = LINK_MASS_2
    l1 = LINK_LENGTH_1
    lc1 = LINK_COM_1
    lc2 = LINK_COM_2
    I1 = LINK_MOI
    I2 = LINK_MOI
    g = GRAVITY

    d1 = (m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * math.cos(theta2))
          + I1 + I2)
    d2 = m2 * (lc2**2 + l1 * lc2 * math.cos(theta2)) + I2

    phi2 = m2 * lc2 * g * math.cos(theta1 + theta2 - math.pi / 2.0)
    phi1 = (-m2 * l1 * lc2 * dtheta2**2 * math.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * math.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * math.cos(theta1 - math.pi / 2.0)
            + phi2)

    # Mass matrix: [[d1, d2], [d2, I2+m2*lc2^2]]
    # [d1  d2] [ddtheta1]   [phi1 - torque]     (torque only on joint 2)
    # [d2  d3] [ddtheta2] = [phi2 + torque]
    d3 = m2 * lc2**2 + I2

    # Solve 2x2 system
    det = d1 * d3 - d2 * d2
    if abs(det) < 1e-12:
        det = 1e-12  # prevent division by zero

    ddtheta2 = (d1 * (torque - phi2) + d2 * phi1) / det
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1 if abs(d1) > 1e-12 else 0.0

    return (dtheta1, dtheta2, ddtheta1, ddtheta2)


def rk4_step(state: Tuple[float, float, float, float],
             torque: float, dt: float) -> Tuple[float, float, float, float]:
    """Fourth-order Runge-Kutta integration step."""
    s = state

    k1 = acrobot_derivs(s, torque)
    s2 = tuple(s[i] + 0.5 * dt * k1[i] for i in range(4))
    k2 = acrobot_derivs(s2, torque)
    s3 = tuple(s[i] + 0.5 * dt * k2[i] for i in range(4))
    k3 = acrobot_derivs(s3, torque)
    s4 = tuple(s[i] + dt * k3[i] for i in range(4))
    k4 = acrobot_derivs(s4, torque)

    new_state = tuple(
        s[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        for i in range(4)
    )
    return new_state


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return ((angle + math.pi) % (2 * math.pi)) - math.pi


def tip_height(theta1: float, theta2: float) -> float:
    """
    Height of the tip of link 2 relative to the fixed point.
    Returns -cos(theta1) - cos(theta1 + theta2).
    Goal: this > 1.0 (i.e., tip above fixed_point + link_length_1).
    """
    return -math.cos(theta1) - math.cos(theta1 + theta2)


def total_energy(state: Tuple[float, float, float, float]) -> float:
    """
    Compute total mechanical energy of the acrobot system.
    Potential + Kinetic energy.
    """
    theta1, theta2, dtheta1, dtheta2 = state

    m1 = LINK_MASS_1
    m2 = LINK_MASS_2
    l1 = LINK_LENGTH_1
    lc1 = LINK_COM_1
    lc2 = LINK_COM_2
    I1 = LINK_MOI
    I2 = LINK_MOI
    g = GRAVITY

    # Heights of centers of mass (relative to fixed point, positive = up)
    y1 = -lc1 * math.cos(theta1)
    y2 = -l1 * math.cos(theta1) - lc2 * math.cos(theta1 + theta2)

    # Potential energy
    PE = m1 * g * y1 + m2 * g * y2

    # Kinetic energy (using the mass matrix)
    d1 = (m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * math.cos(theta2))
          + I1 + I2)
    d2 = m2 * (lc2**2 + l1 * lc2 * math.cos(theta2)) + I2
    d3 = m2 * lc2**2 + I2

    KE = 0.5 * (d1 * dtheta1**2 + 2 * d2 * dtheta1 * dtheta2 + d3 * dtheta2**2)

    return PE + KE


# ─── Controller ──────────────────────────────────────────────────────────────

def controller(state: Tuple[float, float, float, float], cfg: dict) -> float:
    """
    Energy-based swing-up controller.

    Phase 1 (energy pump): If total energy is below energy_target, pump energy
    into the system using torque = sign(dtheta2 * cos(theta1 + phase_offset))
    scaled by energy_pump_gain.

    Phase 2 (catch): When the tip is close to the goal height, switch to a
    linear controller using theta/dtheta weights to stabilize.

    sign_strategy blends between bang-bang (±1) and proportional control.
    """
    theta1, theta2, dtheta1, dtheta2 = state

    energy_pump_gain = float(cfg.get('energy_pump_gain', 2.0))
    phase_offset = float(cfg.get('phase_offset', 0.0))
    velocity_threshold = float(cfg.get('velocity_threshold', 2.0))
    theta1_weight = float(cfg.get('theta1_weight', 1.0))
    theta2_weight = float(cfg.get('theta2_weight', 1.0))
    dtheta1_weight = float(cfg.get('dtheta1_weight', 1.0))
    dtheta2_weight = float(cfg.get('dtheta2_weight', 1.0))
    energy_target = float(cfg.get('energy_target', 10.0))
    sign_strategy = float(cfg.get('sign_strategy', 0.8))
    catch_threshold = float(cfg.get('catch_threshold', 0.5))
    catch_gain = float(cfg.get('catch_gain', 2.0))

    h = tip_height(theta1, theta2)
    E = total_energy(state)

    # Phase 2: Catch mode — tip is near goal height
    if h > 1.0 - catch_threshold:
        # Linear controller to push tip above threshold
        raw = (theta1_weight * wrap_angle(theta1 - math.pi)
               + theta2_weight * wrap_angle(theta2)
               + dtheta1_weight * dtheta1
               + dtheta2_weight * dtheta2)
        torque = -catch_gain * raw
    else:
        # Phase 1: Energy pumping
        pump_signal = dtheta2 * math.cos(theta1 + phase_offset)

        # Blend between bang-bang and proportional
        if sign_strategy > 0.5:
            # More bang-bang
            if abs(pump_signal) > 1e-6:
                torque = math.copysign(1.0, pump_signal) * energy_pump_gain
            else:
                torque = 0.0
        else:
            # More proportional
            torque = pump_signal * energy_pump_gain

        # Reduce pumping if energy is above target
        if E > energy_target:
            torque *= max(0.1, 1.0 - (E - energy_target) / energy_target)

    # Clamp torque to allowed range: -1, 0, or +1
    if torque > 0.3:
        return 1.0
    elif torque < -0.3:
        return -1.0
    else:
        return 0.0


# ─── Episode Simulation ─────────────────────────────────────────────────────

def make_initial_state(seed: int) -> Tuple[float, float, float, float]:
    """
    Start with both links hanging down plus a small random perturbation.
    theta1=theta2=0 means hanging straight down.
    """
    import random
    rng = random.Random(seed)
    theta1 = rng.uniform(-0.1, 0.1)
    theta2 = rng.uniform(-0.1, 0.1)
    dtheta1 = rng.uniform(-0.1, 0.1)
    dtheta2 = rng.uniform(-0.1, 0.1)
    return (theta1, theta2, dtheta1, dtheta2)


def run_episode(cfg: dict, seed: int, verbose: bool = False) -> dict:
    """Run one episode. Returns result dict."""
    state = make_initial_state(seed)
    log = []

    steps_to_goal = None

    for step in range(MAX_STEPS):
        theta1, theta2, dtheta1, dtheta2 = state

        # Check goal: tip of link 2 above fixed_point + link_length_1
        h = tip_height(theta1, theta2)
        if h > 1.0:
            steps_to_goal = step
            if verbose:
                log.append({
                    'step': step,
                    'theta1': round(theta1, 4),
                    'theta2': round(theta2, 4),
                    'dtheta1': round(dtheta1, 4),
                    'dtheta2': round(dtheta2, 4),
                    'tip_height': round(h, 4),
                    'energy': round(total_energy(state), 4),
                    'action': 0.0,
                    'event': 'GOAL_REACHED',
                })
            break

        # Get action from controller
        torque = controller(state, cfg)

        if verbose:
            log.append({
                'step': step,
                'theta1': round(theta1, 4),
                'theta2': round(theta2, 4),
                'dtheta1': round(dtheta1, 4),
                'dtheta2': round(dtheta2, 4),
                'tip_height': round(h, 4),
                'energy': round(total_energy(state), 4),
                'action': torque,
            })

        # Integrate physics
        state = rk4_step(state, torque, DT)

        # Clamp angular velocities (as in gym Acrobot)
        theta1_new, theta2_new, dtheta1_new, dtheta2_new = state
        dtheta1_new = max(-4 * math.pi, min(4 * math.pi, dtheta1_new))
        dtheta2_new = max(-9 * math.pi, min(9 * math.pi, dtheta2_new))
        state = (theta1_new, theta2_new, dtheta1_new, dtheta2_new)

    # Compute score for this episode
    if steps_to_goal is not None:
        score = max(0.0, 1.0 - steps_to_goal / MAX_STEPS)
    else:
        score = 0.0

    return {
        'score': round(score, 6),
        'steps_to_goal': steps_to_goal,
        'max_steps': MAX_STEPS,
        'final_tip_height': round(tip_height(state[0], state[1]), 4),
        'final_energy': round(total_energy(state), 4),
        'log': log if verbose else [],
    }


def run_experiment(cfg: dict, n_episodes: int = DEFAULT_EPISODES,
                   base_seed: int = 42, verbose: bool = False) -> dict:
    """Run N episodes and return aggregate stats."""
    total_score = 0.0
    successes = 0
    total_steps_on_success = 0
    episode_results = []

    for i in range(n_episodes):
        result = run_episode(cfg, base_seed + i, verbose)
        total_score += result['score']
        if result['steps_to_goal'] is not None:
            successes += 1
            total_steps_on_success += result['steps_to_goal']
        episode_results.append(result)

    avg_score = total_score / n_episodes
    success_rate = successes / n_episodes
    avg_steps = (total_steps_on_success / successes) if successes > 0 else MAX_STEPS

    return {
        'score': round(avg_score, 4),
        'success_rate': round(success_rate, 4),
        'avg_steps_to_goal': round(avg_steps, 1),
        'successes': successes,
        'episodes': n_episodes,
        'failures': n_episodes - successes,
        'results': episode_results if verbose else [],
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Acrobot Swing-Up Simulator')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to controller config YAML')
    parser.add_argument('--matches', '-n', type=int, default=DEFAULT_EPISODES,
                        help=f'Number of episodes (default: {DEFAULT_EPISODES})')
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

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Print just the score (what the harness reads)
        print(results['score'])

        # Print summary to stderr (visible to agent but not captured as score)
        print(f"--- Acrobot Summary ---", file=sys.stderr)
        print(f"Score: {results['score']:.4f}", file=sys.stderr)
        print(f"Success rate: {results['success_rate']:.1%} "
              f"({results['successes']}/{results['episodes']})",
              file=sys.stderr)
        print(f"Avg steps to goal (on success): {results['avg_steps_to_goal']}",
              file=sys.stderr)
        print(f"Failures: {results['failures']}", file=sys.stderr)


if __name__ == '__main__':
    main()
