#!/usr/bin/env python3
"""
Lunar Lander Simulator — battleBOT Game Domain

A lander descends toward a landing pad at (0, 0). The agent tunes a PID-like
autopilot controller to land safely. Simplified 2D physics (no Box2D dependency).

Score = average landing quality over N episodes (higher is better, 0.0-1.0).
"""

import yaml
import sys
import random
import math
import argparse
import json
from dataclasses import dataclass

# ─── Game Constants ───────────────────────────────────────────────────────────

MAX_STEPS = 300
DEFAULT_MATCHES = 50

# Physics (tuned so episodes last ~50-150 steps with good control)
DT = 0.1           # time step
GRAVITY = -1.0      # m/s^2 (weak moon-like gravity)
MAIN_THRUST = 2.0   # m/s^2 (thrust-to-weight ratio ~2:1)
SIDE_THRUST = 0.5   # m/s^2 for horizontal
ANGULAR_THRUST = 0.3  # rad/s^2 for rotation (separate from side thrust)

INITIAL_FUEL = 100.0
MAIN_FUEL_COST = 0.5    # per step when firing
SIDE_FUEL_COST = 0.15   # per step when firing

# Landing criteria
CRASH_VY = 2.0          # m/s: vertical speed crash threshold
CRASH_VX = 1.5          # m/s: horizontal speed crash threshold
CRASH_ANGLE = 0.5       # rad: tilt crash threshold (~28 degrees)

# Starting conditions
START_Y = 20.0           # meters altitude
START_X_RANGE = 5.0      # meters: random horizontal offset
START_VX_RANGE = 1.0     # m/s: random horizontal velocity
START_VY_RANGE = 1.0     # m/s: random initial downward velocity
START_ANGLE_RANGE = 0.2  # rad: random tilt
START_AV_RANGE = 0.1     # rad/s: random angular velocity

# Wind
WIND_VX = 0.15           # m/s^2: max wind acceleration
WIND_AV = 0.02           # rad/s^2: max wind torque

# Drag
ANGULAR_DRAG = 0.95      # angular velocity multiplier per step
HORIZONTAL_DRAG = 0.998  # horizontal velocity multiplier per step


# ─── Lander State ────────────────────────────────────────────────────────────

@dataclass
class Lander:
    x: float = 0.0         # horizontal position (m), target = 0
    y: float = START_Y     # altitude (m), ground = 0
    vx: float = 0.0        # horizontal velocity (m/s)
    vy: float = 0.0        # vertical velocity (m/s), negative = falling
    angle: float = 0.0     # tilt (rad), 0 = upright
    av: float = 0.0        # angular velocity (rad/s)
    fuel: float = INITIAL_FUEL
    step: int = 0
    landed: bool = False
    crashed: bool = False

    def is_done(self) -> bool:
        return self.landed or self.crashed or self.step >= MAX_STEPS


# ─── Autopilot Controller ───────────────────────────────────────────────────

def autopilot_step(lander: Lander, cfg: dict) -> tuple:
    """
    PID-like autopilot. Returns (main_pct, side_pct, rotation_pct) where:
      main_pct:     0.0 or 1.0 (fire main engine)
      side_pct:    -1.0, 0.0, or 1.0 (left/right thruster)
      rotation_pct: -1.0 to 1.0 (torque for angle correction)
    """
    descent_rate_target = float(cfg.get('descent_rate_target', 0.15))
    horizontal_correction_gain = float(cfg.get('horizontal_correction_gain', 0.5))
    angle_correction_gain = float(cfg.get('angle_correction_gain', 1.0))
    altitude_thrust_threshold = float(cfg.get('altitude_thrust_threshold', 0.5))
    hover_gain = float(cfg.get('hover_gain', 0.8))
    fuel_reserve = float(cfg.get('fuel_reserve', 10.0))
    angular_damping = float(cfg.get('angular_damping', 1.0))
    horizontal_deadzone = float(cfg.get('horizontal_deadzone', 0.5))
    final_approach_altitude = float(cfg.get('final_approach_altitude', 0.15))
    final_approach_vy_target = float(cfg.get('final_approach_vy_target', 0.05))
    suicide_burn_threshold = float(cfg.get('suicide_burn_threshold', 0.1))

    main_fire = 0.0
    side_fire = 0.0
    rotation = 0.0

    # No thrust if fuel is below reserve
    if lander.fuel <= fuel_reserve:
        return 0.0, 0.0, 0.0

    y = lander.y
    vy = lander.vy
    vx = lander.vx
    x = lander.x
    angle = lander.angle
    av = lander.av

    # Normalize altitude to [0, 1] for threshold comparison
    y_norm = y / START_Y

    # --- Suicide burn: very low and falling fast ---
    if y_norm < suicide_burn_threshold and (-vy) > descent_rate_target * 3:
        main_fire = 1.0

    # --- Final approach: gentle touchdown ---
    elif y_norm < final_approach_altitude:
        target_vy = -final_approach_vy_target * START_Y  # convert to m/s
        if vy < target_vy:  # falling too fast
            main_fire = 1.0
        elif vy > 0:  # going up, let gravity pull
            main_fire = 0.0
        else:
            # Fine control
            speed_excess = (-vy) - final_approach_vy_target * START_Y
            if speed_excess > 0.1:
                main_fire = 1.0

    # --- Active descent control ---
    elif y_norm < altitude_thrust_threshold:
        target_descent = descent_rate_target * START_Y  # desired descent speed in m/s
        speed_excess = (-vy) - target_descent  # positive = too fast
        thrust_cmd = speed_excess * hover_gain / START_Y
        main_fire = 1.0 if thrust_cmd > 0.0 else 0.0

    # --- High altitude: coast, only brake if too fast ---
    else:
        max_coast_speed = descent_rate_target * START_Y * 1.5
        if (-vy) > max_coast_speed:
            main_fire = 1.0

    # --- Angle correction (rotation thrusters) ---
    # PD controller on angle
    angle_error = -angle
    rotation = angle_error * angle_correction_gain - av * angular_damping
    rotation = max(-1.0, min(1.0, rotation))

    # --- Horizontal correction (side thrusters) ---
    if abs(x) > horizontal_deadzone:
        h_cmd = -x * horizontal_correction_gain / START_X_RANGE
        h_cmd -= vx * horizontal_correction_gain * 0.5 / START_VX_RANGE
    else:
        h_cmd = -vx * horizontal_correction_gain * 0.3 / START_VX_RANGE

    if h_cmd > 0.3:
        side_fire = 1.0
    elif h_cmd < -0.3:
        side_fire = -1.0
    else:
        side_fire = 0.0

    return main_fire, side_fire, rotation


# ─── Physics Step ────────────────────────────────────────────────────────────

def physics_step(lander: Lander, main_fire: float, side_fire: float,
                 rotation: float, rng: random.Random):
    """Apply one step of physics to the lander."""

    # --- Forces ---
    # Gravity
    lander.vy += GRAVITY * DT

    # Main thruster (aligned with lander body axis)
    if main_fire > 0.5 and lander.fuel >= MAIN_FUEL_COST:
        # Thrust direction depends on lander angle
        thrust_x = -math.sin(lander.angle) * MAIN_THRUST * DT
        thrust_y = math.cos(lander.angle) * MAIN_THRUST * DT
        lander.vx += thrust_x
        lander.vy += thrust_y
        lander.fuel -= MAIN_FUEL_COST

    # Side thruster (horizontal, body-frame)
    if abs(side_fire) > 0.1 and lander.fuel >= SIDE_FUEL_COST:
        direction = 1.0 if side_fire > 0 else -1.0
        lander.vx += direction * SIDE_THRUST * DT * math.cos(lander.angle)
        lander.vy += direction * SIDE_THRUST * DT * math.sin(lander.angle)
        lander.fuel -= SIDE_FUEL_COST

    # Rotation (angular thruster, separate from side thruster)
    if abs(rotation) > 0.1 and lander.fuel >= SIDE_FUEL_COST * 0.5:
        lander.av += rotation * ANGULAR_THRUST * DT
        lander.fuel -= SIDE_FUEL_COST * 0.5

    # Wind (random perturbation)
    lander.vx += rng.uniform(-WIND_VX, WIND_VX) * DT
    lander.av += rng.uniform(-WIND_AV, WIND_AV) * DT

    # --- Integration ---
    lander.x += lander.vx * DT
    lander.y += lander.vy * DT
    lander.angle += lander.av * DT

    # --- Drag ---
    lander.av *= ANGULAR_DRAG
    lander.vx *= HORIZONTAL_DRAG

    lander.step += 1

    # --- Ground check ---
    if lander.y <= 0.0:
        lander.y = 0.0
        if (abs(lander.vy) > CRASH_VY or
            abs(lander.vx) > CRASH_VX or
            abs(lander.angle) > CRASH_ANGLE):
            lander.crashed = True
        else:
            lander.landed = True


# ─── Episode Scoring ─────────────────────────────────────────────────────────

def score_episode(lander: Lander) -> float:
    """Score a completed episode. Returns 0.0-1.0."""
    if lander.crashed:
        return 0.0
    if not lander.landed:
        return 0.0  # timed out

    # Safe landing — score based on precision
    # Each component measures how close to perfect (0 penalty = perfect)
    x_penalty = 0.25 * min(1.0, abs(lander.x) / 3.0)
    vy_penalty = 0.30 * min(1.0, abs(lander.vy) / CRASH_VY)
    vx_penalty = 0.15 * min(1.0, abs(lander.vx) / CRASH_VX)
    angle_penalty = 0.15 * min(1.0, abs(lander.angle) / CRASH_ANGLE)
    fuel_penalty = 0.15 * min(1.0, (INITIAL_FUEL - lander.fuel) / INITIAL_FUEL)

    score = 1.0 - x_penalty - vy_penalty - vx_penalty - angle_penalty - fuel_penalty
    return max(0.0, min(1.0, score))


# ─── Episode Runner ──────────────────────────────────────────────────────────

def run_episode(cfg: dict, seed: int, verbose: bool = False) -> dict:
    """Run a single landing episode. Returns result dict."""
    rng = random.Random(seed)

    lander = Lander(
        x=rng.uniform(-START_X_RANGE, START_X_RANGE),
        y=START_Y,
        vx=rng.uniform(-START_VX_RANGE, START_VX_RANGE),
        vy=rng.uniform(-START_VY_RANGE, 0),
        angle=rng.uniform(-START_ANGLE_RANGE, START_ANGLE_RANGE),
        av=rng.uniform(-START_AV_RANGE, START_AV_RANGE),
        fuel=INITIAL_FUEL,
    )

    log = []

    while not lander.is_done():
        main_fire, side_fire, rotation = autopilot_step(lander, cfg)
        physics_step(lander, main_fire, side_fire, rotation, rng)

        if verbose:
            log.append({
                'step': lander.step,
                'x': round(lander.x, 3),
                'y': round(lander.y, 3),
                'vx': round(lander.vx, 3),
                'vy': round(lander.vy, 3),
                'angle': round(lander.angle, 4),
                'av': round(lander.av, 4),
                'fuel': round(lander.fuel, 1),
                'main': main_fire > 0.5,
                'side': round(side_fire, 1),
                'rot': round(rotation, 2),
            })

    ep_score = score_episode(lander)

    return {
        'score': round(ep_score, 4),
        'landed': lander.landed,
        'crashed': lander.crashed,
        'timeout': not lander.landed and not lander.crashed,
        'steps': lander.step,
        'final_x': round(lander.x, 3),
        'final_vy': round(lander.vy, 3),
        'final_vx': round(lander.vx, 3),
        'final_angle': round(lander.angle, 4),
        'fuel_remaining': round(lander.fuel, 1),
        'log': log if verbose else [],
    }


# ─── Tournament Runner ──────────────────────────────────────────────────────

def run_tournament(cfg: dict, n_episodes: int = DEFAULT_MATCHES,
                   base_seed: int = 42, verbose: bool = False) -> dict:
    """Run N episodes and return aggregate stats."""
    total_score = 0.0
    landings = 0
    crashes = 0
    timeouts = 0
    total_steps = 0
    total_fuel_used = 0.0
    episode_results = []

    for i in range(n_episodes):
        result = run_episode(cfg, base_seed + i, verbose)
        total_score += result['score']
        if result['landed']:
            landings += 1
        elif result['crashed']:
            crashes += 1
        else:
            timeouts += 1
        total_steps += result['steps']
        total_fuel_used += INITIAL_FUEL - result['fuel_remaining']
        episode_results.append(result)

    avg_score = total_score / n_episodes

    return {
        'score': round(avg_score, 4),
        'landings': landings,
        'crashes': crashes,
        'timeouts': timeouts,
        'episodes': n_episodes,
        'landing_rate': round(landings / n_episodes, 4),
        'avg_steps': round(total_steps / n_episodes, 1),
        'avg_fuel_used': round(total_fuel_used / n_episodes, 1),
        'results': episode_results if verbose else [],
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Lunar Lander Simulator')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to autopilot config YAML')
    parser.add_argument('--matches', '-n', type=int, default=DEFAULT_MATCHES,
                        help=f'Number of episodes (default: {DEFAULT_MATCHES})')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Include per-step logs in output')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output full results as JSON')
    args = parser.parse_args()

    # Load autopilot config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_tournament(cfg, args.matches, args.seed, args.verbose)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Print just the score (what the harness reads)
        print(results['score'])

        # Print summary to stderr (visible to agent but not captured as score)
        print(f"--- Landing Summary ---", file=sys.stderr)
        print(f"Score: {results['score']:.4f}", file=sys.stderr)
        print(f"Landings: {results['landings']}/{results['episodes']} "
              f"({results['landing_rate']:.1%})", file=sys.stderr)
        print(f"Crashes: {results['crashes']}, Timeouts: {results['timeouts']}",
              file=sys.stderr)
        print(f"Avg steps: {results['avg_steps']}, "
              f"Avg fuel used: {results['avg_fuel_used']:.1f}",
              file=sys.stderr)


if __name__ == '__main__':
    main()
