#!/usr/bin/env python3
"""
Arena Combat Simulator — researchRalph Game Domain

Two bots fight in a grid arena. Each bot has HP, energy, and configurable
behavior weights that determine their strategy. The "challenger" bot's
config is what agents optimize; the "champion" bot uses a fixed baseline.

Score = win rate over N matches (higher is better).
"""

import yaml
import sys
import random
import math
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─── Game Constants ───────────────────────────────────────────────────────────

GRID_SIZE = 16
MAX_TURNS = 200
DEFAULT_MATCHES = 50

# ─── Data Types ───────────────────────────────────────────────────────────────

@dataclass
class Bot:
    name: str
    x: int
    y: int
    hp: float
    max_hp: float
    energy: float
    max_energy: float
    attack_power: float
    defense: float
    speed: int  # tiles per turn
    heal_amount: float
    # Strategy weights (these are what agents optimize)
    aggression: float       # 0-1: tendency to move toward enemy
    retreat_threshold: float  # 0-1: HP% below which bot retreats
    heal_threshold: float    # 0-1: HP% below which bot heals (if has energy)
    attack_range: int        # tiles: how far attacks reach
    energy_regen: float      # energy gained per turn
    heal_cost: float         # energy spent to heal
    attack_cost: float       # energy spent to attack
    special_cooldown: int    # turns between special attacks
    special_power: float     # damage multiplier for special
    kite_distance: int       # preferred distance when kiting
    burst_threshold: float   # enemy HP% below which bot goes all-in
    # Runtime state
    special_timer: int = 0
    total_damage_dealt: float = 0
    total_damage_taken: float = 0
    turns_alive: int = 0

    def is_alive(self) -> bool:
        return self.hp > 0

    def hp_pct(self) -> float:
        return self.hp / self.max_hp

    def energy_pct(self) -> float:
        return self.energy / self.max_energy


def distance(a: Bot, b: Bot) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def move_toward(bot: Bot, tx: int, ty: int) -> Tuple[int, int]:
    """Move bot toward target, up to bot.speed tiles."""
    dx = tx - bot.x
    dy = ty - bot.y
    dist = max(abs(dx), abs(dy))
    if dist == 0:
        return bot.x, bot.y
    steps = min(bot.speed, dist)
    # Normalize and scale
    nx = bot.x + round(dx / dist * steps)
    ny = bot.y + round(dy / dist * steps)
    return max(0, min(GRID_SIZE - 1, nx)), max(0, min(GRID_SIZE - 1, ny))


def move_away(bot: Bot, tx: int, ty: int) -> Tuple[int, int]:
    """Move bot away from target."""
    dx = bot.x - tx
    dy = bot.y - ty
    dist = max(abs(dx), abs(dy), 1)
    steps = bot.speed
    nx = bot.x + round(dx / dist * steps)
    ny = bot.y + round(dy / dist * steps)
    return max(0, min(GRID_SIZE - 1, nx)), max(0, min(GRID_SIZE - 1, ny))


def move_to_distance(bot: Bot, enemy: Bot, target_dist: int) -> Tuple[int, int]:
    """Move to maintain a specific distance from enemy."""
    d = distance(bot, enemy)
    if d < target_dist - 0.5:
        return move_away(bot, enemy.x, enemy.y)
    elif d > target_dist + 0.5:
        return move_toward(bot, enemy.x, enemy.y)
    return bot.x, bot.y


# ─── Decision Engine ──────────────────────────────────────────────────────────

def decide_action(bot: Bot, enemy: Bot) -> str:
    """
    Returns one of: 'attack', 'heal', 'special', 'retreat', 'kite', 'charge'
    
    The strategy is driven by the bot's config weights. Different weight
    combinations produce very different playstyles:
    - High aggression + low retreat = berserker
    - Low aggression + high kite_distance = sniper
    - High heal_threshold + high retreat = turtle
    - High burst_threshold + high special_power = assassin
    """
    d = distance(bot, enemy)
    hp_pct = bot.hp_pct()
    enemy_hp_pct = enemy.hp_pct()

    # Priority 1: Go all-in if enemy is low (burst mode)
    if enemy_hp_pct <= bot.burst_threshold and hp_pct > 0.2:
        if d <= bot.attack_range and bot.special_timer == 0 and bot.energy >= bot.attack_cost:
            return 'special'
        return 'charge'

    # Priority 2: Retreat if HP is critically low
    if hp_pct < bot.retreat_threshold:
        if hp_pct < bot.heal_threshold and bot.energy >= bot.heal_cost:
            return 'heal'
        return 'retreat'

    # Priority 3: Heal if below threshold and have energy
    if hp_pct < bot.heal_threshold and bot.energy >= bot.heal_cost:
        # Only heal if not in immediate danger or if very low
        if d > bot.attack_range + 2 or hp_pct < 0.3:
            return 'heal'

    # Priority 4: Special attack if available and in range
    if (bot.special_timer == 0 and d <= bot.attack_range
            and bot.energy >= bot.attack_cost * 1.5):
        return 'special'

    # Priority 5: Normal attack if in range
    if d <= bot.attack_range and bot.energy >= bot.attack_cost:
        return 'attack'

    # Priority 6: Movement decision based on aggression
    r = random.random()
    if r < bot.aggression:
        # Aggressive: close distance
        if d > bot.attack_range:
            return 'charge'
        return 'attack' if bot.energy >= bot.attack_cost else 'charge'
    else:
        # Defensive: maintain preferred distance
        if bot.kite_distance > 0 and d < bot.kite_distance:
            return 'kite'
        if d > bot.attack_range + 2:
            return 'charge'
        return 'kite'


def execute_action(bot: Bot, enemy: Bot, action: str, rng: random.Random):
    """Execute the chosen action, modifying bot and enemy state."""

    if action == 'attack':
        if distance(bot, enemy) <= bot.attack_range and bot.energy >= bot.attack_cost:
            bot.energy -= bot.attack_cost
            # Damage = attack * (1 - defense_reduction) + small random
            defense_factor = 1.0 - (enemy.defense / (enemy.defense + 50))
            dmg = bot.attack_power * defense_factor * (0.85 + rng.random() * 0.3)
            enemy.hp -= dmg
            bot.total_damage_dealt += dmg
            enemy.total_damage_taken += dmg

    elif action == 'special':
        if (distance(bot, enemy) <= bot.attack_range
                and bot.special_timer == 0
                and bot.energy >= bot.attack_cost * 1.5):
            bot.energy -= bot.attack_cost * 1.5
            bot.special_timer = bot.special_cooldown
            defense_factor = 1.0 - (enemy.defense / (enemy.defense + 50))
            dmg = (bot.attack_power * bot.special_power * defense_factor
                   * (0.9 + rng.random() * 0.2))
            enemy.hp -= dmg
            bot.total_damage_dealt += dmg
            enemy.total_damage_taken += dmg

    elif action == 'heal':
        if bot.energy >= bot.heal_cost:
            bot.energy -= bot.heal_cost
            heal = bot.heal_amount * (0.8 + rng.random() * 0.4)
            bot.hp = min(bot.max_hp, bot.hp + heal)

    elif action == 'charge':
        bot.x, bot.y = move_toward(bot, enemy.x, enemy.y)

    elif action == 'retreat':
        bot.x, bot.y = move_away(bot, enemy.x, enemy.y)

    elif action == 'kite':
        bot.x, bot.y = move_to_distance(bot, enemy, bot.kite_distance)

    # Passive effects every turn
    bot.energy = min(bot.max_energy, bot.energy + bot.energy_regen)
    if bot.special_timer > 0:
        bot.special_timer -= 1
    bot.turns_alive += 1


# ─── Match Simulation ────────────────────────────────────────────────────────

def make_bot(name: str, cfg: dict, start_x: int, start_y: int) -> Bot:
    """Create a bot from a config dict."""
    return Bot(
        name=name,
        x=start_x, y=start_y,
        hp=float(cfg.get('max_hp', 100)),
        max_hp=float(cfg.get('max_hp', 100)),
        energy=float(cfg.get('max_energy', 100)),
        max_energy=float(cfg.get('max_energy', 100)),
        attack_power=float(cfg.get('attack_power', 15)),
        defense=float(cfg.get('defense', 20)),
        speed=int(cfg.get('speed', 2)),
        heal_amount=float(cfg.get('heal_amount', 12)),
        aggression=float(cfg.get('aggression', 0.5)),
        retreat_threshold=float(cfg.get('retreat_threshold', 0.2)),
        heal_threshold=float(cfg.get('heal_threshold', 0.4)),
        attack_range=int(cfg.get('attack_range', 3)),
        energy_regen=float(cfg.get('energy_regen', 5)),
        heal_cost=float(cfg.get('heal_cost', 20)),
        attack_cost=float(cfg.get('attack_cost', 10)),
        special_cooldown=int(cfg.get('special_cooldown', 5)),
        special_power=float(cfg.get('special_power', 2.0)),
        kite_distance=int(cfg.get('kite_distance', 4)),
        burst_threshold=float(cfg.get('burst_threshold', 0.25)),
    )


# The champion: a balanced baseline bot that agents try to beat
CHAMPION_CONFIG = {
    'max_hp': 100,
    'max_energy': 100,
    'attack_power': 15,
    'defense': 20,
    'speed': 2,
    'heal_amount': 12,
    'aggression': 0.55,
    'retreat_threshold': 0.25,
    'heal_threshold': 0.45,
    'attack_range': 3,
    'energy_regen': 5,
    'heal_cost': 20,
    'attack_cost': 10,
    'special_cooldown': 5,
    'special_power': 2.0,
    'kite_distance': 4,
    'burst_threshold': 0.25,
}


def compute_stat_cost(cfg: dict) -> float:
    """
    Compute the 'cost' of a config. Prevents agents from just maxing everything.
    Each stat has a weight reflecting its value.
    """
    cost = 0.0
    cost += float(cfg.get('max_hp', 100)) * 1.0
    cost += float(cfg.get('max_energy', 100)) * 0.8
    cost += float(cfg.get('attack_power', 15)) * 4.0
    cost += float(cfg.get('defense', 20)) * 3.0
    cost += float(cfg.get('speed', 2)) * 15.0
    cost += float(cfg.get('heal_amount', 12)) * 3.0
    cost += float(cfg.get('attack_range', 3)) * 12.0
    cost += float(cfg.get('energy_regen', 5)) * 5.0
    cost += float(cfg.get('special_power', 2.0)) * 20.0
    return cost


def validate_config(cfg: dict) -> Tuple[bool, str]:
    """Check config is within allowed bounds."""
    cost = compute_stat_cost(cfg)
    champion_cost = compute_stat_cost(CHAMPION_CONFIG)
    max_allowed = champion_cost * 1.05  # 5% tolerance

    if cost > max_allowed:
        return False, f"Stat budget exceeded: {cost:.1f} > {max_allowed:.1f}"

    # Range checks
    checks = [
        ('max_hp', 50, 200), ('max_energy', 50, 200),
        ('attack_power', 5, 30), ('defense', 5, 40),
        ('speed', 1, 4), ('heal_amount', 0, 25),
        ('attack_range', 1, 6), ('energy_regen', 1, 12),
        ('heal_cost', 5, 40), ('attack_cost', 5, 25),
        ('special_cooldown', 2, 10), ('special_power', 1.0, 4.0),
        ('kite_distance', 0, 8), ('burst_threshold', 0.05, 0.5),
        ('aggression', 0.0, 1.0), ('retreat_threshold', 0.0, 0.6),
        ('heal_threshold', 0.0, 0.8),
    ]
    for name, lo, hi in checks:
        val = float(cfg.get(name, CHAMPION_CONFIG.get(name, 0)))
        if val < lo or val > hi:
            return False, f"{name}={val} out of range [{lo}, {hi}]"

    return True, "OK"


def adapt_champion(base_cfg: dict, challenger_cfg: dict) -> dict:
    """
    Champion partially adapts to counter the challenger's strategy.
    This prevents trivial one-dimensional solutions — agents must find
    builds that work well across multiple counter-strategies.
    
    The adaptation is moderate: it won't perfectly counter anything, but
    it punishes one-dimensional builds.
    """
    adapted = dict(base_cfg)
    
    c_agg = float(challenger_cfg.get('aggression', 0.5))
    c_range = int(challenger_cfg.get('attack_range', 3))
    c_hp = float(challenger_cfg.get('max_hp', 100))
    c_atk = float(challenger_cfg.get('attack_power', 15))
    c_def = float(challenger_cfg.get('defense', 20))
    c_spd = int(challenger_cfg.get('speed', 2))
    
    # Counter glass cannons: boost own defense and HP
    if c_hp < 85 or c_def < 15:
        adapted['defense'] = 28
        adapted['max_hp'] = 115
        adapted['heal_threshold'] = 0.5
        adapted['aggression'] = 0.7
        adapted['burst_threshold'] = 0.4
    
    # Counter high attack: boost defense, kite more
    if c_atk > 19:
        adapted['defense'] = max(adapted.get('defense', 20), 26)
        adapted['kite_distance'] = 5
        adapted['heal_threshold'] = 0.55
        adapted['retreat_threshold'] = 0.3
    
    # Counter long range: rush with speed
    if c_range >= 5:
        adapted['aggression'] = 0.9
        adapted['speed'] = 3
        adapted['kite_distance'] = 1
    
    # Counter kiting: be more aggressive
    if c_agg < 0.3:
        adapted['aggression'] = 0.8
        adapted['speed'] = min(3, c_spd + 1)
    
    return adapted


def run_match(challenger_cfg: dict, champion_cfg: dict, seed: int,
              verbose: bool = False) -> dict:
    """Run a single match. Returns result dict."""
    rng = random.Random(seed)

    # Champion adapts partially to challenger's build
    adapted_champ = adapt_champion(champion_cfg, challenger_cfg)

    challenger = make_bot("challenger", challenger_cfg, 2, GRID_SIZE // 2)
    champion = make_bot("champion", adapted_champ, GRID_SIZE - 3, GRID_SIZE // 2)

    log = []

    for turn in range(MAX_TURNS):
        # Both bots decide simultaneously
        c_action = decide_action(challenger, champion)
        h_action = decide_action(champion, challenger)

        # Execute (order randomized to avoid first-mover advantage)
        if rng.random() < 0.5:
            execute_action(challenger, champion, c_action, rng)
            execute_action(champion, challenger, h_action, rng)
        else:
            execute_action(champion, challenger, h_action, rng)
            execute_action(challenger, champion, c_action, rng)

        if verbose:
            log.append({
                'turn': turn,
                'challenger': {'hp': round(challenger.hp, 1), 'energy': round(challenger.energy, 1),
                               'x': challenger.x, 'y': challenger.y, 'action': c_action},
                'champion': {'hp': round(champion.hp, 1), 'energy': round(champion.energy, 1),
                             'x': champion.x, 'y': champion.y, 'action': h_action},
            })

        if not challenger.is_alive() or not champion.is_alive():
            break

    # Determine winner
    if challenger.is_alive() and not champion.is_alive():
        winner = 'challenger'
    elif champion.is_alive() and not challenger.is_alive():
        winner = 'champion'
    elif challenger.hp > champion.hp:
        winner = 'challenger'
    elif champion.hp > challenger.hp:
        winner = 'champion'
    else:
        winner = 'draw'

    return {
        'winner': winner,
        'turns': turn + 1,
        'challenger_hp': max(0, round(challenger.hp, 2)),
        'champion_hp': max(0, round(champion.hp, 2)),
        'challenger_dmg': round(challenger.total_damage_dealt, 2),
        'champion_dmg': round(champion.total_damage_dealt, 2),
        'log': log if verbose else [],
    }


def run_tournament(challenger_cfg: dict, n_matches: int = DEFAULT_MATCHES,
                   base_seed: int = 42, verbose: bool = False) -> dict:
    """Run N matches and return aggregate stats."""
    valid, msg = validate_config(challenger_cfg)
    if not valid:
        return {'error': msg, 'win_rate': 0.0}

    wins = 0
    draws = 0
    total_turns = 0
    total_dmg_dealt = 0
    total_dmg_taken = 0
    match_results = []

    for i in range(n_matches):
        result = run_match(challenger_cfg, CHAMPION_CONFIG, base_seed + i, verbose)
        if result['winner'] == 'challenger':
            wins += 1
        elif result['winner'] == 'draw':
            draws += 1
        total_turns += result['turns']
        total_dmg_dealt += result['challenger_dmg']
        total_dmg_taken += result['champion_dmg']
        match_results.append(result)

    win_rate = wins / n_matches
    draw_rate = draws / n_matches

    return {
        'win_rate': round(win_rate, 4),
        'draw_rate': round(draw_rate, 4),
        'avg_turns': round(total_turns / n_matches, 1),
        'avg_dmg_dealt': round(total_dmg_dealt / n_matches, 1),
        'avg_dmg_taken': round(total_dmg_taken / n_matches, 1),
        'matches': n_matches,
        'wins': wins,
        'draws': draws,
        'losses': n_matches - wins - draws,
        'stat_cost': round(compute_stat_cost(challenger_cfg), 1),
        'budget_max': round(compute_stat_cost(CHAMPION_CONFIG) * 1.05, 1),
        'results': match_results if verbose else [],
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Arena Combat Simulator')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to challenger config YAML')
    parser.add_argument('--matches', '-n', type=int, default=DEFAULT_MATCHES,
                        help=f'Number of matches (default: {DEFAULT_MATCHES})')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed results (JSON)')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output full results as JSON')
    args = parser.parse_args()

    # Load challenger config
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
        print(results['win_rate'])

        # Print summary to stderr (visible to agent but not captured as score)
        print(f"--- Match Summary ---", file=sys.stderr)
        print(f"Win rate: {results['win_rate']:.1%} ({results['wins']}W "
              f"{results['draws']}D {results['losses']}L / {results['matches']})",
              file=sys.stderr)
        print(f"Avg turns: {results['avg_turns']}", file=sys.stderr)
        print(f"Avg damage dealt/taken: {results['avg_dmg_dealt']}/{results['avg_dmg_taken']}",
              file=sys.stderr)
        print(f"Stat cost: {results['stat_cost']}/{results['budget_max']}",
              file=sys.stderr)


if __name__ == '__main__':
    main()
