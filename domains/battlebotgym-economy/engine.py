#!/usr/bin/env python3
"""
Resource Economy Simulator — researchRalph Game Domain

A village of 100 people must survive 100 days. The agent configures resource
allocation strategy via config.yaml. Workers are assigned to farming, mining,
building, and research. Random events (drought, raid, plague, bounty) test the
village's resilience.

Score = survival rate (villagers alive / 100) after 100 days.
"""

import yaml
import sys
import random
import math
import argparse
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# --- Game Constants -------------------------------------------------------

TOTAL_DAYS = 100
INITIAL_POPULATION = 100
DEFAULT_MATCHES = 200

# Base production per worker per day
BASE_FOOD_PER_FARMER = 2.3
BASE_MATERIALS_PER_MINER = 1.8
BASE_BUILDING_RATE = 0.7       # building points per builder per day
BASE_RESEARCH_RATE = 0.45      # knowledge points per researcher per day

# Consumption
FOOD_PER_PERSON_PER_DAY = 1.1
FOOD_SPOILAGE_RATE = 0.025     # 2.5% of stored food spoils daily

# Building effects
BUILDING_FARM_BONUS = 0.003    # each building point adds 0.3% farm output
BUILDING_MINE_BONUS = 0.002    # each building point adds 0.2% mine output
BUILDING_DEFENSE_PER_POINT = 0.25  # defense rating per building point

# Research effects
RESEARCH_FARM_BONUS = 0.004    # each knowledge point adds 0.4% farm output
RESEARCH_MINE_BONUS = 0.003    # each knowledge point adds 0.3% mine output
RESEARCH_TRADE_BONUS = 0.002   # each knowledge point improves trade rates 0.2%
RESEARCH_MEDICINE_BONUS = 0.004  # each knowledge point reduces plague deaths 0.4%

# Trade base rates (what you get per unit spent)
TRADE_FOOD_TO_GOLD = 0.3       # 1 food -> 0.3 gold
TRADE_MATERIALS_TO_GOLD = 0.5  # 1 material -> 0.5 gold
TRADE_GOLD_TO_FOOD = 1.5       # 1 gold -> 1.5 food
TRADE_GOLD_TO_MATERIALS = 1.2  # 1 gold -> 1.2 materials

# Event probabilities (per day, before modifiers)
EVENT_DROUGHT_BASE = 0.07
EVENT_RAID_BASE = 0.06
EVENT_PLAGUE_BASE = 0.045
EVENT_BOUNTY_BASE = 0.07

# Event severity
DROUGHT_FOOD_LOSS_FACTOR = 0.30  # farming output reduced to 30% during drought
DROUGHT_DURATION = 10            # days
DROUGHT_COOLDOWN = 8             # days of immunity after drought ends

RAID_BASE_DEATHS = 8             # base villager deaths in raid
RAID_FOOD_STEAL_FRACTION = 0.30  # fraction of food stolen
RAID_MATERIAL_STEAL_FRACTION = 0.20
RAID_GOLD_STEAL_FRACTION = 0.25

PLAGUE_BASE_DEATH_RATE = 0.10    # 10% of population dies (before medicine)
PLAGUE_DURATION = 6              # days of reduced productivity
PLAGUE_COOLDOWN = 10             # days of immunity after plague ends

# Stacking control — max 1 new catastrophe (drought/raid/plague) per day
MAX_NEW_CATASTROPHES_PER_DAY = 1

BOUNTY_FOOD_BONUS = 25           # free food
BOUNTY_GOLD_BONUS = 10           # free gold
BOUNTY_MATERIALS_BONUS = 12      # free materials


# --- Data Types -----------------------------------------------------------

@dataclass
class Village:
    population: int
    food: float
    materials: float
    gold: float
    knowledge: float
    buildings: float
    # Worker allocation (absolute counts, recomputed each day from config %)
    farmers: int = 0
    miners: int = 0
    builders: int = 0
    researchers: int = 0
    # Ongoing effects
    drought_days: int = 0
    plague_days: int = 0
    drought_cooldown: int = 0       # immunity countdown after drought ends
    plague_cooldown: int = 0        # immunity countdown after plague ends
    defense_rating: float = 0.0
    # Tracking
    peak_population: int = 100
    total_deaths: int = 0
    total_food_produced: float = 0.0
    total_events: Dict[str, int] = field(default_factory=lambda: {
        'drought': 0, 'raid': 0, 'plague': 0, 'bounty': 0
    })
    starvation_days: int = 0
    day: int = 0


@dataclass
class DayLog:
    day: int
    population: int
    food: float
    materials: float
    gold: float
    knowledge: float
    buildings: float
    event: str
    food_produced: float
    food_consumed: float
    deaths: int
    cause: str


# --- Config Validation ----------------------------------------------------

def validate_config(cfg: dict) -> Tuple[bool, str]:
    """Check config is valid."""
    # Worker allocation must sum to 100
    farming = float(cfg.get('farming_pct', 50))
    mining = float(cfg.get('mining_pct', 20))
    building = float(cfg.get('building_pct', 15))
    research = float(cfg.get('research_pct', 15))
    total = farming + mining + building + research

    if abs(total - 100.0) > 0.5:
        return False, f"Worker allocation sums to {total:.1f}, must be 100"

    # Range checks
    checks = [
        ('farming_pct', 0, 100),
        ('mining_pct', 0, 100),
        ('building_pct', 0, 100),
        ('research_pct', 0, 100),
        ('food_reserve_target', 0, 500),
        ('trade_aggression', 0.0, 1.0),
        ('building_priority', 0.0, 1.0),
        ('research_focus', 0.0, 1.0),
        ('raid_preparedness', 0.0, 1.0),
        ('starvation_response', 0.0, 1.0),
        ('trade_food_threshold', 0, 500),
        ('trade_materials_threshold', 0, 500),
        ('gold_reserve_target', 0, 200),
        ('emergency_food_buy', 0.0, 1.0),
    ]
    for name, lo, hi in checks:
        val = float(cfg.get(name, 0))
        if val < lo or val > hi:
            return False, f"{name}={val} out of range [{lo}, {hi}]"

    return True, "OK"


# --- Simulation -----------------------------------------------------------

def allocate_workers(village: Village, cfg: dict):
    """Assign workers based on config percentages and current population."""
    pop = village.population
    if pop <= 0:
        village.farmers = village.miners = village.builders = village.researchers = 0
        return

    farming_pct = float(cfg.get('farming_pct', 50))
    mining_pct = float(cfg.get('mining_pct', 20))
    building_pct = float(cfg.get('building_pct', 15))
    research_pct = float(cfg.get('research_pct', 15))

    # Starvation response: shift workers to farming when food is low
    starvation_response = float(cfg.get('starvation_response', 0.5))
    food_reserve_target = float(cfg.get('food_reserve_target', 80))

    food_ratio = village.food / max(food_reserve_target, 1)
    if food_ratio < 0.5:
        # Emergency: shift non-farmers proportionally
        shift = starvation_response * (1.0 - food_ratio * 2)  # 0 to starvation_response
        non_farm = mining_pct + building_pct + research_pct
        if non_farm > 0:
            mining_pct *= (1 - shift)
            building_pct *= (1 - shift)
            research_pct *= (1 - shift)
            farming_pct = 100 - mining_pct - building_pct - research_pct

    # Convert percentages to headcounts
    village.farmers = max(0, round(pop * farming_pct / 100))
    village.miners = max(0, round(pop * mining_pct / 100))
    village.builders = max(0, round(pop * building_pct / 100))
    village.researchers = pop - village.farmers - village.miners - village.builders
    village.researchers = max(0, village.researchers)

    # Ensure total doesn't exceed population
    total = village.farmers + village.miners + village.builders + village.researchers
    while total > pop:
        # Trim from smallest group
        groups = [('researchers', village.researchers),
                  ('builders', village.builders),
                  ('miners', village.miners)]
        groups.sort(key=lambda x: x[1])
        for name, count in groups:
            if count > 0:
                setattr(village, name, count - 1)
                total -= 1
                break
        if total <= pop:
            break


def produce_resources(village: Village, cfg: dict, rng: random.Random):
    """Daily production phase."""
    # Farm bonus from buildings and research
    farm_mult = 1.0
    farm_mult += village.buildings * BUILDING_FARM_BONUS
    farm_mult += village.knowledge * RESEARCH_FARM_BONUS

    # Drought halves farming
    if village.drought_days > 0:
        farm_mult *= DROUGHT_FOOD_LOSS_FACTOR

    # Plague reduces all productivity
    plague_mult = 0.7 if village.plague_days > 0 else 1.0

    # Random daily variance (+-15%)
    daily_var = 0.85 + rng.random() * 0.30

    food_produced = village.farmers * BASE_FOOD_PER_FARMER * farm_mult * plague_mult * daily_var
    village.food += food_produced
    village.total_food_produced += food_produced

    # Mining
    mine_mult = 1.0
    mine_mult += village.buildings * BUILDING_MINE_BONUS
    mine_mult += village.knowledge * RESEARCH_MINE_BONUS
    materials_produced = village.miners * BASE_MATERIALS_PER_MINER * mine_mult * plague_mult * daily_var
    village.materials += materials_produced

    # Building
    if village.materials >= village.builders * 0.5:  # builders need materials
        materials_used = village.builders * 0.5 * plague_mult
        village.materials -= materials_used
        village.buildings += village.builders * BASE_BUILDING_RATE * plague_mult * daily_var
    else:
        # Partial building with available materials
        possible = village.materials / 0.5
        village.materials = 0
        village.buildings += possible * BASE_BUILDING_RATE * plague_mult * daily_var * 0.5

    # Research
    knowledge_gained = village.researchers * BASE_RESEARCH_RATE * plague_mult * daily_var
    village.knowledge += knowledge_gained

    # Update defense rating from buildings
    village.defense_rating = village.buildings * BUILDING_DEFENSE_PER_POINT

    return food_produced


def handle_trade(village: Village, cfg: dict, rng: random.Random):
    """Daily trade phase — convert between resources based on strategy."""
    trade_aggression = float(cfg.get('trade_aggression', 0.3))
    trade_food_threshold = float(cfg.get('trade_food_threshold', 120))
    trade_materials_threshold = float(cfg.get('trade_materials_threshold', 80))
    gold_reserve_target = float(cfg.get('gold_reserve_target', 30))
    emergency_food_buy = float(cfg.get('emergency_food_buy', 0.5))
    food_reserve_target = float(cfg.get('food_reserve_target', 80))

    # Trade rate bonus from research
    trade_mult = 1.0 + village.knowledge * RESEARCH_TRADE_BONUS
    # Market fluctuation (+-20%)
    market = 0.80 + rng.random() * 0.40

    # Emergency food purchase if starving
    if village.food < village.population * 2 and village.gold > 5:
        gold_spend = village.gold * emergency_food_buy
        food_gained = gold_spend * TRADE_GOLD_TO_FOOD * trade_mult * market
        village.gold -= gold_spend
        village.food += food_gained
        return

    # Sell excess food for gold
    if village.food > trade_food_threshold * trade_aggression + food_reserve_target:
        excess = (village.food - food_reserve_target) * trade_aggression * 0.3
        if excess > 0:
            gold_gained = excess * TRADE_FOOD_TO_GOLD * trade_mult * market
            village.food -= excess
            village.gold += gold_gained

    # Sell excess materials for gold
    if village.materials > trade_materials_threshold:
        excess = (village.materials - trade_materials_threshold) * trade_aggression * 0.3
        if excess > 0:
            gold_gained = excess * TRADE_MATERIALS_TO_GOLD * trade_mult * market
            village.materials -= excess
            village.gold += gold_gained

    # Buy food with gold if below target
    if village.food < food_reserve_target * 0.6 and village.gold > gold_reserve_target:
        gold_spend = (village.gold - gold_reserve_target) * trade_aggression * 0.5
        if gold_spend > 0:
            food_gained = gold_spend * TRADE_GOLD_TO_FOOD * trade_mult * market
            village.gold -= gold_spend
            village.food += food_gained


def handle_consumption(village: Village, rng: random.Random) -> Tuple[int, str]:
    """Daily food consumption. Returns (deaths, cause)."""
    # Food spoilage
    village.food *= (1.0 - FOOD_SPOILAGE_RATE)

    food_needed = village.population * FOOD_PER_PERSON_PER_DAY
    deaths = 0
    cause = ""

    if village.food >= food_needed:
        village.food -= food_needed
    else:
        # Not enough food — some die
        village.food = max(0, village.food)
        shortfall = food_needed - village.food
        village.food = 0
        # Deaths proportional to shortfall, with randomness
        death_rate = min(0.25, (shortfall / food_needed) * 0.3)
        deaths = max(1, int(village.population * death_rate * (0.7 + rng.random() * 0.6)))
        deaths = min(deaths, village.population)
        village.population -= deaths
        village.total_deaths += deaths
        village.starvation_days += 1
        cause = "starvation"

    return deaths, cause


def roll_events(village: Village, cfg: dict, rng: random.Random) -> str:
    """Check for random events. Returns event name or ''.

    Variance control:
      - Refractory periods prevent drought/plague from re-triggering immediately
        after ending (DROUGHT_COOLDOWN / PLAGUE_COOLDOWN days of immunity).
      - At most MAX_NEW_CATASTROPHES_PER_DAY catastrophes (drought/raid/plague)
        can trigger on a single day, preventing simultaneous triple-stacking.
      - Bounties are always allowed (positive events don't need capping).
    """
    raid_preparedness = float(cfg.get('raid_preparedness', 0.3))

    events = []
    catastrophes_today = 0

    # --- Drought: seasonal, with refractory cooldown ----------------------
    day = village.day
    season_factor = 1.0 + 0.5 * math.sin(day * 2 * math.pi / 100)  # peak around day 25
    drought_eligible = (village.drought_days == 0 and village.drought_cooldown == 0)
    if (drought_eligible
            and catastrophes_today < MAX_NEW_CATASTROPHES_PER_DAY
            and rng.random() < EVENT_DROUGHT_BASE * season_factor):
        village.drought_days = DROUGHT_DURATION
        village.total_events['drought'] += 1
        catastrophes_today += 1
        events.append("drought")

    # --- Raid: wealth-attracted, defense-mitigated ------------------------
    wealth_factor = min(2.0, (village.food + village.materials + village.gold * 3) / 200)
    defense_reduction = village.defense_rating / (village.defense_rating + 50)
    raid_prob = EVENT_RAID_BASE * wealth_factor * (1.0 - defense_reduction * 0.7)
    if (catastrophes_today < MAX_NEW_CATASTROPHES_PER_DAY
            and rng.random() < raid_prob):
        defense_factor = 1.0 - defense_reduction
        prep_factor = 1.0 - raid_preparedness * 0.4

        raid_deaths = max(0, int(RAID_BASE_DEATHS * defense_factor * prep_factor
                                  * (0.5 + rng.random())))
        raid_deaths = min(raid_deaths, village.population // 4)  # cap at 25% pop

        village.population -= raid_deaths
        village.total_deaths += raid_deaths

        village.food -= village.food * RAID_FOOD_STEAL_FRACTION * defense_factor
        village.materials -= village.materials * RAID_MATERIAL_STEAL_FRACTION * defense_factor
        village.gold -= village.gold * RAID_GOLD_STEAL_FRACTION * defense_factor

        building_damage = (1.0 - defense_reduction) * rng.random() * 5
        village.buildings = max(0, village.buildings - building_damage)

        village.food = max(0, village.food)
        village.materials = max(0, village.materials)
        village.gold = max(0, village.gold)

        village.total_events['raid'] += 1
        catastrophes_today += 1
        events.append("raid")

    # --- Plague: density-dependent, with refractory cooldown --------------
    density_factor = village.population / INITIAL_POPULATION
    plague_eligible = (village.plague_days == 0 and village.plague_cooldown == 0)
    if (plague_eligible
            and catastrophes_today < MAX_NEW_CATASTROPHES_PER_DAY
            and rng.random() < EVENT_PLAGUE_BASE * density_factor):
        medicine_factor = 1.0 - min(0.8, village.knowledge * RESEARCH_MEDICINE_BONUS)
        plague_deaths = max(0, int(village.population * PLAGUE_BASE_DEATH_RATE
                                    * medicine_factor * (0.6 + rng.random() * 0.8)))
        plague_deaths = min(plague_deaths, village.population // 3)
        village.population -= plague_deaths
        village.total_deaths += plague_deaths
        village.plague_days = PLAGUE_DURATION
        village.total_events['plague'] += 1
        catastrophes_today += 1
        events.append("plague")

    # --- Bounty: always allowed (positive event) --------------------------
    if rng.random() < EVENT_BOUNTY_BASE:
        village.food += BOUNTY_FOOD_BONUS * (0.5 + rng.random())
        village.gold += BOUNTY_GOLD_BONUS * (0.5 + rng.random())
        village.materials += BOUNTY_MATERIALS_BONUS * (0.5 + rng.random())
        village.total_events['bounty'] += 1
        events.append("bounty")

    # --- Tick ongoing effects and cooldowns -------------------------------
    def tick_effect(days, cooldown, cooldown_duration):
        if days > 0:
            days -= 1
            if days == 0:
                cooldown = cooldown_duration
        elif cooldown > 0:
            cooldown -= 1
        return days, cooldown

    village.drought_days, village.drought_cooldown = tick_effect(
        village.drought_days, village.drought_cooldown, DROUGHT_COOLDOWN)
    village.plague_days, village.plague_cooldown = tick_effect(
        village.plague_days, village.plague_cooldown, PLAGUE_COOLDOWN)

    return "+".join(events)


# --- Main Simulation ------------------------------------------------------

def run_simulation(cfg: dict, seed: int, verbose: bool = False) -> dict:
    """Run a single 100-day simulation. Returns result dict."""
    rng = random.Random(seed)

    village = Village(
        population=INITIAL_POPULATION,
        food=70.0,
        materials=35.0,
        gold=18.0,
        knowledge=0.0,
        buildings=0.0,
    )

    log = []

    for day in range(TOTAL_DAYS):
        village.day = day

        # 1. Allocate workers
        allocate_workers(village, cfg)

        # 2. Produce resources
        food_produced = produce_resources(village, cfg, rng)

        # 3. Trade
        handle_trade(village, cfg, rng)

        # 4. Random events
        event = roll_events(village, cfg, rng)

        # 5. Consume food
        deaths, cause = handle_consumption(village, rng)

        # Track peak
        village.peak_population = max(village.peak_population, village.population)

        if verbose:
            log.append(DayLog(
                day=day,
                population=village.population,
                food=round(village.food, 1),
                materials=round(village.materials, 1),
                gold=round(village.gold, 1),
                knowledge=round(village.knowledge, 1),
                buildings=round(village.buildings, 1),
                event=event,
                food_produced=round(food_produced, 1),
                food_consumed=round(village.population * FOOD_PER_PERSON_PER_DAY, 1),
                deaths=deaths,
                cause=cause,
            ))

        if village.population <= 0:
            break

    survival_rate = village.population / INITIAL_POPULATION

    return {
        'survival_rate': round(survival_rate, 4),
        'final_population': village.population,
        'total_deaths': village.total_deaths,
        'days_survived': village.day + 1,
        'final_food': round(village.food, 1),
        'final_materials': round(village.materials, 1),
        'final_gold': round(village.gold, 1),
        'final_knowledge': round(village.knowledge, 1),
        'final_buildings': round(village.buildings, 1),
        'events': dict(village.total_events),
        'starvation_days': village.starvation_days,
        'log': [vars(e) for e in log] if verbose else [],
    }


def run_campaign(cfg: dict, n_matches: int = DEFAULT_MATCHES,
                 base_seed: int = 42, verbose: bool = False) -> dict:
    """Run N simulations and return aggregate stats."""
    valid, msg = validate_config(cfg)
    if not valid:
        return {'error': msg, 'survival_rate': 0.0}

    total_survival = 0.0
    total_deaths = 0
    total_starvation_days = 0
    min_survival = 1.0
    max_survival = 0.0
    wipe_count = 0  # total village deaths
    match_results = []

    for i in range(n_matches):
        result = run_simulation(cfg, base_seed + i, verbose)
        sr = result['survival_rate']
        total_survival += sr
        total_deaths += result['total_deaths']
        total_starvation_days += result['starvation_days']
        min_survival = min(min_survival, sr)
        max_survival = max(max_survival, sr)
        if result['final_population'] == 0:
            wipe_count += 1
        match_results.append(result)

    avg_survival = total_survival / n_matches

    return {
        'survival_rate': round(avg_survival, 4),
        'min_survival': round(min_survival, 4),
        'max_survival': round(max_survival, 4),
        'avg_deaths': round(total_deaths / n_matches, 1),
        'avg_starvation_days': round(total_starvation_days / n_matches, 1),
        'wipe_count': wipe_count,
        'matches': n_matches,
        'results': match_results if verbose else [],
    }


# --- CLI ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Resource Economy Simulator')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to village config YAML')
    parser.add_argument('--matches', '-n', type=int, default=DEFAULT_MATCHES,
                        help=f'Number of simulations (default: {DEFAULT_MATCHES})')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Include per-day logs in output')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output full results as JSON')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_campaign(cfg, args.matches, args.seed, args.verbose)

    if 'error' in results:
        print(f"INVALID CONFIG: {results['error']}", file=sys.stderr)
        print("0.0")
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Print just the score (what researchRalph reads)
        print(results['survival_rate'])

        # Print summary to stderr (visible to agent but not captured as score)
        print(f"--- Simulation Summary ---", file=sys.stderr)
        print(f"Survival rate: {results['survival_rate']:.1%} "
              f"(min {results['min_survival']:.1%}, max {results['max_survival']:.1%})",
              file=sys.stderr)
        print(f"Avg deaths: {results['avg_deaths']}", file=sys.stderr)
        print(f"Avg starvation days: {results['avg_starvation_days']}", file=sys.stderr)
        print(f"Total wipes: {results['wipe_count']}/{results['matches']}", file=sys.stderr)


if __name__ == '__main__':
    main()
