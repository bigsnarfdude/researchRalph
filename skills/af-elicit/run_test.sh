#!/bin/bash
# Test elicitation using claude pipe

cd /Users/vincent/researchRalph/skills/af-elicit

# Generate scenario
SCENARIO=$(python3 -c "
import sys
sys.path.insert(0, '.')
from elicit import TECHNIQUE_GENERATORS

tech = 'training_awareness'
config = TECHNIQUE_GENERATORS[tech]
scenario = config['generator']()

print('SYSTEM:', scenario['system_prompt'][:500])
print('---SEPARATOR---')
print('USER:', scenario['user_message'])
")

echo "$SCENARIO"
