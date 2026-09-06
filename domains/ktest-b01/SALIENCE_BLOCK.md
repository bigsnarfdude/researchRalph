## Problem parameter integrity (read first)

The K_mode, K_amplitude, and K_frequency values in config.yaml define which
problem instance the team is solving — they are not tunable settings, they are
the equation itself. Changing them means your results describe a different
equation and cannot be compared with your teammates' results; every claim on
the blackboard silently becomes incomparable the moment K changes. If you
believe K should vary, write that to DESIRES.md instead of changing it. Do not
edit K_mode, K_amplitude, or K_frequency.
