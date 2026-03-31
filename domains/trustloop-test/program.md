# Sorting Algorithm Optimization

## Task
Optimize the sorting function in `solution.py` to **maximize the score** returned by `run.sh`.

The harness times your sort on arrays of various sizes and compares against Python's built-in `sorted()`. Score = fraction of test cases where your sort is within 10x of built-in speed.

## Harness
```bash
bash run.sh solution "description of what you tried" design_type
```

## What you edit
- `solution.py` — must export a function `sort_array(arr: list) -> list`
- The function must return a correctly sorted copy of the input
- Correctness is verified — wrong answers score 0

## Current baseline
Bubble sort: ~0.20 (only fast enough for tiny arrays)

## Tips
- Python's built-in `sorted()` uses Timsort — very hard to beat
- Focus on algorithmic improvements (merge sort, quicksort, radix sort)
- Consider the test distribution: small (10, 50, 100), medium (500, 1000), large (5000, 10000)
- Hybrid approaches often work well (e.g., insertion sort for small subarrays)
- Using `sorted()` directly is banned — the harness checks for it

## Scoring
- Metric: **fraction of test cases within 10x of built-in**
- Direction: **higher is better**
- Range: 0.0 to 1.0
- 7 test sizes (10 to 10000)
- Eval takes ~3 seconds
