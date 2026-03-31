"""Sorting solution — agents optimize this file."""

def sort_array(arr: list) -> list:
    """Sort array. Baseline: bubble sort."""
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
