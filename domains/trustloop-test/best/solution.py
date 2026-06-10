"""Sorting solution — agents optimize this file."""

import heapq

def sort_array(arr: list) -> list:
    """Use heapq.nsmallest — C-implemented heap operations."""
    return heapq.nsmallest(len(arr), arr)
