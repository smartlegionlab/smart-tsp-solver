# Copyright (©) 2025, Alexander Suvorov. All rights reserved.
import numpy as np
from numba import njit
from typing import List


@njit(fastmath=True, cache=True)
def _greedy_tsp_numba(cities: np.ndarray, start_point: int) -> np.ndarray:
    n = cities.shape[0]
    route = np.empty(n + 1, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)

    current = start_point
    route[0] = current
    visited[current] = True

    for i in range(1, n):
        min_dist = np.inf
        nearest = current

        for j in range(n):
            if not visited[j]:
                dx = cities[current, 0] - cities[j, 0]
                dy = cities[current, 1] - cities[j, 1]
                dist = dx * dx + dy * dy

                if dist < min_dist:
                    min_dist = dist
                    nearest = j

        current = nearest
        route[i] = current
        visited[current] = True

    route[-1] = route[0]
    return route


def greedy_tsp_v2(cities: np.ndarray, start_point: int = 0) -> List[int]:
    if not isinstance(cities, np.ndarray):
        cities = np.array(cities, dtype=np.float64)

    if len(cities) == 0:
        return []

    n = len(cities)
    start_point = max(0, min(start_point, n - 1))

    route = _greedy_tsp_numba(
        cities.astype(np.float64),
        start_point
    )

    return route.tolist()
