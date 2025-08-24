# -----------------------------------------------------------
# Copyright (©) 2025, Alexander Suvorov. All rights reserved.
# -----------------------------------------------------------
import numpy as np
from numba import njit
from typing import List, Union


@njit(fastmath=True, cache=True)
def _compute_dist_matrix(cities: np.ndarray) -> np.ndarray:
    n = cities.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dx = cities[i, 0] - cities[j, 0]
            dy = cities[i, 1] - cities[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


@njit(fastmath=True, cache=True)
def _fast_2opt(route: np.ndarray, dist_matrix: np.ndarray, max_iter: int = 20) -> np.ndarray:
    n = len(route)
    best_route = route.copy()
    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        for i in range(1, n - 2):
            a, b = best_route[i - 1], best_route[i]
            ab = dist_matrix[a, b]

            for j in range(i + 1, min(i + 50, n - 1)):
                c, d = best_route[j], best_route[j + 1]
                current = ab + dist_matrix[c, d]
                proposed = dist_matrix[a, c] + dist_matrix[b, d]

                if proposed < current - 1e-6:
                    best_route[i:j + 1] = best_route[i:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        iteration += 1
    return best_route


@njit(fastmath=True, cache=True)
def _numba_dynamic_gravity(cities: np.ndarray, delta: float) -> np.ndarray:
    n = cities.shape[0]
    route = np.empty(n + 1, dtype=np.int32)
    remaining = np.ones(n, dtype=np.uint8)

    center = np.array([np.mean(cities[:, 0]), np.mean(cities[:, 1])])
    min_dist = np.inf
    start_idx = 0
    for i in range(n):
        dx = cities[i, 0] - center[0]
        dy = cities[i, 1] - center[1]
        dist = dx * dx + dy * dy
        if dist < min_dist:
            min_dist = dist
            start_idx = i

    route[0] = start_idx
    remaining[start_idx] = 0
    current_pos = center.copy()
    last_vector = np.array([0.0, 0.0])

    for i in range(1, n):
        min_score = np.inf
        nearest = 0

        for j in range(n):
            if remaining[j]:
                dx = cities[j, 0] - current_pos[0]
                dy = cities[j, 1] - current_pos[1]
                dist_sq = dx * dx + dy * dy

                if i > 1:
                    dot_product = dx * last_vector[0] + dy * last_vector[1]
                    norm = np.sqrt(dist_sq * (last_vector[0] ** 2 + last_vector[1] ** 2))
                    if norm > 0:
                        cos_angle = dot_product / norm
                        angle_penalty = 0.3 * (1.0 - cos_angle)
                        dist_sq *= (1.0 + angle_penalty)

                if dist_sq < min_score:
                    min_score = dist_sq
                    nearest = j

        route[i] = nearest
        remaining[nearest] = 0
        last_vector = np.array([cities[nearest, 0] - current_pos[0],
                                cities[nearest, 1] - current_pos[1]])
        current_pos = cities[nearest] * delta + current_pos * (1 - delta)

    route[-1] = route[0]
    return route


def dynamic_gravity_tsp_v2(
        cities: Union[np.ndarray, List[List[float]]],
        delta: float = 0.5,
        post_optimize: bool = True,
        fast_2opt_iter: int = 100
) -> List[int]:
    if not isinstance(cities, np.ndarray):
        cities = np.array(cities, dtype=np.float64)

    if len(cities) == 0:
        return []

    route = _numba_dynamic_gravity(cities, delta)

    if post_optimize:
        dist_matrix = _compute_dist_matrix(cities)
        route = _fast_2opt(route, dist_matrix, fast_2opt_iter)

    return route.tolist()
