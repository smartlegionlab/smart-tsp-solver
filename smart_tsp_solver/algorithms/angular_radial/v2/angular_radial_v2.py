# Copyright (©) 2025, Alexander Suvorov. All rights reserved.
import numpy as np
from numba import njit
from typing import List, Optional, Union


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
def _2opt_swap(route: np.ndarray, i: int, k: int) -> np.ndarray:
    new_route = route.copy()
    new_route[i:k + 1] = route[i:k + 1][::-1]
    return new_route


@njit(fastmath=True, cache=True)
def _fast_2opt(route: np.ndarray, dist_matrix: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    n = len(route)
    best_route = route.copy()
    best_cost = 0.0
    for i in range(n - 1):
        best_cost += dist_matrix[route[i], route[i + 1]]

    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        for i in range(1, n - 2):
            a, b = best_route[i - 1], best_route[i]
            ab_dist = dist_matrix[a, b]

            for j in range(i + 1, n - 1):
                if j == i + 1:
                    continue

                c, d = best_route[j], best_route[j + 1]
                cd_dist = dist_matrix[c, d]
                current = ab_dist + cd_dist
                proposed = dist_matrix[a, c] + dist_matrix[b, d]

                if proposed < current:
                    best_route[i:j + 1] = best_route[i:j + 1][::-1]
                    best_cost += proposed - current
                    improved = True
                    break

            if improved:
                break
        iteration += 1
    return best_route


@njit(fastmath=True, cache=True)
def _numba_angular_radial(
        cities: np.ndarray,
        center: np.ndarray,
        sort_by_code: int,
        start_idx: int,
        look_ahead: int
) -> np.ndarray:
    n = cities.shape[0]
    dxdy = cities - center
    angles = np.empty(n, dtype=np.float64)
    distances = np.empty(n, dtype=np.float64)

    for i in range(n):
        dx = dxdy[i, 0]
        dy = dxdy[i, 1]
        angles[i] = np.arctan2(dy, dx)
        distances[i] = dx * dx + dy * dy

    if sort_by_code == 0:
        indices = np.argsort(distances + angles / (2 * np.pi))
    elif sort_by_code == 1:
        indices = np.argsort(angles + distances / np.max(distances))
    else:
        indices = np.argsort(angles)

    route = np.empty(n + 1, dtype=np.int32)
    route[0] = start_idx
    remaining = np.ones(n, dtype=np.bool_)
    remaining[start_idx] = False

    last_point = cities[start_idx]

    for i in range(1, n):
        min_dist = np.inf
        best_candidate = -1
        search_end = min(i + look_ahead, n)

        for j in range(search_end):
            candidate = indices[j]
            if remaining[candidate]:
                dx = cities[candidate, 0] - last_point[0]
                dy = cities[candidate, 1] - last_point[1]
                dist = dx * dx + dy * dy
                if dist < min_dist:
                    min_dist = dist
                    best_candidate = candidate

        if best_candidate == -1:
            for j in range(n):
                if remaining[j]:
                    dx = cities[j, 0] - last_point[0]
                    dy = cities[j, 1] - last_point[1]
                    dist = dx * dx + dy * dy
                    if dist < min_dist:
                        min_dist = dist
                        best_candidate = j

        route[i] = best_candidate
        remaining[best_candidate] = False
        last_point = cities[best_candidate]

    route[-1] = route[0]
    return route


def angular_radial_tsp_v2(
        cities: Union[np.ndarray, List[List[float]]],
        sort_by: str = 'angle_distance',
        start_point: Optional[int] = None,
        look_ahead: int = 100,
        post_optimize: bool = True,
        max_2opt_iter: int = 100
) -> List[int]:
    if not isinstance(cities, np.ndarray):
        cities = np.array(cities, dtype=np.float64)

    if len(cities) == 0:
        return []

    n = len(cities)
    if start_point is not None and start_point >= n:
        start_point = n - 1

    sort_by_code = 0 if sort_by == 'angle_distance' else 1 if sort_by == 'distance_angle' else 2
    start_idx = 0 if start_point is None else start_point
    center = np.mean(cities, axis=0) if start_point is None else cities[start_point]

    route = _numba_angular_radial(
        cities.astype(np.float64),
        center.astype(np.float64),
        sort_by_code,
        start_idx,
        look_ahead
    )

    if post_optimize:
        dist_matrix = _compute_dist_matrix(cities)
        route = _fast_2opt(route, dist_matrix, max_2opt_iter)

    return route.tolist()
