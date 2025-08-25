# Copyright (©) 2025, Alexander Suvorov. All rights reserved.
import numpy as np
from typing import List, Union, Tuple


def _compute_dist_matrix_fast(cities: np.ndarray) -> np.ndarray:
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _fast_2opt(route: np.ndarray, dist_matrix: np.ndarray, max_iter: int = 20) -> np.ndarray:
    n = len(route)
    if n <= 3:
        return route.copy()

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


def _greedy_tsp_simple(points: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0, 0], dtype=int)

    visited = np.zeros(n, dtype=bool)
    route = np.zeros(n + 1, dtype=int)

    current = 0
    route[0] = current
    visited[current] = True

    for i in range(1, n):
        min_dist = np.inf
        next_city = -1

        for j in range(n):
            if not visited[j] and dist_matrix[current, j] < min_dist:
                min_dist = dist_matrix[current, j]
                next_city = j

        if next_city == -1:
            break

        route[i] = next_city
        visited[next_city] = True
        current = next_city

    route[n] = route[0]
    return route


def _simple_kmeans(data: np.ndarray, k: int, max_iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]

    centroid_indices = np.random.choice(n, k, replace=False)
    centroids = data[centroid_indices]
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        for i in range(n):
            min_dist = np.inf
            for j in range(k):
                dist = np.sum((data[i] - centroids[j]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = j

        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=int)
        for i in range(n):
            cluster_idx = labels[i]
            new_centroids[cluster_idx] += data[i]
            counts[cluster_idx] += 1

        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                new_centroids[j] = data[np.random.randint(n)]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels, centroids


def hierarchical_tsp_optimized(
        cities: Union[np.ndarray, List[List[float]]],
        cluster_size: int = 100,
        post_optimize: bool = True
) -> List[int]:
    if not isinstance(cities, np.ndarray):
        cities = np.array(cities, dtype=np.float64)

    n = cities.shape[0]

    if n <= cluster_size:
        dist_matrix = _compute_dist_matrix_fast(cities)
        route = _greedy_tsp_simple(cities, dist_matrix)
        if post_optimize:
            route = _fast_2opt(route, dist_matrix, min(100, n * 2))
        return route[:-1].tolist()

    k = max(2, n // cluster_size)
    labels, centroids = _simple_kmeans(cities, k)

    centers_dist_matrix = _compute_dist_matrix_fast(centroids)
    centers_route = _greedy_tsp_simple(centroids, centers_dist_matrix)
    if post_optimize:
        centers_route = _fast_2opt(centers_route, centers_dist_matrix, 50)

    final_route = []

    for cluster_idx in centers_route[:-1]:
        cluster_mask = (labels == cluster_idx)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_points = cities[cluster_mask]

        if len(cluster_points) == 0:
            continue

        sub_route_indices = hierarchical_tsp_optimized(
            cluster_points, cluster_size, post_optimize
        )

        for local_idx in sub_route_indices:
            final_route.append(cluster_indices[local_idx])

    if post_optimize and final_route:
        final_route.append(final_route[0])
        final_route_np = np.array(final_route, dtype=int)
        dist_matrix = _compute_dist_matrix_fast(cities)
        optimized_route = _fast_2opt(final_route_np, dist_matrix, 50)
        return optimized_route[:-1].tolist()

    return final_route


def solve_tsp(cities: Union[np.ndarray, List[List[float]]]) -> List[int]:
    return hierarchical_tsp_optimized(cities, cluster_size=100, post_optimize=True)


def hierarchical_tsp_solver(cities: np.ndarray, **kwargs) -> List[int]:
    cluster_size = kwargs.get('cluster_size', 100)
    post_optimize = kwargs.get('post_optimize', True)
    return hierarchical_tsp_optimized(cities, cluster_size, post_optimize)


def hierarchical_tsp_simple(cities: np.ndarray, **kwargs) -> List[int]:
    if not isinstance(cities, np.ndarray):
        cities = np.array(cities, dtype=np.float64)

    n = cities.shape[0]

    if n > 1000:
        k = max(2, n // 200)
        labels, centroids = _simple_kmeans(cities, k)

        centers_dist_matrix = _compute_dist_matrix_fast(centroids)
        centers_route = _greedy_tsp_simple(centroids, centers_dist_matrix)
        centers_route = _fast_2opt(centers_route, centers_dist_matrix, 30)

        final_route = []
        for cluster_idx in centers_route[:-1]:
            cluster_indices = np.where(labels == cluster_idx)[0]
            if len(cluster_indices) > 0:
                final_route.append(cluster_indices[0])

        return final_route

    return hierarchical_tsp_optimized(cities, **kwargs)
