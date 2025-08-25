# Copyright (©) 2025, Alexander Suvorov. All rights reserved.
import numpy as np
from typing import List, Union


def _compute_dist_matrix(cities: np.ndarray) -> np.ndarray:
    n = cities.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dx = cities[i, 0] - cities[j, 0]
            dy = cities[i, 1] - cities[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


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


def _greedy_tsp(cities: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    n = cities.shape[0]
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


def _simple_kmeans_cluster(data: np.ndarray, k: int, max_iters: int = 50) -> np.ndarray:
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

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


def hierarchical_tsp(
        cities: Union[np.ndarray, List[List[float]]],
        cluster_size: int = 100,
        post_optimize: bool = True
) -> List[int]:
    if not isinstance(cities, np.ndarray):
        cities = np.array(cities, dtype=np.float64)

    n = cities.shape[0]

    if n <= cluster_size:
        dist_matrix = _compute_dist_matrix(cities)
        route = _greedy_tsp(cities, dist_matrix)
        if post_optimize:
            route = _fast_2opt(route, dist_matrix, min(100, n * 2))
        return route[:-1].tolist()

    k = max(2, n // cluster_size)
    labels = _simple_kmeans_cluster(cities, k)

    centers = []
    for i in range(k):
        cluster_points = cities[labels == i]
        if len(cluster_points) > 0:
            centers.append(np.mean(cluster_points, axis=0))
    centers = np.array(centers)

    centers_dist_matrix = _compute_dist_matrix(centers)
    centers_route = _greedy_tsp(centers, centers_dist_matrix)
    if post_optimize:
        centers_route = _fast_2opt(centers_route, centers_dist_matrix, 50)

    final_route = []

    for cluster_idx in centers_route[:-1]:
        cluster_indices = np.where(labels == cluster_idx)[0]
        cluster_points = cities[cluster_indices]

        if len(cluster_points) <= cluster_size:
            cluster_dist_matrix = _compute_dist_matrix(cluster_points)
            cluster_route = _greedy_tsp(cluster_points, cluster_dist_matrix)
            if post_optimize:
                cluster_route = _fast_2opt(cluster_route, cluster_dist_matrix, 50)

            for point_idx in cluster_route[:-1]:
                final_route.append(cluster_indices[point_idx])
        else:
            sub_route = hierarchical_tsp(cluster_points, cluster_size, post_optimize)
            for point_idx in sub_route:
                final_route.append(cluster_indices[point_idx])

    if post_optimize and final_route:
        final_route_np = np.array(final_route + [final_route[0]], dtype=int)
        dist_matrix = _compute_dist_matrix(cities)
        optimized_route = _fast_2opt(final_route_np, dist_matrix, 50)
        return optimized_route[:-1].tolist()

    return final_route


def solve_tsp(cities: Union[np.ndarray, List[List[float]]]) -> List[int]:
    return hierarchical_tsp(cities, cluster_size=100, post_optimize=True)


def hierarchical_tsp_solver_v1(cities: np.ndarray, **kwargs) -> List[int]:
    cluster_size = kwargs.get('cluster_size', 100)
    post_optimize = kwargs.get('post_optimize', True)
    return hierarchical_tsp(cities, cluster_size, post_optimize)
