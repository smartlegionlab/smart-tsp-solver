# --------------------------------------------------------
# Copyright © 2025, A.A. Suvorov
# All rights reserved.
# --------------------------------------------------------
from typing import List

import numpy as np

from smart_tsp_heuristics.benchmark.utils.calculators.length import calculate_length
from smart_tsp_heuristics.benchmark.utils.optimizers.two_opt import two_opt_swap


def fast_post_optimize(cities: np.ndarray, route: List[int],
                       max_iter: int = 50) -> List[int]:
    best_route = route
    best_length = calculate_length(cities, route)
    n = len(route)

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, min(n - 1, i + 20)):
                if k - i < 2: continue
                new_route = two_opt_swap(best_route, i, k)
                new_length = calculate_length(cities, new_route)
                if new_length < best_length:
                    best_route = new_route
                    best_length = new_length
                    improved = True
        if not improved:
            break
    return best_route
