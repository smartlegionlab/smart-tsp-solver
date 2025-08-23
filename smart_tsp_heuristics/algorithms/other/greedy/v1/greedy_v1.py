# --------------------------------------------------------
# Copyright © 2025, A.A. Suvorov
# All rights reserved.
# --------------------------------------------------------
from typing import List

import numpy as np
from scipy.spatial import distance


def greedy_tsp_v1(cities: np.ndarray,
               start_point: int = 0) -> List[int]:
    n = len(cities)
    if start_point >= n:
        start_point = n - 1
    elif start_point < 0:
        start_point = 0

    unvisited = set(range(n))
    route = [start_point]
    unvisited.remove(start_point)

    while unvisited:
        last = route[-1]
        nearest = min(unvisited, key=lambda x: distance.euclidean(cities[last], cities[x]))
        route.append(nearest)
        unvisited.remove(nearest)

    return route + [route[0]]
