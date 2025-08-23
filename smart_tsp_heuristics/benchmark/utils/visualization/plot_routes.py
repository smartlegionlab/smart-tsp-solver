# --------------------------------------------------------
# Copyright © 2025, A.A. Suvorov
# All rights reserved.
# --------------------------------------------------------
from typing import Dict, List
import numpy as np
from matplotlib import pyplot as plt
from smart_tsp_heuristics.benchmark.utils.calculators.length import calculate_length


from itertools import cycle


def plot_routes(cities: np.ndarray, routes: Dict[str, List[int]]):
    plt.figure(figsize=(18, 6))
    colors = cycle(['red', 'green', 'blue', 'purple', 'orange'])

    n_points = len(cities)

    if n_points > 5000:
        display_step = max(1, n_points // 1000)
        route_step = max(1, len(next(iter(routes.values()))) // 500)
        alpha = 0.3
        linewidth = 1
    else:
        display_step = 1
        route_step = 1
        alpha = 0.7
        linewidth = 2

    display_mask = np.zeros(n_points, dtype=bool)
    display_mask[::display_step] = True

    for i, (name, route) in enumerate(routes.items()):
        plt.subplot(1, len(routes), i + 1)

        plt.scatter(cities[display_mask, 0], cities[display_mask, 1],
                    c='gray', alpha=alpha * 0.5, s=10)

        simplified_route = route[::route_step]
        if len(route) > 0:
            simplified_route.append(route[0])

        color = next(colors)
        plt.plot(cities[simplified_route, 0], cities[simplified_route, 1],
                 color=color, linewidth=linewidth, label=name)

        if len(route) > 0:
            plt.scatter(cities[route[0], 0], cities[route[0], 1],
                        c='black', s=100, marker='*', zorder=3)

        plt.title(f"{name}\nPoints: {n_points:,}\nLength: {calculate_length(cities, route):.2f}")
        plt.axis('equal')
        plt.legend()

    plt.tight_layout()
    plt.show()
