# --------------------------------------------------------
# Copyright © 2025, A.A. Suvorov
# All rights reserved.
# --------------------------------------------------------
from smart_tsp_benchmark.tsp_benchmark import TSPBenchmark, AlgorithmConfig

from smart_tsp_solver.algorithms.angular_radial.v1.angular_radial_v1 import angular_radial_tsp_v1
from smart_tsp_solver.algorithms.angular_radial.v2.angular_radial_v2 import angular_radial_tsp_v2
from smart_tsp_solver.algorithms.dynamic_gravity.v1.dynamic_gravity_v1 import dynamic_gravity_tsp_v1
from smart_tsp_solver.algorithms.dynamic_gravity.v2.dynamic_gravity_v2 import dynamic_gravity_tsp_v2
from smart_tsp_solver.algorithms.other.greedy.v2.greedy_v2 import greedy_tsp_v2


def main():
    config = {
        'n_dots': 1001,
        'seed': 123,
        'dot_generation': 'random',
        'use_post_optimization': False,
        'plot_results': True,
        'verbose': True
    }
    benchmark = TSPBenchmark(config=config)
    benchmark.add_algorithm(
        name='Angular-radial v1',
        config=AlgorithmConfig(
            function=angular_radial_tsp_v1,
            params={
                "sort_by": "angle_distance",
                "look_ahead": 1001,
                "max_2opt_iter": 1001
            },
            post_optimize=True,
            description="Angular-radial v1",
            is_class=False
        )
    )
    benchmark.add_algorithm(
        name='Angular-radial v2',
        config=AlgorithmConfig(
            function=angular_radial_tsp_v2,
            params={
                "sort_by": "angle_distance",
                "look_ahead": 1001,
                "max_2opt_iter": 1001
            },
            post_optimize=True,
            description="Angular-radial v2",
            is_class=False
        )
    )
    benchmark.add_algorithm(
        name='Dynamic-gravity v1',
        config=AlgorithmConfig(
            function=dynamic_gravity_tsp_v1,
            params={
                "delta": 0.5,
                "fast_2opt_iter": 1001
            },
            post_optimize=True,
            description="Dynamic-gravity v1",
            is_class=False
        )
    )
    benchmark.add_algorithm(
        name='Dynamic-gravity v2',
        config=AlgorithmConfig(
            function=dynamic_gravity_tsp_v2,
            params={
                "delta": 0.5,
                "fast_2opt_iter": 1001
            },
            post_optimize=True,
            description="Dynamic-gravity v2",
            is_class=False
        )
    )
    benchmark.add_algorithm(
        name='Greedy v2',
        config=AlgorithmConfig(
            function=greedy_tsp_v2,
            params={},
            post_optimize=False,
            description="Classic greedy TSP algorithm",
            is_class=False,
        )
    )
    benchmark.run_benchmark()


if __name__ == '__main__':
    main()
