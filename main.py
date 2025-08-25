# Copyright (©) 2025, Alexander Suvorov. All rights reserved.
from smart_tsp_benchmark.tsp_benchmark import TSPBenchmark, AlgorithmConfig

from smart_tsp_solver import hierarchical_tsp_solver_v2
from smart_tsp_solver.algorithms.angular_radial.v1 import angular_radial_tsp_v1
from smart_tsp_solver.algorithms.angular_radial.v2 import angular_radial_tsp_v2
from smart_tsp_solver.algorithms.dynamic_gravity.v1 import dynamic_gravity_tsp_v1
from smart_tsp_solver.algorithms.dynamic_gravity.v2 import dynamic_gravity_tsp_v2
from smart_tsp_solver.algorithms.other.greedy.v2 import greedy_tsp_v2


def main():
    config = {
        'n_dots': 100,
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
                "look_ahead": 100,
                "max_2opt_iter": 100
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
                "look_ahead": 100,
                "max_2opt_iter": 100
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
                "fast_2opt_iter": 100
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
                "fast_2opt_iter": 100
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
    benchmark.add_algorithm(
        name='Hierarchical TSP',
        config=AlgorithmConfig(
            function=hierarchical_tsp_solver_v2,
            params={
                "cluster_size": 100,
                "post_optimize": True
            },
            post_optimize=False,
            description="Hierarchical clustering TSP solver",
            is_class=False
        )
    )
    benchmark.run_benchmark()


if __name__ == '__main__':
    main()
