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
    algorithms = {
        'Angular-radial v1': AlgorithmConfig(
            function=angular_radial_tsp_v1,
            params={},
            post_optimize=True,
            description="Angular-radial v1",
            is_class=False
        ),
        'Angular-radial v2': AlgorithmConfig(
            function=angular_radial_tsp_v2,
            params={},
            post_optimize=True,
            description="Angular-radial v2",
            is_class=False
        ),
        'Dynamic-gravity v1': AlgorithmConfig(
            function=dynamic_gravity_tsp_v1,
            params={},
            post_optimize=True,
            description="Dynamic gravity v1",
            is_class=False,
        ),
        'Dynamic-gravity v2': AlgorithmConfig(
            function=dynamic_gravity_tsp_v2,
            params={},
            post_optimize=True,
            description="Dynamic gravity v2",
            is_class=False,
        ),
        'Greedy v2': AlgorithmConfig(
            function=greedy_tsp_v2,
            params={},
            post_optimize=False,
            description="Classic greedy TSP algorithm",
            is_class=False,
        ),
    }
    benchmark = TSPBenchmark(config_path='tsp_config.json')
    benchmark.algorithms = algorithms
    benchmark.run_benchmark()


if __name__ == '__main__':
    main()
