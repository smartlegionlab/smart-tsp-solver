# --------------------------------------------------------
# Copyright © 2025, A.A. Suvorov
# All rights reserved.
# --------------------------------------------------------
import time
import json
import os
from dataclasses import dataclass
from typing import Dict, Callable, Any, Union, List

from smart_tsp_heuristics.algorithms.angular_radial.v1.angular_radial_v1 import angular_radial_tsp_v1
from smart_tsp_heuristics.algorithms.angular_radial.v2.angular_radial_v2 import angular_radial_tsp_v2
from smart_tsp_heuristics.algorithms.dynamic_gravity.v1.dynamic_gravity_v1 import dynamic_gravity_tsp_v1
from smart_tsp_heuristics.algorithms.dynamic_gravity.v2.dynamic_gravity_v2 import dynamic_gravity_tsp_v2
from smart_tsp_heuristics.algorithms.other.greedy.v2.greedy_v2 import greedy_tsp_v2
from smart_tsp_heuristics.benchmark.utils.calculators.length import calculate_length
from smart_tsp_heuristics.benchmark.utils.generators.cities import generate_cities
from smart_tsp_heuristics.benchmark.utils.optimizers.fast_opt import fast_post_optimize
from smart_tsp_heuristics.benchmark.utils.visualization.plot_routes import plot_routes


class BenchmarkStep:

    def execute(self, benchmark: 'TSPBenchmark', results: Dict):
        raise NotImplementedError


class CityGenerationStep(BenchmarkStep):

    def execute(self, benchmark: 'TSPBenchmark', results: Dict):
        benchmark.cities = generate_cities(
            benchmark.benchmark_config['n_cities'],
            benchmark.benchmark_config['seed'],
            method=benchmark.benchmark_config['city_generation']
        )


class AlgorithmExecutionStep(BenchmarkStep):

    def execute(self, benchmark: 'TSPBenchmark', results: Dict):
        for name, config in benchmark.algorithms.items():
            if not config.enabled:
                continue

            benchmark._print_algorithm_start(name, config)

            start_time = time.perf_counter()
            route = benchmark._execute_algorithm(config)
            route = benchmark._apply_post_optimization(config, route)
            exec_time = time.perf_counter() - start_time
            route_length = calculate_length(benchmark.cities, route)

            results[name] = benchmark._create_result(route, exec_time, route_length, config)
            benchmark._print_algorithm_end(exec_time, route_length)


class VisualizationStep(BenchmarkStep):

    def execute(self, benchmark: 'TSPBenchmark', results: Dict):
        if benchmark.benchmark_config['plot_results']:
            plot_routes(benchmark.cities, {k: v['route'] for k, v in results.items()})


class SummaryStep(BenchmarkStep):

    def execute(self, benchmark: 'TSPBenchmark', results: Dict):
        if benchmark.benchmark_config['verbose']:
            benchmark._print_comparison_table(results)


@dataclass
class AlgorithmConfig:
    function: Union[Callable, type]
    params: Dict[str, Any]
    post_optimize: bool = True
    enabled: bool = True
    description: str = ""
    param_description: str = ""
    is_class: bool = False
    _manually_configured: bool = False

    def mark_as_manual(self):
        self._manually_configured = True

    def was_manually_configured(self) -> bool:
        return self._manually_configured


class TSPBenchmark:
    CONFIG_FILE = os.path.abspath('tsp_config.json')
    DEFAULT_CONFIG = {
        'benchmark': {
            'n_cities': 1000,
            'seed': 777,
            'city_generation': 'random',
            'use_post_optimization': False,
            'plot_results': True,
            'verbose': True
        },
        'algorithms': {
            'Angular-radial v1': {
                'enabled': True,
                'params': {
                    'sort_by': 'angle_distance',
                    'look_ahead': 100,
                }
            },
            'Angular-radial v2': {
                'enabled': True,
                'params': {
                    'sort_by': 'angle_distance',
                    'look_ahead': 100,
                }
            },
            "Dynamic-gravity v1": {
                "enabled": True,
                "params": {
                    "delta": 0.5
                }
            },
            "Dynamic-gravity v2": {
                "enabled": True,
                "params": {
                    "delta": 0.5
                }
            },
            'Greedy v2': {
                'enabled': True,
                'params': {
                    'start_point': 0
                }
            }
        }
    }

    def __init__(self):
        self.cities = None
        self._init_algorithms()
        self.benchmark_config = self.DEFAULT_CONFIG['benchmark'].copy()
        self._load_config()
        self._init_benchmark_steps()

    def _init_algorithms(self):
        self.algorithms = {
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

    def _init_benchmark_steps(self):
        self.benchmark_steps = [
            CityGenerationStep(),
            AlgorithmExecutionStep(),
            VisualizationStep(),
            SummaryStep()
        ]

    def _load_config(self):
        config = self.DEFAULT_CONFIG.copy()

        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    file_config = json.load(f)
                config = self._deep_update(config, file_config)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in config file {self.CONFIG_FILE}")
            except Exception as e:
                print(f"Error loading config: {str(e)}")

        self._update_algorithms_from_config(config)
        self.benchmark_config.update(config['benchmark'])

    def _deep_update(self, original, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in original:
                original[key] = self._deep_update(original[key], value)
            else:
                original[key] = value
        return original

    def _update_algorithms_from_config(self, config):
        for algo_name, algo_cfg in config['algorithms'].items():
            if algo_name in self.algorithms and not self.algorithms[algo_name].was_manually_configured():
                self.algorithms[algo_name].enabled = algo_cfg.get('enabled', True)
                params = algo_cfg.get('params', {}).copy()
                params.pop('is_class', None)
                self.algorithms[algo_name].params.update(params)
                self._update_param_description(algo_name)

    def _update_param_description(self, algo_name):
        desc = ", ".join(f"{k}={v}" for k, v in self.algorithms[algo_name].params.items())
        self.algorithms[algo_name].param_description = desc

    def run_benchmark(self) -> Dict:
        results = {}

        if self.benchmark_config['verbose']:
            self._print_benchmark_header()

        for step in self.benchmark_steps:
            step.execute(self, results)

        return results

    def _print_benchmark_header(self):
        cfg = self.benchmark_config
        print("\n" + "=" * 50)
        print("SMART TSP ALGORITHMS BENCHMARK".center(50))
        print("=" * 50)
        print(f"{'Cities:':<15} {cfg['n_cities']}")
        print(f"{'Seed:':<15} {cfg['seed']}")
        print(f"{'Generation:':<15} {cfg['city_generation']}")
        print(f"{'Post-opt:':<15} {'ON' if cfg['use_post_optimization'] else 'OFF'}")

        print(f"{'Algorithms:':<15}")
        for name, cfg in self.algorithms.items():
            if cfg.enabled:
                print(f"  - {name}: {cfg.param_description}")
        print("=" * 50 + "\n")

    def _print_algorithm_start(self, name: str, config: AlgorithmConfig):
        if self.benchmark_config['verbose']:
            print(f"\n{'=' * 50}")
            print(f"Running {name} algorithm...")
            print(f"Description: {config.description}")
            print(f"Parameters: {config.param_description}")

    def _execute_algorithm(self, config: AlgorithmConfig) -> List[int]:
        if config.is_class:
            solver = config.function(**config.params)
            return solver.solve(self.cities)
        return config.function(self.cities, **config.params)

    def _apply_post_optimization(self, config: AlgorithmConfig, route: List[int]) -> List[int]:
        if self.benchmark_config['use_post_optimization'] and config.post_optimize:
            return fast_post_optimize(self.cities, route)
        return route

    def _create_result(self, route: List[int], exec_time: float, route_length: float, config: AlgorithmConfig) -> Dict:
        return {
            'route': route,
            'time': exec_time,
            'length': route_length,
            'cities': self.benchmark_config['n_cities'],
            'params': config.params
        }

    def _print_algorithm_end(self, exec_time: float, route_length: float):
        if self.benchmark_config['verbose']:
            print(f"Completed in {exec_time:.4f} seconds")
            print(f"Route length: {route_length:.2f}")
            print(f"{'=' * 50}")

    def _print_comparison_table(self, results: Dict):
        if not results:
            return

        best_time = min(r['time'] for r in results.values())
        best_length = min(r['length'] for r in results.values())

        table_rows = []
        color_codes = {
            'green': '\033[92m',
            'reset': '\033[0m'
        }

        for name, data in sorted(results.items(), key=lambda x: (x[1]['time'], x[1]['length'])):
            time_plain = f"{data['time']:.4f}"
            length_plain = f"{data['length']:.2f}"
            time_diff_plain = "BEST" if data['time'] == best_time else f"+{(data['time'] / best_time - 1) * 100:.2f}%"
            length_diff_plain = "BEST" if data[
                                              'length'] == best_length else f"+{(data['length'] / best_length - 1) * 100:.2f}%"
            params_plain = ", ".join(f"{k}={v}" for k, v in data['params'].items())

            time_str = f"{color_codes['green']}{time_plain}{color_codes['reset']}" if data[
                                                                                          'time'] == best_time else time_plain
            length_str = f"{color_codes['green']}{length_plain}{color_codes['reset']}" if data[
                                                                                              'length'] == best_length else length_plain
            time_diff = f"{color_codes['green']}{time_diff_plain}{color_codes['reset']}" if data[
                                                                                                'time'] == best_time else time_diff_plain
            length_diff = f"{color_codes['green']}{length_diff_plain}{color_codes['reset']}" if data['length'] == best_length else length_diff_plain

            table_rows.append({
                'name': name,
                'time': (time_str, time_plain),
                'time_diff': (time_diff, time_diff_plain),
                'length': (length_str, length_plain),
                'length_diff': (length_diff, length_diff_plain),
                'params': params_plain
            })

        col_widths = {
            'name': max(len(row['name']) for row in table_rows) + 2,
            'time': max(len(row['time'][1]) for row in table_rows),
            'time_diff': max(len(row['time_diff'][1]) for row in table_rows),
            'length': max(len(row['length'][1]) for row in table_rows),
            'length_diff': max(len(row['length_diff'][1]) for row in table_rows),
            'params': max(len(row['params']) for row in table_rows) + 2
        }

        row_fmt = (
            "{name:<{name_w}} | "
            "{time:>{time_w}} | "
            "{time_diff:^{time_diff_w}} | "
            "{length:>{length_w}} | "
            "{length_diff:^{length_diff_w}} | "
            "{params:<{params_w}}"
        )

        header = row_fmt.format(
            name="Algorithm",
            name_w=col_widths['name'],
            time="Time (s)",
            time_w=col_widths['time'],
            time_diff="vs Best",
            time_diff_w=col_widths['time_diff'],
            length="Length",
            length_w=col_widths['length'],
            length_diff="vs Best",
            length_diff_w=col_widths['length_diff'],
            params="Params",
            params_w=col_widths['params']
        )

        separator = "-" * len(header)
        full_width = len(header)

        print("\n" + "=" * full_width)
        print("DETAILED ALGORITHM COMPARISON".center(full_width))
        print("=" * full_width)
        print(header)
        print(separator)

        for row in table_rows:
            print(row_fmt.format(
                name=row['name'],
                name_w=col_widths['name'],
                time=row['time'][0],
                time_w=col_widths['time'],
                time_diff=row['time_diff'][0],
                time_diff_w=col_widths['time_diff'],
                length=row['length'][0],
                length_w=col_widths['length'],
                length_diff=row['length_diff'][0],
                length_diff_w=col_widths['length_diff'],
                params=row['params'],
                params_w=col_widths['params']
            ))

        print("=" * full_width + "\n")
        self._print_performance_analysis(results, best_time, best_length)

    def _format_value(self, value, best_value, format_spec):
        if value == best_value:
            return f"\033[92m{value:{format_spec}}\033[0m"
        return f"{value:{format_spec}}"

    def _format_diff(self, value, best_value):
        if value == best_value:
            return "\033[92mBEST\033[0m"
        diff = (value / best_value - 1) * 100
        return f"+{diff:.2f}%"

    def _print_table_header(self, col_widths):
        headers = [
            "Algorithm",
            "Time (s)",
            "vs Best",
            "Length",
            "vs Best",
            "Params"
        ]

        header = " | ".join(f"{h:<{w}}" if i == 0 or i == len(headers) - 1 else f"{h:^{w}}"
                            for i, (h, w) in enumerate(zip(headers, col_widths)))

        separator = "-" * len(header)
        full_width = sum(col_widths) + len(col_widths) * 3 - 1

        print("\n" + "=" * full_width)
        print("DETAILED ALGORITHM COMPARISON".center(full_width))
        print("=" * full_width)
        print(header)
        print(separator)

    def _print_table_row(self, row, col_widths):
        formatted_row = (
            f"{row[0]:<{col_widths[0]}} | "
            f"{row[1]:>{col_widths[1]}} | "
            f"{row[2]:^{col_widths[2]}} | "
            f"{row[3]:>{col_widths[3]}} | "
            f"{row[4]:^{col_widths[4]}} | "
            f"{row[5]:<{col_widths[5]}}"
        )
        print(formatted_row)

    def _print_table_footer(self, col_widths):
        full_width = sum(col_widths) + len(col_widths) * 3 - 1
        print("=" * full_width + "\n")

    def _print_performance_analysis(self, results, best_time, best_length):
        time_leaders = [name for name, data in results.items() if data['time'] == best_time]
        length_leaders = [name for name, data in results.items() if data['length'] == best_length]

        print("PERFORMANCE ANALYSIS:")
        print(f"- Fastest algorithm(s): {', '.join(time_leaders)} ({best_time:.4f} sec)")
        print(f"- Shortest route(s): {', '.join(length_leaders)} ({best_length:.2f} units)")

        if set(time_leaders) == set(length_leaders):
            print(f"\n⭐️ BEST BALANCED: {', '.join(time_leaders)} (fastest and shortest)")

    def set_algorithm_params(self, algo_name: str, **params):
        if algo_name in self.algorithms:
            params.pop('is_class', None)
            self.algorithms[algo_name].params.update(params)
            self._update_param_description(algo_name)
            self.algorithms[algo_name].mark_as_manual()

    def enable_algorithm(self, name: str, enable: bool = True):
        if name in self.algorithms:
            self.algorithms[name].enabled = enable
            self.algorithms[name].mark_as_manual()

    def add_algorithm(self, name: str, config: AlgorithmConfig):
        self.algorithms[name] = config


def main():
    benchmark = TSPBenchmark()
    benchmark.run_benchmark()
