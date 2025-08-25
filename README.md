# Smart TSP Solver <sup>v0.1.6</sup>

---

A high-performance Python library for solving the Traveling Salesman Problem (TSP) using novel heuristic approaches. 
Features advanced algorithms that outperform classical methods by **25%** on real-world 
clustered data while maintaining practical computational efficiency.

---

![GitHub top language](https://img.shields.io/github/languages/top/smartlegionlab/smart-tsp-solver)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/smartlegionlab/smart-tsp-solver)](https://github.com/smartlegionlab/smart-tsp-solver/)
[![GitHub](https://img.shields.io/github/license/smartlegionlab/smart-tsp-solver)](https://github.com/smartlegionlab/smart-tsp-solver/blob/master/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/smartlegionlab/smart-tsp-solver?style=social)](https://github.com/smartlegionlab/smart-tsp-solver/)
[![GitHub watchers](https://img.shields.io/github/watchers/smartlegionlab/smart-tsp-solver?style=social)](https://github.com/smartlegionlab/smart-tsp-solver/)
[![GitHub forks](https://img.shields.io/github/forks/smartlegionlab/smart-tsp-solver?style=social)](https://github.com/smartlegionlab/smart-tsp-solver/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/smart-tsp-solver?label=pypi%20downloads)](https://pypi.org/project/smart-tsp-solver/)
[![PyPI](https://img.shields.io/pypi/v/smart-tsp-solver)](https://pypi.org/project/smart-tsp-solver)
[![PyPI - Format](https://img.shields.io/pypi/format/smart-tsp-solver)](https://pypi.org/project/smart-tsp-solver)

---

> **Research-driven design:** This library implements cutting-edge spatial optimization 
> techniques including **dynamic gravitational attraction modeling** and **angular-radial 
> spatial indexing** for intelligent pathfinding.

## 🚀 Features

*   **🧠 Dynamic Gravity Algorithms:** Physics-inspired approach simulating momentum and gravitational attraction for natural, efficient routing
*   **📊 Angular-Radial Methods:** Space-partitioning heuristics with adaptive look-ahead for superior performance on geographical data
*   **⚡ Benchmarking Framework:** Professional-grade testing infrastructure with configurable scenarios and detailed metrics
*   **🦾 High-Performance Core:** Numba JIT compilation with cache optimization for near-native execution speed

## 🔬 Scientific Foundation

---

## 🧠 Algorithmic Innovations

Library implements two advanced heuristic approaches, each tackling the classic speed-quality trade-off in a unique way.

### 🧲 Dynamic Gravity Approach

**Complexity:** `O(n²)`

**Concept:** This algorithm models a physical process of attraction, where the next point is selected based on a combination of proximity and current direction of movement. The `delta` parameter acts as an "inertia coefficient," preventing sharp turns and creating smooth, natural-looking routes.

| Strengths | Ideal Use Case |
| :--- | :--- |
| • Predictable execution time<br>• Consistently high solution quality<br>• Efficient cluster traversal | The balance of speed and quality, processing medium-sized datasets |

### 📐 Angular-Radial Method

**Complexity:** `O(n²)` *with near O(n·log n) practical performance*

**Concept:** A "smart look-ahead" strategy (`look_ahead`). Points are pre-sorted in a polar coordinate system, which drastically narrows the search space for each subsequent choice. This is equivalent to a navigator scanning the nearest sector on the horizon instead of re-examining the entire map every time.

| Strengths | Ideal Use Case |
| :--- | :--- |
| • Best-in-class final route quality<br>• Near-linear practical performance<br>• Exceptional efficiency on clustered data | Offline calculations where route length is critical and tasks require scaling |

### Performance Comparison

| Algorithm | Complexity | Quality | Speed | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy v2** | `O(n²)` | ███░░ | █████ | Real-time, microseconds |
| **Dynamic-gravity v2** | `O(n²)` | █████ | ████░ | Balanced, milliseconds |
| **Angular-radial v2** | `O(n²)`* | █████ | ███░░ | Quality, offline |

*Practical performance approaches O(n·log n) due to spatial heuristics.*
*Worst-case complexity. Practical performance is near O(n·log n) due to spatial heuristics

---

### Benchmarking Methodology

All algorithms are compared against a **highly optimized greedy implementation** featuring:
- Numba JIT compilation with `fastmath` and caching
- Euclidean distance optimization with squared distance comparisons
- Memory-efficient visited node tracking
- Reproducible results through seed-based initialization

This ensures fair comparison against a professionally implemented baseline rather than naive reference implementations.

## 📦 Installation

### Install

```bash
pip install smart-tsp-solver
```

### Example

### Launch using [Smart TSP Benchmark](https://github.com/smartlegionlab/smart-tsp-benchmark)

`pip install smart-tsp-benchmark`

```python
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

```


### 🚀 Example

```bash
git clone https://github.com/smartlegionlab/smart-tsp-solver.git
cd smart-tsp-solver
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Testing with different settings

- change basic settings
- add algorithms for comparison

`python test.py`

## 📊 Comprehensive Performance Analysis

### Experimental Results

### 📊 Smart TSP Algorithms Benchmark Report

Our comprehensive benchmarking reveals a clear performance-quality tradeoff across different problem scales, highlighting the strengths of each algorithm.

#### 🔵 Dataset: 100 Points (Random Distribution)

**Key Insight:** For small-scale problems, the **Dynamic-gravity v2** algorithm demonstrates the best balance, achieving near-optimal path quality while maintaining near-real-time performance.

| Algorithm | Time (s) | Δ vs Best | Route Length | Δ vs Best | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Greedy v2 | **0.0015** | **BASELINE** | 1026.37 | +20.66% | `start_point=0` |
| Dynamic-gravity v1 | 0.0071 | +367% | 850.86 | +0.03% | `delta=0.5` |
| **Dynamic-gravity v2** | 0.0071 | +367% | 856.14 | +0.65% | `delta=0.5` |
| Angular-radial v2 | 0.0086 | +469% | **850.62** | **BASELINE** | `look_ahead=100` |
| Angular-radial v1 | 0.0812 | +5243% | **850.62** | **BASELINE** | `look_ahead=100` |

#### 🔴 Dataset: 1001 Points (Large Random Distribution)

**Key Insight:** For large-scale problems, **Angular-radial v2** becomes the undisputed leader in solution quality (providing a **17.3%** shorter route than the Greedy algorithm). Its acceptable processing time makes it ideal for quality-sensitive offline applications.

| Algorithm | Time (s) | Δ vs Best | Route Length | Δ vs Best | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Greedy v2 | **0.0023** | **BASELINE** | 2985.82 | +17.31% | `start_point=0` |
| Dynamic-gravity v2 | 0.0186 | +696% | 2726.36 | +7.12% | `delta=0.5` |
| Dynamic-gravity v1 | 0.0321 | +1269% | 2837.78 | +11.50% | `delta=0.5` |
| **Angular-radial v2** | 0.1346 | +5647% | **2545.21** | **BASELINE** | `look_ahead=1001` |
| Angular-radial v1 | 0.2555 | +10809% | **2545.21** | **BASELINE** | `look_ahead=1001` |

### 🎯 Key Insights & Analysis

1.  **Algorithm Evolution (v1 vs. v2):**
    *   **Angular-radial v2** shows a **~2x speedup** over v1 while delivering identical, best-in-class route quality.
    *   **Dynamic-gravity v2** also demonstrates a significant speed improvement (nearly 2x on 1001 points) over v1, maintaining consistently high solution quality with better stability.

2.  **Algorithm Characteristics:**
    *   **Greedy v2:** Extremely fast (`O(n²)`), ideal for real-time applications, but sacrifices solution quality (+17-20% longer routes).
    *   **Dynamic-gravity:** Offers significantly better quality than the greedy approach. It has `O(n²)` complexity with higher constant factors, making it the optimal choice for medium-sized problems where a balance between speed and quality is required.
    *   **Angular-radial:** The quality leader. Its use of spatial partitioning (`O(n log n)`) allows it to scale best on large datasets. It is the recommended choice for offline processing where final route cost is the primary concern.

3.  **Practical Recommendations:**
    The library provides a continuum of solutions for different use cases:
    *   **Microsecond Response:** **Greedy v2** for interactive and real-time systems.
    *   **Millisecond Response:** **Dynamic-gravity v2** for balanced needs and medium-scale problems.
    *   **Best Quality:** **Angular-radial v2** for final calculations and offline processing where route cost is paramount.

## 🎨 Advanced Visualization

![LOGO](https://github.com/smartlegionlab/smart-tsp-solver/raw/master/data/images/tsp100.png)
![LOGO](https://github.com/smartlegionlab/smart-tsp-solver/raw/master/data/images/tsp1001.png)
![LOGO](https://github.com/smartlegionlab/smart-tsp-solver/raw/master/data/images/tsp_25_08_25.png)
*Visual analysis showing Angular-radial's optimal sector-based routing, Dynamic-gravity's smooth trajectories, Greedy's suboptimal clustering*

## 🏗️ Architecture & Implementation

### Performance Optimization

- **Numba JIT Compilation:** Critical paths compiled to native code
- **Memory Efficiency:** Pre-allocated arrays and minimal copying
- **Cache Optimization:** Intelligent memoization and reuse
- **Vectorized Operations:** NumPy-based efficient computations

---

## 👨‍💻 Author

**Alexander Suvorov**

- Researcher specializing in computational optimization and high-performance algorithms
- Focused on bridging theoretical computer science with practical engineering applications
- This project represents extensive research into spatial optimization techniques

*Explore other projects on [GitHub](https://github.com/smartlegionlab).*

## 🔗 Related Research

For those interested in the theoretical foundations:

- **Exact TSP Solutions (TSP ORACLE):** [exact-tsp-solver](https://github.com/smartlegionlab/exact-tsp-solver) - Optimal solutions for small instances
- **Smart TSP Benchmark** - [Smart TSP Benchmark](https://github.com/smartlegionlab/smart-tsp-benchmark)  is a professional algorithm testing infrastructure with customizable scenarios and detailed metrics.
- **Spatial Optimization:** Computational geometry approaches for large-scale problems
- **Heuristic Analysis:** Comparative study of modern TSP approaches

---

## 📊 Sample Output

```
==================================================
          SMART TSP ALGORITHMS BENCHMARK          
==================================================
Dots:           100
Seed:           123
Generation:     random
Post-opt:       OFF
Algorithms:    
  - Angular-radial v1: 
  - Angular-radial v2: 
  - Dynamic-gravity v1: 
  - Dynamic-gravity v2: 
  - Greedy v2: 
  - Hierarchical TSP: 
==================================================


==================================================
Running Angular-radial v1 algorithm...
Description: Angular-radial v1
Parameters: 
Completed in 2.5245 seconds
Route length: 850.62
==================================================

==================================================
Running Angular-radial v2 algorithm...
Description: Angular-radial v2
Parameters: 
Completed in 0.6046 seconds
Route length: 850.62
==================================================

==================================================
Running Dynamic-gravity v1 algorithm...
Description: Dynamic-gravity v1
Parameters: 
Completed in 0.5134 seconds
Route length: 850.86
==================================================

==================================================
Running Dynamic-gravity v2 algorithm...
Description: Dynamic-gravity v2
Parameters: 
Completed in 0.5180 seconds
Route length: 856.14
==================================================

==================================================
Running Greedy v2 algorithm...
Description: Classic greedy TSP algorithm
Parameters: 
Completed in 0.1141 seconds
Route length: 1026.37
==================================================

==================================================
Running Hierarchical TSP algorithm...
Description: Hierarchical clustering TSP solver
Parameters: 
Completed in 0.0273 seconds
Route length: 822.31
==================================================

=============================================================================================================================
                                                DETAILED ALGORITHM COMPARISON                                                
=============================================================================================================================
Algorithm            | Time (s) |  vs Best  |  Length | vs Best | Params                                                     
-----------------------------------------------------------------------------------------------------------------------------
Hierarchical TSP     | 0.0273 | BEST | 822.31 | BEST | cluster_size=100, post_optimize=True                       
Greedy v2            | 0.1141 | +318.52%  | 1026.37 | +24.81% |                                                            
Dynamic-gravity v1   | 0.5134 | +1782.85% |  850.86 | +3.47%  | delta=0.5, fast_2opt_iter=100                              
Dynamic-gravity v2   | 0.5180 | +1799.68% |  856.14 | +4.11%  | delta=0.5, fast_2opt_iter=100                              
Angular-radial v2    | 0.6046 | +2117.44% |  850.62 | +3.44%  | sort_by=angle_distance, look_ahead=100, max_2opt_iter=100  
Angular-radial v1    | 2.5245 | +9158.34% |  850.62 | +3.44%  | sort_by=angle_distance, look_ahead=100, max_2opt_iter=100  
=============================================================================================================================

PERFORMANCE ANALYSIS:
- Fastest algorithm(s): Hierarchical TSP (0.0273 sec)
- Shortest route(s): Hierarchical TSP (822.31 units)

⭐️ BEST BALANCED: Hierarchical TSP (fastest and shortest)
```

```
==================================================
          SMART TSP ALGORITHMS BENCHMARK          
==================================================
Dots:           50
Seed:           123
Generation:     random
Post-opt:       OFF
Algorithms:    
  - Angular-radial v1: 
  - Angular-radial v2: 
  - Dynamic-gravity v1: 
  - Dynamic-gravity v2: 
  - Greedy v2: 
==================================================


==================================================
Running Angular-radial v1 algorithm...
Description: Angular-radial v1
Parameters: 
Completed in 0.0798 seconds
Route length: 658.16
==================================================

==================================================
Running Angular-radial v2 algorithm...
Description: Angular-radial v2
Parameters: 
Completed in 0.0082 seconds
Route length: 658.16
==================================================

==================================================
Running Dynamic-gravity v1 algorithm...
Description: Dynamic-gravity v1
Parameters: 
Completed in 0.0067 seconds
Route length: 582.13
==================================================

==================================================
Running Dynamic-gravity v2 algorithm...
Description: Dynamic-gravity v2
Parameters: 
Completed in 0.0065 seconds
Route length: 577.06
==================================================

==================================================
Running Greedy v2 algorithm...
Description: Classic greedy TSP algorithm
Parameters: 
Completed in 0.0016 seconds
Route length: 720.50
==================================================

============================================================================================================================
                                               DETAILED ALGORITHM COMPARISON                                                
============================================================================================================================
Algorithm            | Time (s) |  vs Best  | Length | vs Best | Params                                                     
----------------------------------------------------------------------------------------------------------------------------
Greedy v2            | 0.0016 | BEST | 720.50 | +24.86% |                                                            
Dynamic-gravity v2   | 0.0065 | +321.25%  | 577.06 | BEST | delta=0.5, fast_2opt_iter=100                              
Dynamic-gravity v1   | 0.0067 | +328.66%  | 582.13 | +0.88%  | delta=0.5, fast_2opt_iter=100                              
Angular-radial v2    | 0.0082 | +426.35%  | 658.16 | +14.05% | sort_by=angle_distance, look_ahead=100, max_2opt_iter=100  
Angular-radial v1    | 0.0798 | +5035.05% | 658.16 | +14.05% | sort_by=angle_distance, look_ahead=100, max_2opt_iter=100  
============================================================================================================================

PERFORMANCE ANALYSIS:
- Fastest algorithm(s): Greedy v2 (0.0016 sec)
- Shortest route(s): Dynamic-gravity v2 (577.06 units)
```

```
==================================================
          SMART TSP ALGORITHMS BENCHMARK          
==================================================
Dots:           50
Seed:           123
Generation:     cluster
Post-opt:       OFF
Algorithms:    
  - Angular-radial v1: 
  - Angular-radial v2: 
  - Dynamic-gravity v1: 
  - Dynamic-gravity v2: 
  - Greedy v2: 
==================================================


==================================================
Running Angular-radial v1 algorithm...
Description: Angular-radial v1
Parameters: 
Completed in 0.0798 seconds
Route length: 519.29
==================================================

==================================================
Running Angular-radial v2 algorithm...
Description: Angular-radial v2
Parameters: 
Completed in 0.0081 seconds
Route length: 519.29
==================================================

==================================================
Running Dynamic-gravity v1 algorithm...
Description: Dynamic-gravity v1
Parameters: 
Completed in 0.0066 seconds
Route length: 495.87
==================================================

==================================================
Running Dynamic-gravity v2 algorithm...
Description: Dynamic-gravity v2
Parameters: 
Completed in 0.0066 seconds
Route length: 495.87
==================================================

==================================================
Running Greedy v2 algorithm...
Description: Classic greedy TSP algorithm
Parameters: 
Completed in 0.0015 seconds
Route length: 621.53
==================================================

============================================================================================================================
                                               DETAILED ALGORITHM COMPARISON                                                
============================================================================================================================
Algorithm            | Time (s) |  vs Best  | Length | vs Best | Params                                                     
----------------------------------------------------------------------------------------------------------------------------
Greedy v2            | 0.0015 | BEST | 621.53 | +25.34% |                                                            
Dynamic-gravity v2   | 0.0066 | +331.67%  | 495.87 | BEST | delta=0.5, fast_2opt_iter=100                              
Dynamic-gravity v1   | 0.0066 | +333.80%  | 495.87 | BEST | delta=0.5, fast_2opt_iter=100                              
Angular-radial v2    | 0.0081 | +431.08%  | 519.29 | +4.72%  | sort_by=angle_distance, look_ahead=100, max_2opt_iter=100  
Angular-radial v1    | 0.0798 | +5150.37% | 519.29 | +4.72%  | sort_by=angle_distance, look_ahead=100, max_2opt_iter=100  
============================================================================================================================

PERFORMANCE ANALYSIS:
- Fastest algorithm(s): Greedy v2 (0.0015 sec)
- Shortest route(s): Dynamic-gravity v1, Dynamic-gravity v2 (495.87 units)
```

```
==================================================
          SMART TSP ALGORITHMS BENCHMARK          
==================================================
Dots:           100
Seed:           123
Generation:     random
Post-opt:       OFF
Algorithms:    
  - Angular-radial v1: 
  - Angular-radial v2: 
  - Dynamic-gravity v1: 
  - Dynamic-gravity v2: 
  - Greedy v2: 
==================================================


==================================================
Running Angular-radial v1 algorithm...
Description: Angular-radial v1
Parameters: 
Completed in 0.0812 seconds
Route length: 850.62
==================================================

==================================================
Running Angular-radial v2 algorithm...
Description: Angular-radial v2
Parameters: 
Completed in 0.0086 seconds
Route length: 850.62
==================================================

==================================================
Running Dynamic-gravity v1 algorithm...
Description: Dynamic-gravity v1
Parameters: 
Completed in 0.0071 seconds
Route length: 850.86
==================================================

==================================================
Running Dynamic-gravity v2 algorithm...
Description: Dynamic-gravity v2
Parameters: 
Completed in 0.0071 seconds
Route length: 856.14
==================================================

==================================================
Running Greedy v2 algorithm...
Description: Classic greedy TSP algorithm
Parameters: 
Completed in 0.0015 seconds
Route length: 1026.37
==================================================

=============================================================================================================================
                                                DETAILED ALGORITHM COMPARISON                                                
=============================================================================================================================
Algorithm            | Time (s) |  vs Best  |  Length | vs Best | Params                                                     
-----------------------------------------------------------------------------------------------------------------------------
Greedy v2            | 0.0015 | BEST | 1026.37 | +20.66% |                                                            
Dynamic-gravity v1   | 0.0071 | +367.32%  |  850.86 | +0.03%  | delta=0.5, fast_2opt_iter=100                              
Dynamic-gravity v2   | 0.0071 | +367.38%  |  856.14 | +0.65%  | delta=0.5, fast_2opt_iter=100                              
Angular-radial v2    | 0.0086 | +468.54%  | 850.62 | BEST | sort_by=angle_distance, look_ahead=100, max_2opt_iter=100  
Angular-radial v1    | 0.0812 | +5243.04% | 850.62 | BEST | sort_by=angle_distance, look_ahead=100, max_2opt_iter=100  
=============================================================================================================================

PERFORMANCE ANALYSIS:
- Fastest algorithm(s): Greedy v2 (0.0015 sec)
- Shortest route(s): Angular-radial v1, Angular-radial v2 (850.62 units)
```

```
==================================================
          SMART TSP ALGORITHMS BENCHMARK          
==================================================
Dots:           1001
Seed:           123
Generation:     random
Post-opt:       OFF
Algorithms:    
  - Angular-radial v1: 
  - Angular-radial v2: 
  - Dynamic-gravity v1: 
  - Dynamic-gravity v2: 
  - Greedy v2: 
==================================================


==================================================
Running Angular-radial v1 algorithm...
Description: Angular-radial v1
Parameters: 
Completed in 0.2555 seconds
Route length: 2545.21
==================================================

==================================================
Running Angular-radial v2 algorithm...
Description: Angular-radial v2
Parameters: 
Completed in 0.1346 seconds
Route length: 2545.21
==================================================

==================================================
Running Dynamic-gravity v1 algorithm...
Description: Dynamic-gravity v1
Parameters: 
Completed in 0.0321 seconds
Route length: 2837.78
==================================================

==================================================
Running Dynamic-gravity v2 algorithm...
Description: Dynamic-gravity v2
Parameters: 
Completed in 0.0186 seconds
Route length: 2726.36
==================================================

==================================================
Running Greedy v2 algorithm...
Description: Classic greedy TSP algorithm
Parameters: 
Completed in 0.0023 seconds
Route length: 2985.82
==================================================

================================================================================================================================
                                                 DETAILED ALGORITHM COMPARISON                                                  
================================================================================================================================
Algorithm            | Time (s) |  vs Best   |  Length | vs Best | Params                                                       
--------------------------------------------------------------------------------------------------------------------------------
Greedy v2            | 0.0023 | BEST | 2985.82 | +17.31% |                                                              
Dynamic-gravity v2   | 0.0186 |  +695.63%  | 2726.36 | +7.12%  | delta=0.5, fast_2opt_iter=1001                               
Dynamic-gravity v1   | 0.0321 | +1269.12%  | 2837.78 | +11.50% | delta=0.5, fast_2opt_iter=1001                               
Angular-radial v2    | 0.1346 | +5646.96%  | 2545.21 | BEST | sort_by=angle_distance, look_ahead=1001, max_2opt_iter=1001  
Angular-radial v1    | 0.2555 | +10808.60% | 2545.21 | BEST | sort_by=angle_distance, look_ahead=1001, max_2opt_iter=1001  
================================================================================================================================

PERFORMANCE ANALYSIS:
- Fastest algorithm(s): Greedy v2 (0.0023 sec)
- Shortest route(s): Angular-radial v1, Angular-radial v2 (2545.21 units)
```

---

**Disclaimer:** Performance results shown are for clustered/random distributions. 
Results may vary based on spatial characteristics. 
Always evaluate algorithms on your specific problem domains.

---

## 📜 Licensing

This project is offered under a dual-licensing model.

### 🆓 Option 1: BSD 3-Clause License (for Non-Commercial Use)
This license is **free of charge** and allows you to use the software for:
- Personal and educational purposes
- Academic research and open-source projects
- Evaluation and testing

**Important:** Any use by a commercial organization or for commercial purposes (including internal development and prototyping) requires a commercial license.

### 💼 Option 2: Commercial License (for Commercial Use)
A commercial license is **required** for:
- Integrating this software into proprietary products
- Using it in internal operations within a company
- SaaS and hosted services that incorporate this software

**Important:** The commercial license provides usage rights but **does not include any indemnification or liability**. The software is provided "AS IS" without any warranties as described in the full license agreement.

**To obtain a commercial license,** please contact us directly at:  
📧 **smartlegiondev@gmail.com**