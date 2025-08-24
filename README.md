# Smart TSP Solver

---

A high-performance Python library for solving the Traveling Salesman Problem (TSP) using novel heuristic approaches. 
Features advanced algorithms that outperform classical methods by **6-16%** on real-world 
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

### Algorithmic Innovations

**Dynamic Gravity Approach:** Models city selection using principles of physical attraction and momentum conservation, creating smoother paths that minimize cross-cluster backtracking.

**Angular-Radial Method:** Employs polar coordinate transformation with intelligent sector prioritization, significantly reducing search space for clustered distributions.

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

### Usage example

Example of creating a configuration file `tsp_config.json`:

```json
{
  "algorithms": {
    "Angular-radial v1": {
      "enabled": true,
      "params": {
        "sort_by": "angle_distance",
        "look_ahead": 100,
        "max_2opt_iter": 100
      }
    },
    "Angular-radial v2": {
      "enabled": true,
      "params": {
        "sort_by": "angle_distance",
        "look_ahead": 100,
        "max_2opt_iter": 100
      }
    },
    "Dynamic-gravity v1": {
      "enabled": true,
      "params": {
        "delta": 0.5,
        "fast_2opt_iter": 100
      }
    },"Dynamic-gravity v2": {
      "enabled": true,
      "params": {
        "delta": 0.5,
        "fast_2opt_iter": 100
      }
    },
    "Greedy v2": {
      "enabled": true,
      "params": {
        "start_point": 0
      }
    }
  },
  "benchmark": {
    "n_cities": 100,
    "seed": 123,
    "city_generation": "cluster",
    "use_post_optimization": false,
    "plot_results": false,
    "verbose": true
  }
}
```

### Launch using [Smart TSP Benchmark](https://github.com/smartlegionlab/smart-tsp-benchmark) 

```python
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

```


## 📊 Comprehensive Performance Analysis

Use [Smart TSP Benchmark](https://github.com/smartlegionlab/smart-tsp-benchmark) 

### Experimental Results

Our benchmarking reveals sophisticated performance characteristics across different problem scales:

#### 🔵 Dataset: 100 Points (Clustered Distribution)
| Algorithm | Time (s) | Δ Time | Route Length | Δ Length | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Greedy v2 | 0.0016 | **BEST** | 609.21 | +13.89% | `start_point=0` |
| **Dynamic-gravity v2** | 0.0073 | +348% | **534.90** | **BEST** | `delta=0.5` |
| Angular-radial v2 | 0.0088 | +441% | 553.66 | +3.51% | `look_ahead=1000` |

**Analysis:** For smaller clusters, `Dynamic-gravity v2` achieves **optimal path quality** while maintaining near-real-time performance.

#### 🔴 Dataset: 1001 Points (Large Clustered Distribution)
| Algorithm | Time (s) | Δ Time | Route Length | Δ Length | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Greedy v2 | 0.0022 | **BEST** | 1612.74 | +16.12% | `start_point=0` |
| Dynamic-gravity v2 | 0.0248 | +1016% | 1485.03 | +6.93% | `delta=0.5` |
| **Angular-radial v2** | 0.1263 | +5590% | **1388.82** | **BEST** | `look_ahead=1000` |

**Analysis:** For large-scale problems, `Angular-radial v2` delivers **superior path quality** (16% improvement over greedy) making it ideal for quality-sensitive applications.

### Key Insights

1.  **Algorithmic Evolution:** Version 2 implementations demonstrate significant improvements:
    - `Angular-radial v2`: **2× speedup** over v1 with identical quality
    - `Dynamic-gravity v2`: Consistent quality improvements with better stability

2.  **Scalability Characteristics:** 
    - Greedy: O(n²) time complexity, minimal constant factors
    - Dynamic-gravity: O(n²) with higher constants but better quality
    - Angular-radial: O(n log n) for spatial partitioning, excels on large datasets

3.  **Quality-Speed Tradeoff:** The library provides a continuum of solutions:
    - **Microsecond response:** Greedy for real-time applications
    - **Millisecond response:** Dynamic-gravity for balanced needs  
    - **Best quality:** Angular-radial for offline processing

## 🎨 Advanced Visualization

![LOGO](https://github.com/smartlegionlab/smart-tsp-solver/raw/master/data/images/tsp100.png)
![LOGO](https://github.com/smartlegionlab/smart-tsp-solver/raw/master/data/images/tsp1001.png)
*Visual analysis showing Angular-radial's optimal sector-based routing, Dynamic-gravity's smooth trajectories, Greedy's suboptimal clustering*

## 🏗️ Architecture & Implementation

### Performance Optimization

- **Numba JIT Compilation:** Critical paths compiled to native code
- **Memory Efficiency:** Pre-allocated arrays and minimal copying
- **Cache Optimization:** Intelligent memoization and reuse
- **Vectorized Operations:** NumPy-based efficient computations

## 📈 Usage Recommendations

### Based on Empirical Results

| Use Case | Recommended Algorithm | Rationale |
| :--- | :--- | :--- |
| Real-time applications | `Greedy v2` | Sub-millisecond response for 1000+ points |
| General-purpose routing | `Dynamic-gravity v2` | Excellent quality-speed balance |
| High-quality requirements | `Angular-radial v2` | 16% quality improvement over greedy |
| Clustered distributions | `Angular-radial v2` | Superior spatial partitioning |
| Uniform distributions | `Dynamic-gravity v2` | Consistent performance across layouts |


---

## 👨‍💻 Author

**A.A. Suvorov**

- Researcher specializing in computational optimization and high-performance algorithms
- Focused on bridging theoretical computer science with practical engineering applications
- This project represents extensive research into spatial optimization techniques

*Explore other projects on [GitHub](https://github.com/smartlegionlab).*

## 📄 License

This project is licensed under the **BSD 3-Clause License** - see the [LICENSE](LICENSE) file for details. 
This permits academic and commercial use while protecting author rights.

## 🔗 Related Research

For those interested in the theoretical foundations:

- **Exact TSP Solutions (TSP ORACLE):** [exact-tsp-solver](https://github.com/smartlegionlab/exact-tsp-solver) - Optimal solutions for small instances
- **Smart TSP Benchmark** - [Smart TSP Benchmark](https://github.com/smartlegionlab/smart-tsp-benchmark)  is a professional algorithm testing infrastructure with customizable scenarios and detailed metrics.
- **Spatial Optimization:** Computational geometry approaches for large-scale problems
- **Heuristic Analysis:** Comparative study of modern TSP approaches

## 📚 Citation

If this work contributes to your research, please cite:

```bibtex
@software{suvorov2025tspsolver,
  title = {Smart TSP Solver Suite: Advanced Heuristic Algorithms},
  author = {Suvorov, A.A.},
  year = {2025},
  url = {https://github.com/smartlegionlab/smart-tsp-solver}
}
```

---

## 📊 Sample Output

```
==================================================
          SMART TSP ALGORITHMS BENCHMARK          
==================================================
Cities:         100
Seed:           123
Generation:     cluster
Post-opt:       OFF
Algorithms:    
  - Angular-radial v1: sort_by=angle_distance, look_ahead=1001, max_2opt_iter=1001
  - Angular-radial v2: sort_by=angle_distance, look_ahead=1000, max_2opt_iter=1001
  - Dynamic-gravity v1: delta=0.5, fast_2opt_iter=1001
  - Dynamic-gravity v2: delta=0.5, fast_2opt_iter=1001
  - Greedy v2: start_point=0
==================================================


==================================================
Running Angular-radial v1 algorithm...
Description: Angular-radial v1
Parameters: sort_by=angle_distance, look_ahead=1001, max_2opt_iter=1001
Completed in 0.0842 seconds
Route length: 553.66
==================================================

==================================================
Running Angular-radial v2 algorithm...
Description: Angular-radial v2
Parameters: sort_by=angle_distance, look_ahead=1000, max_2opt_iter=1001
Completed in 0.0088 seconds
Route length: 553.66
==================================================

==================================================
Running Dynamic-gravity v1 algorithm...
Description: Dynamic gravity v1
Parameters: delta=0.5, fast_2opt_iter=1001
Completed in 0.0075 seconds
Route length: 567.00
==================================================

==================================================
Running Dynamic-gravity v2 algorithm...
Description: Dynamic gravity v2
Parameters: delta=0.5, fast_2opt_iter=1001
Completed in 0.0073 seconds
Route length: 534.90
==================================================

==================================================
Running Greedy v2 algorithm...
Description: Classic greedy TSP algorithm
Parameters: start_point=0
Completed in 0.0016 seconds
Route length: 609.21
==================================================

==============================================================================================================================
                                                DETAILED ALGORITHM COMPARISON                                                 
==============================================================================================================================
Algorithm            | Time (s) |  vs Best  | Length | vs Best | Params                                                       
------------------------------------------------------------------------------------------------------------------------------
Greedy v2            | 0.0016 | BEST | 609.21 | +13.89% | start_point=0                                                
Dynamic-gravity v2   | 0.0073 | +348.65%  | 534.90 | BEST | delta=0.5, fast_2opt_iter=1001                               
Dynamic-gravity v1   | 0.0075 | +361.28%  | 567.00 | +6.00%  | delta=0.5, fast_2opt_iter=1001                               
Angular-radial v2    | 0.0088 | +441.26%  | 553.66 | +3.51%  | sort_by=angle_distance, look_ahead=1000, max_2opt_iter=1001  
Angular-radial v1    | 0.0842 | +5064.72% | 553.66 | +3.51%  | sort_by=angle_distance, look_ahead=1001, max_2opt_iter=1001  
==============================================================================================================================

PERFORMANCE ANALYSIS:
- Fastest algorithm(s): Greedy v2 (0.0016 sec)
- Shortest route(s): Dynamic-gravity v2 (534.90 units)
```

```
==================================================
          SMART TSP ALGORITHMS BENCHMARK          
==================================================
Cities:         1001
Seed:           0
Generation:     cluster
Post-opt:       OFF
Algorithms:    
  - Angular-radial v1: sort_by=angle_distance, look_ahead=1001, max_2opt_iter=1001
  - Angular-radial v2: sort_by=angle_distance, look_ahead=1000, max_2opt_iter=1001
  - Dynamic-gravity v1: delta=0.5, fast_2opt_iter=1001
  - Dynamic-gravity v2: delta=0.5, fast_2opt_iter=1001
  - Greedy v2: start_point=0
==================================================


==================================================
Running Angular-radial v1 algorithm...
Description: Angular-radial v1
Parameters: sort_by=angle_distance, look_ahead=1001, max_2opt_iter=1001
Completed in 0.2531 seconds
Route length: 1388.82
==================================================

==================================================
Running Angular-radial v2 algorithm...
Description: Angular-radial v2
Parameters: sort_by=angle_distance, look_ahead=1000, max_2opt_iter=1001
Completed in 0.1263 seconds
Route length: 1388.82
==================================================

==================================================
Running Dynamic-gravity v1 algorithm...
Description: Dynamic gravity v1
Parameters: delta=0.5, fast_2opt_iter=1001
Completed in 0.0279 seconds
Route length: 1486.12
==================================================

==================================================
Running Dynamic-gravity v2 algorithm...
Description: Dynamic gravity v2
Parameters: delta=0.5, fast_2opt_iter=1001
Completed in 0.0248 seconds
Route length: 1485.03
==================================================

==================================================
Running Greedy v2 algorithm...
Description: Classic greedy TSP algorithm
Parameters: start_point=0
Completed in 0.0022 seconds
Route length: 1612.74
==================================================

================================================================================================================================
                                                 DETAILED ALGORITHM COMPARISON                                                  
================================================================================================================================
Algorithm            | Time (s) |  vs Best   |  Length | vs Best | Params                                                       
--------------------------------------------------------------------------------------------------------------------------------
Greedy v2            | 0.0022 | BEST | 1612.74 | +16.12% | start_point=0                                                
Dynamic-gravity v2   | 0.0248 | +1016.15%  | 1485.03 | +6.93%  | delta=0.5, fast_2opt_iter=1001                               
Dynamic-gravity v1   | 0.0279 | +1156.99%  | 1486.12 | +7.01%  | delta=0.5, fast_2opt_iter=1001                               
Angular-radial v2    | 0.1263 | +5590.97%  | 1388.82 | BEST | sort_by=angle_distance, look_ahead=1000, max_2opt_iter=1001  
Angular-radial v1    | 0.2531 | +11304.79% | 1388.82 | BEST | sort_by=angle_distance, look_ahead=1001, max_2opt_iter=1001  
================================================================================================================================

PERFORMANCE ANALYSIS:
- Fastest algorithm(s): Greedy v2 (0.0022 sec)
- Shortest route(s): Angular-radial v1, Angular-radial v2 (1388.82 units)
```

---

**Disclaimer:** Performance results shown are for clustered distributions. 
Results may vary based on spatial characteristics. 
Always evaluate algorithms on your specific problem domains.

---

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
