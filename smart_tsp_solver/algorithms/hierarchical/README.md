# Hierarchical TSP Solver

A traveling salesman problem (TSP) solver using hierarchical decomposition and metaheuristics.

---

## 🧠 Algorithm Overview

### Core Philosophy: Divide-and-Conquer with Geometric Intelligence

The solver employs a multi-level hierarchical approach that mirrors 
human problem-solving strategies for large-scale routing:

1. **Spatial Decomposition**: Recursively partition the problem into manageable clusters
2. **Local Optimization**: Solve subproblems optimally within each cluster
3. **Global Integration**: Intelligently combine local solutions into a global route
4. **Refinement**: Apply local search to polish the final solution