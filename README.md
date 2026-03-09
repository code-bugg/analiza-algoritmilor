# Lab 3: BFS and DFS Empirical Analysis

## Overview

This laboratory project provides a comprehensive empirical analysis of **Breadth-First Search (BFS)** and **Depth-First Search (DFS)** graph traversal algorithms. It compares original implementations against optimized variants across various graph sizes, measuring both execution time and memory consumption.

## Project Contents

- **main.py** — The complete benchmark suite containing all implementations and visualization tools
- **report3.tex** — LaTeX source for the detailed lab report
- **report3.pdf** — Compiled PDF report with results and analysis
- **Generated outputs:**
  - `viz_bfs.png` — BFS level-by-level traversal visualization
  - `viz_dfs.png` — DFS depth-first traversal visualization  
  - `plot_all_comparison.png` — Comparison of all four algorithm variants
  - `plot_per_algorithm.png` — Per-algorithm performance comparison

## Algorithm Implementations

### BFS (Breadth-First Search)

**Original Implementation:**
- Uses a standard `set()` for tracking visited nodes
- Marks nodes as visited upon enqueueing
- Time complexity: O(V + E)
- Space complexity: O(V)

**Optimized Implementation:**
- Replaces hash-set with `bytearray` bitmap for O(1) direct array access
- Eliminates hashing overhead on integer node IDs
- Same asymptotic complexity but significantly faster in practice
- Better cache locality

### DFS (Depth-First Search)

**Original Implementation:**
- Classic recursive approach
- Uses `set()` for visited tracking
- Limited by Python's recursion depth limit
- Time complexity: O(V + E)
- Space complexity: O(V) (call stack)

**Optimized Implementation:**
- Iterative approach using explicit stack
- Eliminates per-call frame allocation overhead
- No recursion depth limitations
- Comparable time complexity but more predictable performance
- Better memory efficiency for large graphs

## Requirements

```bash
pip install matplotlib networkx
```

- **Python 3.7+**
- **matplotlib** — Visualization library
- **networkx** — Graph generation and manipulation

## Usage

### Running the Benchmarks

```bash
python main.py
```

This will:
1. Generate traversal visualizations (BFS and DFS on a 12-node graph)
2. Run comprehensive benchmarks on graphs with 50 to 1000 nodes
3. Display results in a formatted table
4. Generate comparison plots (PNG files)

### Output Files

After running, the following PNG files will be generated:

| File | Description |
|------|-------------|
| `viz_bfs.png` | BFS traversal with color-coded levels |
| `viz_dfs.png` | DFS traversal with visit order |
| `plot_all_comparison.png` | All four variants on single chart |
| `plot_per_algorithm.png` | 2×2 grid: time and memory per algorithm |

## Benchmark Configuration

- **Graph Sizes:** 50, 100, 250, 500, 750, 1000 nodes
- **Graph Type:** Erdős–Rényi random undirected graphs G(n, p)
- **Edge Probability:** p = 0.10
- **Starting Node:** 0 (first node)
- **Metrics:**
  - Execution time (seconds)
  - Peak memory usage (bytes)
  - Number of visited nodes

## Results Summary

The benchmarks measure the performance of four algorithm variants:

1. **BFS (original)** — Standard set-based implementation
2. **BFS (optimized)** — Bytearray bitmap optimization
3. **DFS (original)** — Recursive implementation
4. **DFS (optimized)** — Iterative stack-based implementation

### Key Findings

- **BFS (optimized)** outperforms the original due to O(1) bitmap access vs hash set overhead
- **DFS (optimized)** eliminates recursion overhead and avoids depth limitations
- Both optimizations scale better as graph size increases
- Memory usage improvements are significant for large graphs

## Project Structure

```
lab3/
├── main.py              # Benchmark suite and visualizations
├── report3.tex          # LaTeX source document
├── report3.pdf          # Compiled report
└── README.md            # This file
```

## Report Compilation

To recompile the LaTeX report:

```bash
pdflatex report3.tex
```

Ensure all generated PNG files are in the same directory as `report3.tex` before compilation.

## Implementation Details

### Graph Generation

The `generate_graph(n, edge_prob, seed)` function creates Erdős–Rényi random graphs:
- **n** — Number of vertices
- **edge_prob** — Probability of edge between any two vertices (default: 0.10)
- **seed** — Random seed for reproducibility

### Benchmarking Methodology

For each graph size and algorithm:
1. Generate random graph with consistent seed
2. Measure execution time using `time.perf_counter()` for precision
3. Track peak memory with `tracemalloc`
4. Record traversal order and statistics

### Visualization Approach

- **BFS visualization:** Color-coded by traversal level (plasma colormap)
- **DFS visualization:** Color-coded by visit order with colorbar
- Both use NetworkX spring layout for graph positioning

## Educational Value

This project demonstrates:
- Fundamental graph algorithms and their implementations
- Practical optimization techniques (data structure choice, iterative vs recursive)
- Empirical performance analysis and benchmarking
- Scientific visualization with matplotlib
- Trade-offs between algorithmic complexity and practical performance

## Notes

- Graph adjacency lists are implemented as dictionaries
- All graph traversals start from node 0
- Reproducibility is ensured through seeded random generation
- The bytearray optimization assumes non-negative integer node IDs
