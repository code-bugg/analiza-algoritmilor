"""
Laboratory Work No. 3 -- BFS and DFS Empirical Analysis
========================================================
Generates all benchmark data and saves the following figures:
    viz_bfs.png             -- BFS traversal visualization
    viz_dfs.png             -- DFS traversal visualization
    plot_all_comparison.png -- All four variants on one chart
    plot_per_algorithm.png  -- Per-algorithm original vs optimized

Run:
    pip install matplotlib networkx
    python lab3_benchmark.py
"""

import random
import time
import tracemalloc
from collections import deque, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Graph generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_graph(n, edge_prob=0.10, seed=42):
    """Undirected Erdos-Renyi G(n, p) graph as adjacency list."""
    random.seed(seed)
    graph = {i: [] for i in range(n)}
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < edge_prob:
                graph[u].append(v)
                graph[v].append(u)
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# BFS -- original
# ─────────────────────────────────────────────────────────────────────────────

def bfs(graph, start):
    """BFS (original) -- marks visited at dequeue time."""
    visited = set()
    queue   = deque([start])
    visited.add(start)
    order   = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return order


# ─────────────────────────────────────────────────────────────────────────────
# BFS -- optimized
# ─────────────────────────────────────────────────────────────────────────────

def bfs_opt(graph, start):
    """BFS (optimized) -- replaces the hash-set visited check with a bytearray
    bitmap.  For integer node IDs, bytearray[node] is a direct O(1) array
    access with no hashing, reducing the per-node overhead vs set lookup."""
    n       = len(graph)
    visited = bytearray(n)              # 0 = unvisited, 1 = visited
    visited[start] = 1
    queue   = deque([start])
    order   = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph[node]:
            if not visited[neighbour]:
                visited[neighbour] = 1
                queue.append(neighbour)
    return order


# ─────────────────────────────────────────────────────────────────────────────
# DFS -- original (recursive)
# ─────────────────────────────────────────────────────────────────────────────

def dfs(graph, start, visited=None):
    """DFS (original) -- classic recursive implementation."""
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for neighbour in graph[start]:
        if neighbour not in visited:
            order.extend(dfs(graph, neighbour, visited))
    return order


# ─────────────────────────────────────────────────────────────────────────────
# DFS -- optimized (iterative)
# ─────────────────────────────────────────────────────────────────────────────

def dfs_opt(graph, start):
    """DFS (optimized) -- iterative explicit stack; eliminates Python's
    per-call frame allocation overhead of the recursive version and avoids
    the recursion-limit ceiling on large graphs."""
    visited = set()
    stack   = [start]
    order   = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbour in graph[node]:
            if neighbour not in visited:
                stack.append(neighbour)
    return order


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

SIZES = [50, 100, 250, 500, 750, 1000]

ALGORITHMS = [
    ("BFS (original)",  bfs),
    ("BFS (optimized)", bfs_opt),
    ("DFS (original)",  dfs),
    ("DFS (optimized)", dfs_opt),
]

def run_benchmarks():
    results = {name: {"time": [], "memory": [], "nodes": []} for name, _ in ALGORITHMS}

    for V in SIZES:
        print(f"  Benchmarking V={V} ...", flush=True)
        graph = generate_graph(V, edge_prob=0.10, seed=42)

        for name, func in ALGORITHMS:
            # Time measurement
            tracemalloc.start()
            t0 = time.perf_counter()
            order = func(graph, 0)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            results[name]["time"].append(elapsed)
            results[name]["memory"].append(peak)
            results[name]["nodes"].append(len(order))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

CMAP = plt.cm.plasma

def _draw_graph_base(ax, G, pos, title):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=1.2, edge_color="#888888")


def make_viz_bfs(filename="viz_bfs.png"):
    """BFS traversal visualization on a small graph."""
    random.seed(7)
    n = 12
    G = nx.gnm_random_graph(n, 18, seed=7)
    pos = nx.spring_layout(G, seed=3)

    # Run BFS manually to capture level info
    visited = {}
    queue   = deque([0])
    visited[0] = 0
    level_of  = {0: 0}
    while queue:
        node = queue.popleft()
        for nb in sorted(G.neighbors(node)):
            if nb not in visited:
                visited[nb] = len(visited)
                level_of[nb] = level_of[node] + 1
                queue.append(nb)

    max_level = max(level_of.values()) if level_of else 1
    node_colors = [CMAP(level_of.get(n, 0) / max(max_level, 1)) for n in G.nodes()]
    visit_order = {v: k for k, v in enumerate(visited)}

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_graph_base(ax, G, pos, "BFS -- Level-by-Level Traversal")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=550, alpha=0.92)
    labels = {n: f"{n}\n(L{level_of.get(n,'?')})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_color="white")

    # Legend for levels
    patches = [mpatches.Patch(color=CMAP(l / max(max_level, 1)), label=f"Level {l}")
               for l in range(max_level + 1)]
    ax.legend(handles=patches, loc="lower right", fontsize=8, title="BFS Level")

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def make_viz_dfs(filename="viz_dfs.png"):
    """DFS traversal visualization on a small graph."""
    random.seed(7)
    G = nx.gnm_random_graph(12, 18, seed=7)
    pos = nx.spring_layout(G, seed=3)

    # Run DFS iteratively, capture visit order
    visited = {}
    stack   = [0]
    adj     = {n: sorted(G.neighbors(n), reverse=True) for n in G.nodes()}
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited[node] = len(visited)
        for nb in adj[node]:
            if nb not in visited:
                stack.append(nb)

    n_visited = len(visited)
    node_colors = [CMAP(visited.get(n, 0) / max(n_visited - 1, 1)) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_graph_base(ax, G, pos, "DFS -- Depth-First Traversal (visit order)")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=550, alpha=0.92)
    labels = {n: f"{n}\n(#{visited.get(n, '?')})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_color="white")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, n_visited - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Visit order", fontsize=9)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "BFS (original)":  "#1f77b4",
    "BFS (optimized)": "#1f77b4",
    "DFS (original)":  "#d62728",
    "DFS (optimized)": "#d62728",
}

def plot_all_comparison(results, filename="plot_all_comparison.png"):
    """All four timing curves on one chart."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, _ in ALGORITHMS:
        style = "--" if "original" in name else "-"
        lw    = 1.8 if "original" in name else 2.5
        ax.plot(SIZES, results[name]["time"], style,
                color=COLORS[name], linewidth=lw,
                marker="o", markersize=4, label=name)

    ax.set_xlabel("Graph size $V$ (nodes)", fontsize=12)
    ax.set_ylabel("Execution time (s)", fontsize=12)
    ax.set_title("BFS vs DFS -- All Variants", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_per_algorithm(results, filename="plot_per_algorithm.png"):
    """2x2 grid: per-algorithm time + memory comparisons."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    pairs = [
        ("BFS (original)", "BFS (optimized)", "BFS", "#1f77b4"),
        ("DFS (original)", "DFS (optimized)", "DFS", "#d62728"),
    ]

    for col, (orig_key, opt_key, algo, color) in enumerate(pairs):
        # Top row: execution time
        ax = axes[0][col]
        ax.plot(SIZES, results[orig_key]["time"], "--", color=color,
                linewidth=1.8, marker="o", markersize=4, label="Original")
        ax.plot(SIZES, results[opt_key]["time"],  "-",  color=color,
                linewidth=2.5, marker="s", markersize=4, label="Optimized")
        ax.set_title(f"{algo} -- Execution Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Graph size $V$ (nodes)", fontsize=10)
        ax.set_ylabel("Time (s)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Bottom row: peak memory
        ax = axes[1][col]
        orig_mem = [m / 1024 for m in results[orig_key]["memory"]]
        opt_mem  = [m / 1024 for m in results[opt_key]["memory"]]
        ax.plot(SIZES, orig_mem, "--", color=color,
                linewidth=1.8, marker="o", markersize=4, label="Original")
        ax.plot(SIZES, opt_mem,  "-",  color=color,
                linewidth=2.5, marker="s", markersize=4, label="Optimized")
        ax.set_title(f"{algo} -- Peak Memory", fontsize=12, fontweight="bold")
        ax.set_xlabel("Graph size $V$ (nodes)", fontsize=10)
        ax.set_ylabel("Peak memory (KB)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Per-Algorithm Comparison: Original vs Optimized",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def print_results_table(results):
    header = f"{'V':>6}  {'BFS orig (s)':>14}  {'BFS opt (s)':>13}  {'DFS orig (s)':>14}  {'DFS opt (s)':>13}"
    print("\n" + header)
    print("-" * len(header))
    for i, V in enumerate(SIZES):
        bo = results["BFS (original)"]["time"][i]
        bp = results["BFS (optimized)"]["time"][i]
        do = results["DFS (original)"]["time"][i]
        dp = results["DFS (optimized)"]["time"][i]
        print(f"{V:>6}  {bo:>14.6f}  {bp:>13.6f}  {do:>14.6f}  {dp:>13.6f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Lab 3: BFS & DFS Empirical Analysis ===\n")

    print("[1/4] Generating traversal visualizations ...")
    make_viz_bfs("viz_bfs.png")
    make_viz_dfs("viz_dfs.png")

    print("\n[2/4] Running benchmarks ...")
    results = run_benchmarks()

    print("\n[3/4] Results table:")
    print_results_table(results)

    print("[4/4] Saving plots ...")
    plot_all_comparison(results, "plot_all_comparison.png")
    plot_per_algorithm(results,  "plot_per_algorithm.png")

    print("\nDone! Generated files:")
    for f in ["viz_bfs.png", "viz_dfs.png",
              "plot_all_comparison.png", "plot_per_algorithm.png"]:
        print(f"  {f}")
    print("\nPlace all .png files in the same directory as raport_lab3.tex before compiling.")
