"""
Laboratory Work No. 4 — Dijkstra & Floyd-Warshall
===================================================
Generates all figures referenced in report4.tex:
  viz_dijkstra.png
  viz_floyd.png
  plot_sparse.png
  plot_dense.png
  plot_per_algorithm.png
  plot_density_time.png
  plot_density_memory.png

Run:  python lab4.py
All .png files are written to the current directory.
"""

import random
import time
import tracemalloc
import heapq
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  SHARED STYLE
# ─────────────────────────────────────────────────────────────────────────────

BLUE_ORIG  = "#1A5276"
BLUE_OPT   = "#5DADE2"
RED_ORIG   = "#922B21"
RED_OPT    = "#F1948A"
GRID_COLOR = "#E8E8E8"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         GRID_COLOR,
    "grid.linewidth":     0.8,
    "figure.dpi":         150,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_weighted_graph(n, edge_prob=0.05, seed=42, max_w=20):
    """Directed weighted Erdős-Rényi G(n,p).
    Returns (adj, edges):
      adj   — {node: [(neighbour, weight), ...]}
      edges — [(u, v, weight), ...]
    """
    random.seed(seed)
    adj   = {i: [] for i in range(n)}
    edges = []
    for u in range(n):
        for v in range(n):
            if u != v and random.random() < edge_prob:
                w = random.randint(1, max_w)
                adj[u].append((v, w))
                edges.append((u, v, w))
    return adj, edges


def small_graph_example():
    """Fixed 7-node weighted directed graph for visualisations."""
    n = 7
    edge_list = [
        (0, 1, 4), (0, 2, 2),
        (1, 2, 5), (1, 3, 10),
        (2, 4, 3),
        (3, 5, 11),
        (4, 3, 4), (4, 5, 8), (4, 6, 7),
        (5, 6, 9),
    ]
    adj = {i: [] for i in range(n)}
    for u, v, w in edge_list:
        adj[u].append((v, w))
    return n, adj, edge_list


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ALGORITHM IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

INF = float("inf")


# ── Dijkstra original ─────────────────────────────────────────────────────────

def dijkstra(graph, source):
    """Single-source Dijkstra — binary heap, dict distance table."""
    dist = {node: INF for node in graph}
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:          # stale entry
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


# ── Dijkstra optimized ────────────────────────────────────────────────────────

def dijkstra_opt(graph, source, n):
    """Single-source Dijkstra — list distances, bytearray settled bitmap."""
    dist    = [INF] * n
    settled = bytearray(n)
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if settled[u]:
            continue
        settled[u] = 1
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


# ── Floyd-Warshall original ───────────────────────────────────────────────────

def floyd_warshall(n, edges):
    """All-pairs Floyd-Warshall — pure-Python list-of-lists."""
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edges:
        if w < d[u][v]:
            d[u][v] = w
    for k in range(n):
        for i in range(n):
            if d[i][k] == INF:
                continue
            for j in range(n):
                nd = d[i][k] + d[k][j]
                if nd < d[i][j]:
                    d[i][j] = nd
    return d


# ── Floyd-Warshall optimized ──────────────────────────────────────────────────

def floyd_warshall_opt(n, edges):
    """All-pairs Floyd-Warshall — NumPy vectorised row relaxation."""
    _INF = 1e18
    d = np.full((n, n), _INF, dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    for u, v, w in edges:
        if w < d[u, v]:
            d[u, v] = float(w)
    for k in range(n):
        mask = d[:, k] < _INF
        d[mask] = np.minimum(d[mask], d[mask, k : k + 1] + d[k])
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BENCHMARKING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bench(func, *args, repeats=3):
    """Return (min_elapsed_seconds, peak_bytes)."""
    best_time = INF
    best_mem  = 0
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if elapsed < best_time:
            best_time = elapsed
            best_mem  = peak
    return best_time, best_mem


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VIZ — Dijkstra step-by-step on small graph
# ─────────────────────────────────────────────────────────────────────────────

def _node_positions_circle(n, cx=0.5, cy=0.5, r=0.38):
    """Evenly spaced angles, source at top."""
    pos = {}
    for i in range(n):
        angle = math.pi / 2 - 2 * math.pi * i / n
        pos[i] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
    return pos


def draw_dijkstra_viz():
    n, adj, edge_list = small_graph_example()
    pos = _node_positions_circle(n)

    # Run Dijkstra and record settlement order + distances
    dist    = [INF] * n
    settled = bytearray(n)
    dist[0] = 0
    heap    = [(0, 0)]
    order   = []
    while heap:
        d, u = heapq.heappop(heap)
        if settled[u]:
            continue
        settled[u] = 1
        order.append(u)
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    # Reconstruct shortest-path tree edges
    prev = {i: None for i in range(n)}
    dist2    = [INF] * n
    settled2 = bytearray(n)
    dist2[0] = 0
    heap2    = [(0, 0)]
    while heap2:
        d, u = heapq.heappop(heap2)
        if settled2[u]:
            continue
        settled2[u] = 1
        for v, w in adj[u]:
            nd = d + w
            if nd < dist2[v]:
                dist2[v] = nd
                prev[v]  = u
                heapq.heappush(heap2, (nd, v))
    spt_edges = {(prev[v], v) for v in range(n) if prev[v] is not None}

    # Colour map: settled order → gradient
    cmap = plt.cm.plasma
    node_colors = {}
    for rank, node in enumerate(order):
        node_colors[node] = cmap(0.15 + 0.7 * rank / max(len(order) - 1, 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Dijkstra — Shortest-Path Tree from Node 0",
                 fontsize=13, fontweight="bold", pad=12)

    # Draw edges
    for u, v, w in edge_list:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        is_spt  = (u, v) in spt_edges
        lw      = 2.8 if is_spt else 0.9
        color   = "#E74C3C" if is_spt else "#BDC3C7"
        zorder  = 3 if is_spt else 1
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=14),
                    zorder=zorder)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, str(w), ha="center", va="center",
                fontsize=7.5, color="#555555",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Draw nodes
    for node in range(n):
        x, y = pos[node]
        rank  = order.index(node) if node in order else -1
        color = node_colors.get(node, "#95A5A6")
        circle = plt.Circle((x, y), 0.06, color=color, zorder=4, ec="white", lw=1.5)
        ax.add_patch(circle)
        label = f"{node}\nd={dist[node] if dist[node] != INF else '∞'}"
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=5)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#E74C3C", label="Shortest-path tree edge"),
        mpatches.Patch(color="#BDC3C7", label="Other edge"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)

    # Settlement order annotation
    order_str = " → ".join(str(v) for v in order)
    ax.text(0.5, 0.03, f"Settlement order:  {order_str}",
            ha="center", va="bottom", fontsize=8.5,
            transform=ax.transAxes, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F9F9F9", ec="#CCCCCC"))

    fig.tight_layout()
    fig.savefig("viz_dijkstra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved viz_dijkstra.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VIZ — Floyd-Warshall matrix evolution
# ─────────────────────────────────────────────────────────────────────────────

def draw_floyd_viz():
    """Show the distance matrix after k=0,1,2,3,4 passes on a 5-node graph."""
    n = 5
    edge_list = [
        (0, 1, 3), (0, 2, 8), (0, 4, -4),
        (1, 3, 1), (1, 4, 7),
        (2, 1, 4),
        (3, 0, 2), (3, 2, -5),
        (4, 3, 6),
    ]

    _INF = 999
    d = [[_INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edge_list:
        d[u][v] = w

    snapshots = [("Initial (k = —)", [row[:] for row in d])]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] != _INF and d[k][j] != _INF:
                    nd = d[i][k] + d[k][j]
                    if nd < d[i][j]:
                        d[i][j] = nd
        snapshots.append((f"After k = {k}", [row[:] for row in d]))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    fig.suptitle("Floyd–Warshall Distance Matrix Evolution",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, (title, mat) in enumerate(snapshots):
        ax = axes[idx]
        display = [[str(v) if v != _INF else "∞" for v in row] for row in mat]

        # Highlight changed cells vs previous snapshot
        if idx > 0:
            prev_mat = snapshots[idx - 1][1]
            changed  = [[mat[i][j] != prev_mat[i][j] for j in range(n)] for i in range(n)]
        else:
            changed  = [[False] * n for _ in range(n)]

        cell_colors = []
        for i in range(n):
            row_colors = []
            for j in range(n):
                if i == j:
                    row_colors.append("#D5E8D4")   # diagonal
                elif changed[i][j]:
                    row_colors.append("#FFE6CC")   # updated
                else:
                    row_colors.append("#FFFFFF")
            cell_colors.append(row_colors)

        tbl = ax.table(
            cellText   = display,
            cellColours= cell_colors,
            rowLabels  = [f"i={i}" for i in range(n)],
            colLabels  = [f"j={j}" for j in range(n)],
            loc        = "center",
            cellLoc    = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    # Hide unused subplot (we have 6 axes, 6 snapshots: initial + 5 passes)
    for idx in range(len(snapshots), len(axes)):
        axes[idx].axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(color="#FFE6CC", label="Updated in this pass"),
        mpatches.Patch(color="#D5E8D4", label="Diagonal (d[i][i] = 0)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    fig.savefig("viz_floyd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved viz_floyd.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SIZE-SCALING BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

SIZES = [50, 100, 250, 500, 750, 1000]

def run_size_benchmark(edge_prob, label):
    """Returns dict: algo_name -> (times_list, mems_list)."""
    results = {
        "Dijkstra (original)":        ([], []),
        "Dijkstra (optimized)":       ([], []),
        "Floyd-Warshall (original)":  ([], []),
        "Floyd-Warshall (optimized)": ([], []),
    }

    for V in SIZES:
        print(f"    V={V:4d}  [{label}]", end="  ", flush=True)
        adj, edges = generate_weighted_graph(V, edge_prob=edge_prob, seed=42)

        # Dijkstra original
        t, m = bench(dijkstra, adj, 0)
        results["Dijkstra (original)"][0].append(t)
        results["Dijkstra (original)"][1].append(m / 1024)
        print("D.orig", end=" ", flush=True)

        # Dijkstra optimized
        t, m = bench(dijkstra_opt, adj, 0, V)
        results["Dijkstra (optimized)"][0].append(t)
        results["Dijkstra (optimized)"][1].append(m / 1024)
        print("D.opt", end=" ", flush=True)

        # Floyd-Warshall — skip large sizes for pure-Python (too slow)
        if V <= 250:
            t, m = bench(floyd_warshall, V, edges, repeats=1)
        else:
            t, m = None, None
        results["Floyd-Warshall (original)"][0].append(t)
        results["Floyd-Warshall (original)"][1].append(m / 1024 if m else None)
        print("FW.orig", end=" ", flush=True)

        # Floyd-Warshall optimized (NumPy — tolerate up to V=1000)
        if V <= 750:
            t, m = bench(floyd_warshall_opt, V, edges, repeats=1)
        else:
            t, m = None, None
        results["Floyd-Warshall (optimized)"][0].append(t)
        results["Floyd-Warshall (optimized)"][1].append(m / 1024 if m else None)
        print("FW.opt")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOT — plot_sparse.png  &  plot_dense.png
# ─────────────────────────────────────────────────────────────────────────────

_ALGO_STYLE = {
    "Dijkstra (original)":        dict(color=BLUE_ORIG, ls="--", lw=2,   marker="o", ms=5),
    "Dijkstra (optimized)":       dict(color=BLUE_OPT,  ls="-",  lw=2,   marker="o", ms=5),
    "Floyd-Warshall (original)":  dict(color=RED_ORIG,  ls="--", lw=2,   marker="s", ms=5),
    "Floyd-Warshall (optimized)": dict(color=RED_OPT,   ls="-",  lw=2,   marker="s", ms=5),
}


def plot_size_scaling(results, filename, title_suffix):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, (times, _) in results.items():
        xs, ys = [], []
        for x, t in zip(SIZES, times):
            if t is not None:
                xs.append(x)
                ys.append(t)
        if xs:
            ax.plot(xs, ys, label=name, **_ALGO_STYLE[name])

    ax.set_yscale("log")
    ax.set_xlabel("Graph size V (nodes)", fontsize=11)
    ax.set_ylabel("Execution time (s)  [log scale]", fontsize=11)
    ax.set_title(f"Execution Time vs. Graph Size — {title_suffix}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOT — plot_per_algorithm.png  (2×2 grid: time + memory, per algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_algorithm(results_sparse):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Per-Algorithm Comparison: Original vs. Optimized\n"
                 "(Sparse graph, p = 0.05)",
                 fontsize=12, fontweight="bold")

    pairs = [
        ("Dijkstra",       "Dijkstra (original)",        "Dijkstra (optimized)"),
        ("Floyd-Warshall", "Floyd-Warshall (original)",  "Floyd-Warshall (optimized)"),
    ]
    row_labels = ["Execution time (s)", "Peak memory (KB)"]

    for col, (algo, orig_key, opt_key) in enumerate(pairs):
        for row in range(2):
            ax = axes[row][col]
            data_idx = row  # 0 = times, 1 = mems

            for key, ls, color, label in [
                (orig_key, "--", BLUE_ORIG if "Dijkstra" in algo else RED_ORIG, "Original"),
                (opt_key,  "-",  BLUE_OPT  if "Dijkstra" in algo else RED_OPT,  "Optimized"),
            ]:
                series = results_sparse[key][data_idx]
                xs, ys = [], []
                for x, y in zip(SIZES, series):
                    if y is not None:
                        xs.append(x)
                        ys.append(y)
                if xs:
                    ax.plot(xs, ys, ls=ls, color=color, lw=2,
                            marker="o", ms=5, label=label)

            ax.set_xlabel("Graph size V (nodes)", fontsize=9)
            ax.set_ylabel(row_labels[row], fontsize=9)
            ax.set_title(f"{algo} — {row_labels[row]}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig("plot_per_algorithm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved plot_per_algorithm.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  DENSITY BENCHMARK  (fixed V=300, vary p)
# ─────────────────────────────────────────────────────────────────────────────

DENSITY_V = 300
DENSITY_PS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]


def run_density_benchmark():
    results = {k: ([], []) for k in _ALGO_STYLE}

    for p in DENSITY_PS:
        print(f"    p={p:.2f}", end="  ", flush=True)
        adj, edges = generate_weighted_graph(DENSITY_V, edge_prob=p, seed=42)

        t, m = bench(dijkstra, adj, 0)
        results["Dijkstra (original)"][0].append(t)
        results["Dijkstra (original)"][1].append(m / 1024)
        print("D.orig", end=" ", flush=True)

        t, m = bench(dijkstra_opt, adj, 0, DENSITY_V)
        results["Dijkstra (optimized)"][0].append(t)
        results["Dijkstra (optimized)"][1].append(m / 1024)
        print("D.opt", end=" ", flush=True)

        # Pure-Python FW is feasible at V=300 for all densities
        t, m = bench(floyd_warshall, DENSITY_V, edges, repeats=1)
        results["Floyd-Warshall (original)"][0].append(t)
        results["Floyd-Warshall (original)"][1].append(m / 1024)
        print("FW.orig", end=" ", flush=True)

        t, m = bench(floyd_warshall_opt, DENSITY_V, edges, repeats=1)
        results["Floyd-Warshall (optimized)"][0].append(t)
        results["Floyd-Warshall (optimized)"][1].append(m / 1024)
        print("FW.opt")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10.  PLOT — plot_density_time.png  &  plot_density_memory.png
# ─────────────────────────────────────────────────────────────────────────────

def plot_density(density_results):
    xs = DENSITY_PS

    for metric_idx, (ylabel, filename) in enumerate([
        ("Execution time (s)  [log scale]", "plot_density_time.png"),
        ("Peak memory (KB)",                "plot_density_memory.png"),
    ]):
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for name, (times, mems) in density_results.items():
            ys = times if metric_idx == 0 else mems
            ax.plot(xs, ys, label=name, **_ALGO_STYLE[name])

        if metric_idx == 0:
            ax.set_yscale("log")

        # Shade crossover region (p ~ 0.3 – 0.7 where choice is non-obvious)
        ax.axvspan(0.28, 0.72, alpha=0.07, color="grey",
                   label="Crossover region")

        ax.set_xlabel("Edge probability p", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"{'Execution Time' if metric_idx == 0 else 'Peak Memory'} "
            f"vs. Edge Density  (V = {DENSITY_V})",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  PRINT RESULT TABLES
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results, sizes, label):
    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"{'─'*72}")
    header = f"{'V':>6}  {'Dijk.orig(s)':>14}  {'Dijk.opt(s)':>13}  "
    header += f"{'FW.orig(s)':>12}  {'FW.opt(s)':>11}"
    print(header)
    print("─" * 72)
    for i, V in enumerate(sizes):
        row = f"{V:>6}  "
        for key in ["Dijkstra (original)", "Dijkstra (optimized)",
                    "Floyd-Warshall (original)", "Floyd-Warshall (optimized)"]:
            t = results[key][0][i]
            row += f"{'N/A':>13}  " if t is None else f"{t:>13.6f}  "
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Lab 4 — Dijkstra & Floyd-Warshall")
    print("=" * 60)

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\n[1/4] Generating algorithm visualisations …")
    draw_dijkstra_viz()
    draw_floyd_viz()

    # ── Size-scaling sparse ───────────────────────────────────────────────────
    print("\n[2/4] Size-scaling benchmark — sparse (p=0.05) …")
    res_sparse = run_size_benchmark(edge_prob=0.05, label="sparse")
    print_table(res_sparse, SIZES, "Sparse graph (p = 0.05)")
    plot_size_scaling(res_sparse, "plot_sparse.png", "Sparse graph (p = 0.05)")
    plot_per_algorithm(res_sparse)

    # ── Size-scaling dense ────────────────────────────────────────────────────
    print("\n[3/4] Size-scaling benchmark — dense (p=0.50) …")
    res_dense = run_size_benchmark(edge_prob=0.50, label="dense")
    print_table(res_dense, SIZES, "Dense graph (p = 0.50)")
    plot_size_scaling(res_dense, "plot_dense.png", "Dense graph (p = 0.50)")

    # ── Density sweep ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Density benchmark (V={DENSITY_V}, p varies) …")
    res_density = run_density_benchmark()
    plot_density(res_density)

    print("\n" + "=" * 60)
    print("  All figures saved.  Copy the .png files next to report4.tex")
    print("  and run:  pdflatex report4.tex  (twice for cross-references)")
    print("=" * 60)


if __name__ == "__main__":
    main()
