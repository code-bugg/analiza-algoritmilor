import time
import random
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.setrecursionlimit(20000)

# =============================================================================
# ORIGINAL IMPLEMENTATIONS
# =============================================================================

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot  = arr[len(arr) // 2]
    left   = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right  = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge_orig(left, right)

def _merge_orig(left, right):
    res = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    res.extend(left[i:])
    res.extend(right[j:])
    return res


def heapify(arr, n, i):
    """Recursive heapify."""
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr


MIN_MERGE = 32

def calc_min_run(n):
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r

def _insertion_sort(arr, left, right):
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def _tim_merge(arr, left, mid, right):
    left_part  = arr[left:mid + 1]
    right_part = arr[mid + 1:right + 1]
    i = j = 0
    k = left
    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]; i += 1
        else:
            arr[k] = right_part[j]; j += 1
        k += 1
    while i < len(left_part):
        arr[k] = left_part[i]; i += 1; k += 1
    while j < len(right_part):
        arr[k] = right_part[j]; j += 1; k += 1

# "Naive" Timsort: fixed run size = 32, no calc_min_run balancing heuristic
def tim_sort_naive(arr):
    n = len(arr)
    if n <= 1:
        return arr
    RUN = 32
    for start in range(0, n, RUN):
        end = min(start + RUN - 1, n - 1)
        _insertion_sort(arr, start, end)
    size = RUN
    while size < n:
        for left in range(0, n, 2 * size):
            mid   = min(left + size - 1, n - 1)
            right = min(left + 2 * size - 1, n - 1)
            if mid < right:
                _tim_merge(arr, left, mid, right)
        size *= 2
    return arr


# =============================================================================
# OPTIMIZED IMPLEMENTATIONS
# =============================================================================

# Optimized QuickSort:
#   - median-of-three pivot selection
#   - insertion sort fallback for small subarrays (< 16 elements)
#   - tail-call optimization: always recurse on smaller partition first
INSERTION_THRESHOLD = 16

def _median_of_three(arr, a, b, c):
    if arr[a] < arr[b]:
        if arr[b] < arr[c]: return b
        elif arr[a] < arr[c]: return c
        else: return a
    else:
        if arr[a] < arr[c]: return a
        elif arr[b] < arr[c]: return c
        else: return b

def _insertion_sort_inplace(arr, lo, hi):
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def _quick_sort_opt(arr, lo, hi):
    while lo < hi:
        if hi - lo < INSERTION_THRESHOLD:
            _insertion_sort_inplace(arr, lo, hi)
            return
        mid = (lo + hi) // 2
        p   = _median_of_three(arr, lo, mid, hi)
        arr[p], arr[hi] = arr[hi], arr[p]
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        p = i + 1
        if p - lo < hi - p:
            _quick_sort_opt(arr, lo, p - 1)
            lo = p + 1
        else:
            _quick_sort_opt(arr, p + 1, hi)
            hi = p - 1

def quick_sort_opt(arr):
    if len(arr) <= 1:
        return arr
    _quick_sort_opt(arr, 0, len(arr) - 1)
    return arr


# Optimized MergeSort:
#   - bottom-up iterative (eliminates recursion overhead)
#   - early-exit: skip merge when adjacent runs are already in order
def merge_sort_opt(arr):
    n = len(arr)
    if n <= 1:
        return arr
    arr = list(arr)
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid = min(lo + width, n)
            hi  = min(lo + 2 * width, n)
            if mid < n and arr[mid - 1] <= arr[mid]:
                continue                           # already sorted, skip
            left  = arr[lo:mid]
            right = arr[mid:hi]
            i = j = 0
            k = lo
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    arr[k] = left[i]; i += 1
                else:
                    arr[k] = right[j]; j += 1
                k += 1
            while i < len(left):
                arr[k] = left[i]; i += 1; k += 1
            while j < len(right):
                arr[k] = right[j]; j += 1; k += 1
        width *= 2
    return arr


# Optimized HeapSort:
#   - iterative heapify (eliminates recursion stack overhead)
def _heapify_iter(arr, n, i):
    while True:
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest

def heap_sort_opt(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        _heapify_iter(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        _heapify_iter(arr, i, 0)
    return arr


# Optimized Timsort:
#   - calc_min_run balancing heuristic (the full algorithm)
#   - ensures merge passes operate on balanced run pairs
def tim_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr
    min_run = calc_min_run(n)
    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        _insertion_sort(arr, start, end)
    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid   = min(left + size - 1, n - 1)
            right = min(left + 2 * size - 1, n - 1)
            if mid < right:
                _tim_merge(arr, left, mid, right)
        size *= 2
    return arr


# =============================================================================
# BENCHMARKING
# =============================================================================

ORIGINALS = [
    ("QuickSort",  quick_sort),
    ("MergeSort",  merge_sort),
    ("HeapSort",   heap_sort),
    ("Timsort",    tim_sort_naive),
]
OPTIMIZED = [
    ("QuickSort (opt)",  quick_sort_opt),
    ("MergeSort (opt)",  merge_sort_opt),
    ("HeapSort (opt)",   heap_sort_opt),
    ("Timsort (opt)",    tim_sort),
]

SIZES = [500, 1000, 2500, 5000, 7500, 10000]
orig_results = {name: [] for name, _ in ORIGINALS}
opt_results  = {name: [] for name, _ in OPTIMIZED}

print("Benchmarking original implementations...")
for n in SIZES:
    data = [random.randint(0, 100_000) for _ in range(n)]
    for name, func in ORIGINALS:
        temp = data.copy()
        start = time.perf_counter()
        func(temp)
        orig_results[name].append(time.perf_counter() - start)

print("Benchmarking optimized implementations...")
for n in SIZES:
    data = [random.randint(0, 100_000) for _ in range(n)]
    for name, func in OPTIMIZED:
        temp = data.copy()
        start = time.perf_counter()
        func(temp)
        opt_results[name].append(time.perf_counter() - start)

print("Done benchmarking.\n")


# =============================================================================
# PLOT 1 — All originals vs all optimized
# =============================================================================

COLORS = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

fig, ax = plt.subplots(figsize=(13, 7))
for (name, _), color in zip(ORIGINALS, COLORS):
    ax.plot(SIZES, orig_results[name], label=name,
            color=color, marker='o', linewidth=2, linestyle='--', markersize=6)
for (name, _), color in zip(OPTIMIZED, COLORS):
    ax.plot(SIZES, opt_results[name], label=name,
            color=color, marker='s', linewidth=2, linestyle='-', markersize=6)

ax.set_title("Original vs Optimized — All Algorithms\n"
             "(dashed = original, solid = optimized)", fontsize=14, pad=12)
ax.set_xlabel("Array Size (n)", fontsize=12)
ax.set_ylabel("Execution Time (seconds)", fontsize=12)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("plot_all_comparison.png", dpi=150)
plt.close()
print("Saved plot_all_comparison.png")


# =============================================================================
# PLOT 2 — Per-algorithm original vs optimized (2x2 subplots)
# =============================================================================

algo_names = ["QuickSort", "MergeSort", "HeapSort", "Timsort"]
orig_keys  = [n for n, _ in ORIGINALS]
opt_keys   = [n for n, _ in OPTIMIZED]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for idx, (aname, okey, optkey, color) in enumerate(
        zip(algo_names, orig_keys, opt_keys, COLORS)):
    ax = axes[idx]
    ax.plot(SIZES, orig_results[okey],  label="Original",
            color=color, marker='o', linewidth=2, linestyle='--', markersize=6)
    ax.plot(SIZES, opt_results[optkey], label="Optimized",
            color=color, marker='s', linewidth=2, linestyle='-',  markersize=6, alpha=0.75)
    ax.set_title(aname, fontsize=13, fontweight='bold')
    ax.set_xlabel("Array Size (n)", fontsize=10)
    ax.set_ylabel("Time (s)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.55)

fig.suptitle("Original vs Optimized — Per-Algorithm Comparison",
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("plot_per_algorithm.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot_per_algorithm.png")


# =============================================================================
# VISUALIZATION — QuickSort partition step
# =============================================================================

def draw_quicksort_viz():
    arr   = [5, 3, 8, 1, 9, 2, 7, 4, 6]
    pivot_i = len(arr) // 2
    pivot   = arr[pivot_i]
    left_p  = [x for x in arr if x < pivot]
    mid_p   = [x for x in arr if x == pivot]
    right_p = [x for x in arr if x > pivot]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    fig.suptitle("QuickSort — Single Partition Step (pivot = middle element)",
                 fontsize=13, fontweight='bold')

    def draw_array(ax, a, highlight=None, colors=None, title=''):
        ax.set_xlim(-0.5, len(a) - 0.5)
        ax.set_ylim(-0.2, 1.3)
        ax.axis('off')
        ax.set_title(title, fontsize=9, loc='left', pad=3)
        for i, v in enumerate(a):
            c = '#BBDEFB'
            if colors:   c = colors[i]
            elif highlight is not None and i == highlight: c = '#FF8A65'
            ax.add_patch(plt.Rectangle((i - 0.4, 0.1), 0.8, 0.8,
                         color=c, ec='#555555', lw=1.2, zorder=2))
            ax.text(i, 0.5, str(v), ha='center', va='center',
                    fontsize=13, fontweight='bold', zorder=3)
        if highlight is not None:
            ax.annotate('pivot', xy=(highlight, 0.9), xytext=(highlight, 1.15),
                        ha='center', fontsize=8, color='#BF360C',
                        arrowprops=dict(arrowstyle='->', color='#BF360C', lw=1.2))

    draw_array(axes[0], arr, highlight=pivot_i,
               title=f'  Step 1: Original array. Pivot = arr[{pivot_i}] = {pivot}')

    clrs = ['#A5D6A7' if v < pivot else '#FF8A65' if v == pivot else '#EF9A9A'
            for v in arr]
    draw_array(axes[1], arr, colors=clrs,
               title=f'  Step 2: Classify each element  '
                     f'(green < {pivot},  orange = {pivot},  red > {pivot})')

    combined = left_p + mid_p + right_p
    clrs_c   = (['#A5D6A7'] * len(left_p) +
                ['#FF8A65'] * len(mid_p)  +
                ['#EF9A9A'] * len(right_p))
    draw_array(axes[2], combined, colors=clrs_c,
               title=f'  Step 3: Partitioned  [ left | pivot | right ]  '
                     f'→ recurse on left and right sub-arrays')

    handles = [mpatches.Patch(color='#A5D6A7', label=f'< {pivot}'),
               mpatches.Patch(color='#FF8A65', label=f'= {pivot} (pivot)'),
               mpatches.Patch(color='#EF9A9A', label=f'> {pivot}')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig("viz_quicksort.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved viz_quicksort.png")

draw_quicksort_viz()


# =============================================================================
# VISUALIZATION — MergeSort divide-and-merge tree
# =============================================================================

def draw_mergesort_viz():
    levels = [
        ([5, 3, 8, 1, 9, 2, 7, 4], "  Step 1: Original array",           None),
        ([3, 5, 1, 8, 2, 9, 4, 7], "  Step 2: Pairs sorted (pass 1)",    [2, 4, 6]),
        ([1, 3, 5, 8, 2, 4, 7, 9], "  Step 3: Groups of 4 merged (pass 2)", [4]),
        ([1, 2, 3, 4, 5, 7, 8, 9], "  Step 4: Final sorted array (pass 3)", None),
    ]
    n = 8
    run_colors = ['#BBDEFB', '#C8E6C9', '#FFE0B2', '#F8BBD9']

    fig, axes = plt.subplots(4, 1, figsize=(11, 7))
    fig.suptitle("MergeSort — Bottom-Up Divide and Merge",
                 fontsize=13, fontweight='bold')

    for ax, (a, title, divs) in zip(axes, levels):
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.2, 1.3)
        ax.axis('off')
        ax.set_title(title, fontsize=9, loc='left', pad=2)

        if divs:
            boundaries = [0] + divs + [n]
            for gi in range(len(boundaries) - 1):
                lo, hi = boundaries[gi], boundaries[gi + 1]
                ax.add_patch(plt.Rectangle((lo - 0.48, 0.06), hi - lo - 0.04, 0.9,
                             color=run_colors[gi % len(run_colors)],
                             alpha=0.3, zorder=1, ec='#AAAAAA', lw=0.8))

        for i, v in enumerate(a):
            ax.add_patch(plt.Rectangle((i - 0.38, 0.15), 0.76, 0.7,
                         color='white', ec='#444444', lw=1.1, zorder=2))
            ax.text(i, 0.5, str(v), ha='center', va='center',
                    fontsize=12, fontweight='bold', zorder=3)

        if divs:
            for d in divs:
                ax.axvline(x=d - 0.5, color='#B71C1C', lw=1.3,
                           linestyle='--', alpha=0.65)

    plt.tight_layout()
    plt.savefig("viz_mergesort.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved viz_mergesort.png")

draw_mergesort_viz()


# =============================================================================
# VISUALIZATION — HeapSort: input → max-heap → sorted
# =============================================================================

def draw_heapsort_viz():
    stages = [
        ([4, 10, 3, 5, 1, 8, 7, 2, 9, 6], "Step 1: Input array\n(unsorted)", None),
        ([10, 9, 8, 5, 6, 3, 7, 2, 4, 1], "Step 2: Max-heap built\n(root = maximum)", 0),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Step 3: Sorted output\n(ascending)", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("HeapSort — Three Key Stages", fontsize=13, fontweight='bold')

    for ax, (a, title, hi_idx) in zip(axes, stages):
        ax.set_xlim(-0.5, len(a) - 0.5)
        ax.set_ylim(-0.4, 1.4)
        ax.axis('off')
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
        for i, v in enumerate(a):
            color = '#FF8A65' if i == hi_idx else '#BBDEFB'
            ax.add_patch(plt.Rectangle((i - 0.4, 0.2), 0.8, 0.7,
                         color=color, ec='#555555', lw=1.2, zorder=2))
            ax.text(i, 0.55, str(v), ha='center', va='center',
                    fontsize=11, fontweight='bold', zorder=3)
            ax.text(i, 0.1, f'[{i}]', ha='center', va='center',
                    fontsize=7, color='#777777')
        if hi_idx is not None:
            ax.annotate('max', xy=(hi_idx, 0.9), xytext=(hi_idx, 1.2),
                        ha='center', fontsize=8, color='#BF360C',
                        arrowprops=dict(arrowstyle='->', color='#BF360C', lw=1.2))

    plt.tight_layout()
    plt.savefig("viz_heapsort.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved viz_heapsort.png")

draw_heapsort_viz()


# =============================================================================
# VISUALIZATION — Timsort: run detection → sorted runs → merged
# =============================================================================

def draw_timsort_viz():
    arr        = [3, 5, 7, 1,  1, 2, 4, 6,  8, 9, 10, 11,  12, 13, 14, 15]
    sorted_arr = sorted(arr)
    n          = len(arr)
    runs       = [(0, 3), (4, 7), (8, 11), (12, 15)]
    run_colors = ['#BBDEFB', '#C8E6C9', '#FFE0B2', '#F8BBD9']

    arr_sorted_runs = [1, 3, 5, 7,  1, 2, 4, 6,  8, 9, 10, 11,  12, 13, 14, 15]
    color_map_runs  = []
    for ri, (lo, hi) in enumerate(runs):
        color_map_runs += [run_colors[ri]] * (hi - lo + 1)

    fig, axes = plt.subplots(3, 1, figsize=(13, 6.5))
    fig.suptitle("Timsort — Run Detection and Bottom-Up Merging",
                 fontsize=13, fontweight='bold')

    def draw_arr(ax, a, cmap, title):
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.2, 1.35)
        ax.axis('off')
        ax.set_title(title, fontsize=9, loc='left', pad=2)
        for i, v in enumerate(a):
            ax.add_patch(plt.Rectangle((i - 0.38, 0.15), 0.76, 0.7,
                         color=cmap[i], ec='#555555', lw=1.1, zorder=2))
            ax.text(i, 0.5, str(v), ha='center', va='center',
                    fontsize=10, fontweight='bold', zorder=3)

    # Stage 1 — original with run brackets
    ax = axes[0]
    draw_arr(ax, arr,
             [c for ri, (lo, hi) in enumerate(runs)
              for c in [run_colors[ri]] * (hi - lo + 1)],
             "  Step 1: Original array — each colour marks one identified run")
    for ri, (lo, hi) in enumerate(runs):
        cx = (lo + hi) / 2
        ax.text(cx, 1.15, f"Run {ri+1}", ha='center', fontsize=8,
                fontweight='bold', color='#333333')
        ax.add_patch(plt.Rectangle((lo - 0.46, 0.08), hi - lo + 0.92, 0.95,
                     color=run_colors[ri], alpha=0.2,
                     zorder=1, ec=run_colors[ri], lw=1.8))

    # Stage 2 — after insertion sort within each run
    draw_arr(axes[1], arr_sorted_runs, color_map_runs,
             "  Step 2: After insertion sort within each run "
             "(runs are individually sorted)")

    # Stage 3 — fully merged
    draw_arr(axes[2], sorted_arr, ['#B0BEC5'] * n,
             "  Step 3: After bottom-up merge passes — fully sorted array")

    plt.tight_layout()
    plt.savefig("viz_timsort.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved viz_timsort.png")

draw_timsort_viz()

print("\nAll plots and visualizations saved successfully.")
