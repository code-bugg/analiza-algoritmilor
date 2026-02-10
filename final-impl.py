"""
Tribonacci Sequence Algorithm Analysis
========================================
Laboratory Work 1: Study and Empirical Analysis of Algorithms for
Determining Tribonacci N-th Term

This module implements and benchmarks five different algorithms for
computing the n-th Tribonacci number.
"""

import cmath
import time
import matplotlib.pyplot as plt
import sys

# Increase recursion depth for testing
sys.setrecursionlimit(2000)


# ============================================================================
# ALGORITHM IMPLEMENTATIONS
# ============================================================================

def tribonacci_recursive(n):
    """
    Compute n-th Tribonacci number using naive recursion.
    
    Time Complexity: O(3^n) - exponential
    Space Complexity: O(n) - recursion depth
    Practical Limit: n ≈ 35
    
    Args:
        n: Index of Tribonacci number to compute
        
    Returns:
        The n-th Tribonacci number
    """
    if n <= 1: 
        return 0
    if n == 2: 
        return 1
    return (tribonacci_recursive(n-1) + 
            tribonacci_recursive(n-2) + 
            tribonacci_recursive(n-3))


def tribonacci_binet(n):
    """
    Compute n-th Tribonacci number using Binet's formula with complex roots.
    
    Time Complexity: O(1) - constant time
    Space Complexity: O(1)
    Limitation: Floating-point overflow for n > 1100
    
    The characteristic equation x³ - x² - x - 1 = 0 has three roots.
    This method uses the closed-form solution based on these roots.
    
    Args:
        n: Index of Tribonacci number to compute
        
    Returns:
        The n-th Tribonacci number (approximate for large n)
    """
    if n <= 1: 
        return 0
    
    n_adj = n - 1
    
    # Compute cubic roots: A = (19 + 3√33)^(1/3), B = (19 - 3√33)^(1/3)
    inner_sqrt = (33**0.5) * 3
    A_val = (19 + inner_sqrt)**(1/3)
    B_val = (19 - inner_sqrt)**(1/3)
    
    # ω = e^(2πi/3) - primitive cube root of unity
    omega = cmath.exp(2j * cmath.pi / 3)
    
    # Three roots of characteristic equation
    t1 = (1 + A_val + B_val) / 3                    # Real root (Tribonacci constant ≈ 1.839)
    t2 = (1 + omega * A_val + (omega**2) * B_val) / 3  # Complex root
    t3 = (1 + (omega**2) * A_val + omega * B_val) / 3  # Complex conjugate
    
    def term(r, o1, o2):
        """Compute contribution from each root."""
        return (r**(n_adj+1)) / ((r-o1)*(r-o2))
    
    # Closed-form solution
    total = term(t1, t2, t3) + term(t2, t1, t3) + term(t3, t1, t2)
    return int(round(total.real))


def tribonacci_dp_list(n):
    """
    Compute n-th Tribonacci number using dynamic programming with list storage.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - stores all values
    
    Args:
        n: Index of Tribonacci number to compute
        
    Returns:
        The n-th Tribonacci number
    """
    if n <= 1: 
        return 0
    if n == 2: 
        return 1
    
    res = [0, 1, 1]
    for i in range(3, n + 1):
        res.append(res[i-1] + res[i-2] + res[i-3])
    
    return res[n]


def tribonacci_dp_optimized(n):
    """
    Compute n-th Tribonacci number using space-optimized dynamic programming.
    
    Time Complexity: O(n) - single iteration
    Space Complexity: O(1) - only three variables
    
    This is the most practical algorithm for computing large Tribonacci numbers,
    balancing performance with minimal memory usage.
    
    Args:
        n: Index of Tribonacci number to compute
        
    Returns:
        The n-th Tribonacci number
    """
    if n <= 1: 
        return 0
    if n == 2: 
        return 1
    
    a, b, c = 0, 1, 1
    for _ in range(3, n):
        a, b, c = b, c, a + b + c
    
    return c


def tribonacci_matrix(n):
    """
    Compute n-th Tribonacci number using matrix exponentiation.
    
    Time Complexity: O(log n) - binary exponentiation
    Space Complexity: O(1) - constant number of 3×3 matrices
    
    Uses the transition matrix:
        T = [[1, 1, 1],
             [1, 0, 0],
             [0, 1, 0]]
    
    The n-th Tribonacci number is T^(n-2)[0][0].
    
    Args:
        n: Index of Tribonacci number to compute
        
    Returns:
        The n-th Tribonacci number
    """
    if n <= 1: 
        return 0
    
    def mul(A, B):
        """Multiply two 3×3 matrices."""
        C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i][j] += A[i][k] * B[k][j]
        return C
    
    def pwr(A, p):
        """Compute matrix power using binary exponentiation."""
        # Identity matrix
        res = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        while p > 0:
            if p % 2 == 1:
                res = mul(res, A)
            A = mul(A, A)
            p //= 2
        
        return res
    
    # Transition matrix
    T = [[1, 1, 1], 
         [1, 0, 0], 
         [0, 1, 0]]
    
    return pwr(T, n-2)[0][0]


# ============================================================================
# BENCHMARKING AND VISUALIZATION
# ============================================================================

def run_benchmarks():
    """
    Execute performance benchmarks for all algorithms and generate plots.
    
    Returns:
        Dictionary containing benchmark results for each algorithm
    """
    print("Starting Tribonacci Algorithm Benchmarks...")
    print("=" * 60)
    
    # Test values for efficient methods (up to Binet overflow)
    fast_ns = [10, 100, 250, 500, 750, 1000, 1100]
    
    # Test values for recursive method (exponential - only small n)
    recursive_ns = [5, 10, 15, 20, 25, 30, 31, 32, 33, 34, 35]
    
    # Results storage
    results = {
        "Binet": [], 
        "DP_List": [], 
        "DP_Opt": [], 
        "Matrix": [], 
        "Recursive": []
    }
    
    # Benchmark efficient methods
    print("\nBenchmarking efficient methods...")
    for n in fast_ns:
        print(f"Testing n = {n}...")
        
        # Binet (check for overflow)
        try:
            t0 = time.perf_counter()
            tribonacci_binet(n)
            elapsed = time.perf_counter() - t0
            results["Binet"].append(elapsed)
            print(f"  Binet: {elapsed:.6f}s")
        except (OverflowError, ZeroDivisionError):
            results["Binet"].append(None)
            print(f"  Binet: OVERFLOW")
        
        # DP List
        t0 = time.perf_counter()
        tribonacci_dp_list(n)
        elapsed = time.perf_counter() - t0
        results["DP_List"].append(elapsed)
        print(f"  DP List: {elapsed:.6f}s")
        
        # DP Optimized
        t0 = time.perf_counter()
        tribonacci_dp_optimized(n)
        elapsed = time.perf_counter() - t0
        results["DP_Opt"].append(elapsed)
        print(f"  DP Optimized: {elapsed:.6f}s")
        
        # Matrix Power
        t0 = time.perf_counter()
        tribonacci_matrix(n)
        elapsed = time.perf_counter() - t0
        results["Matrix"].append(elapsed)
        print(f"  Matrix: {elapsed:.6f}s")
    
    # Benchmark recursive method
    print("\nBenchmarking recursive method (WARNING: slow)...")
    for n in recursive_ns:
        print(f"Testing n = {n}...")
        t0 = time.perf_counter()
        tribonacci_recursive(n)
        elapsed = time.perf_counter() - t0
        results["Recursive"].append(elapsed)
        print(f"  Recursive: {elapsed:.6f}s")
    
    # Generate plots
    print("\nGenerating performance plots...")
    plot_results(fast_ns, recursive_ns, results)
    
    print("\n" + "=" * 60)
    print("Benchmarking complete!")
    print("Plot saved as: tribonacci_performance.png")
    
    return results


def plot_results(fast_ns, recursive_ns, results):
    """
    Create visualization of benchmark results.
    
    Args:
        fast_ns: List of n values for efficient methods
        recursive_ns: List of n values for recursive method
        results: Dictionary of benchmark results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Efficient methods
    ax1.plot(fast_ns, results["Binet"], 
             label="Binet (Fastest)", marker='o', linewidth=2)
    ax1.plot(fast_ns, results["DP_List"], 
             label="DP List", marker='s', linewidth=2)
    ax1.plot(fast_ns, results["DP_Opt"], 
             label="DP Optimized", marker='^', linewidth=2)
    ax1.plot(fast_ns, results["Matrix"], 
             label="Matrix", marker='x', linewidth=2)
    
    ax1.set_title("Efficient Methods (Up to Binet Overflow)", 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("n", fontsize=11)
    ax1.set_ylabel("Time (seconds)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recursive method
    ax2.plot(recursive_ns, results["Recursive"], 
             color='red', marker='v', linewidth=2, label='Recursive')
    
    ax2.set_title("Recursive Method (Exponential Explosion)", 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("n", fontsize=11)
    ax2.set_ylabel("Time (seconds)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("tribonacci_performance.png", dpi=300, bbox_inches='tight')
    plt.show()


def verify_correctness():
    """
    Verify that all algorithms produce correct results for small values of n.
    """
    print("\nVerifying algorithm correctness...")
    print("-" * 60)
    
    # Known Tribonacci sequence values
    expected = [0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149, 274, 504, 927, 1705]
    
    all_correct = True
    
    for n in range(len(expected)):
        rec = tribonacci_recursive(n) if n <= 30 else None
        binet = tribonacci_binet(n)
        dp_list = tribonacci_dp_list(n)
        dp_opt = tribonacci_dp_optimized(n)
        matrix = tribonacci_matrix(n)
        
        exp = expected[n]
        
        # Check each algorithm
        if rec is not None and rec != exp:
            print(f"ERROR at n={n}: Recursive returned {rec}, expected {exp}")
            all_correct = False
        if binet != exp:
            print(f"ERROR at n={n}: Binet returned {binet}, expected {exp}")
            all_correct = False
        if dp_list != exp:
            print(f"ERROR at n={n}: DP List returned {dp_list}, expected {exp}")
            all_correct = False
        if dp_opt != exp:
            print(f"ERROR at n={n}: DP Opt returned {dp_opt}, expected {exp}")
            all_correct = False
        if matrix != exp:
            print(f"ERROR at n={n}: Matrix returned {matrix}, expected {exp}")
            all_correct = False
    
    if all_correct:
        print("✓ All algorithms produce correct results for n = 0..14")
    else:
        print("✗ Some algorithms produced incorrect results")
    
    print("-" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRIBONACCI SEQUENCE ALGORITHM ANALYSIS")
    print("Laboratory Work 1")
    print("=" * 60)
    
    # Verify correctness
    verify_correctness()
    
    # Run benchmarks
    results = run_benchmarks()
    
    # Display sample computations
    print("\n" + "=" * 60)
    print("SAMPLE COMPUTATIONS")
    print("=" * 60)
    
    test_values = [10, 20, 30, 100, 1000]
    
    for n in test_values:
        print(f"\nT({n}):")
        
        if n <= 30:
            print(f"  Recursive:    {tribonacci_recursive(n)}")
        
        try:
            print(f"  Binet:        {tribonacci_binet(n)}")
        except (OverflowError, ZeroDivisionError):
            print(f"  Binet:        OVERFLOW")
        
        print(f"  DP List:      {tribonacci_dp_list(n)}")
        print(f"  DP Optimized: {tribonacci_dp_optimized(n)}")
        print(f"  Matrix:       {tribonacci_matrix(n)}")
    
    print("\n" + "=" * 60)
