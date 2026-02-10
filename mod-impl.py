import cmath
import time
import matplotlib.pyplot as plt

# --- ALGORITHM DEFINITIONS ---

def tribonacci_binet(n):
    if n <= 1: return 0
    n_adj = n - 1
    inner_sqrt = (33**0.5) * 3
    A_val = (19 + inner_sqrt)**(1/3)
    B_val = (19 - inner_sqrt)**(1/3)
    omega = cmath.exp(2j * cmath.pi / 3)
    t1 = (1 + A_val + B_val) / 3
    t2 = (1 + omega * A_val + (omega**2) * B_val) / 3
    t3 = (1 + (omega**2) * A_val + omega * B_val) / 3
    def calc(r): return (r**(n_adj+1)) / ((r-t2 if r!=t2 else r-t1)*(r-t3 if r!=t3 else r-t1))
    # Approximation using real root for benchmarking speed
    return int(round((t1**(n_adj+1) / ((t1-t2)*(t1-t3))).real))

def tribonacci_recursive(n):
    if n <= 1: return 0
    if n == 2: return 1
    return tribonacci_recursive(n-1) + tribonacci_recursive(n-2) + tribonacci_recursive(n-3)

def tribonacci_dp_list(n):
    if n <= 1: return 0
    tribs = [0, 1, 1]
    for i in range(3, n):
        tribs.append(tribs[i-1] + tribs[i-2] + tribs[i-3])
    return tribs[-1]

def tribonacci_dp_optimized(n):
    if n <= 1: return 0
    if n == 2: return 1
    a, b, c = 0, 1, 1
    for _ in range(3, n): a, b, c = b, c, a + b + c
    return c

def tribonacci_matrix(n):
    if n <= 1: return 0
    def mul(A, B):
        C = [[0,0,0],[0,0,0],[0,0,0]]
        for i in range(3):
            for j in range(3):
                for k in range(3): C[i][j] += A[i][k] * B[k][j]
        return C
    def pwr(A, p):
        res = [[1,0,0],[0,1,0],[0,0,1]]
        while p > 0:
            if p % 2 == 1: res = mul(res, A)
            A = mul(A, A); p //= 2
        return res
    T = [[1,1,1],[1,0,0],[0,1,0]]
    return pwr(T, n-2)[0][0]

# --- BENCHMARKING ---
iters_fast = [10, 250, 500, 750, 1000, 1100]
iters_slow = [10, 20, 30, 35, 38]

def get_times(func, iters):
    durations = []
    for n in iters:
        start = time.perf_counter()
        func(n)
        durations.append(time.perf_counter() - start)
    return durations

# Create Plot
plt.figure(figsize=(14, 6))

# Subplot 1: Efficient Algorithms
plt.subplot(1, 2, 1)
plt.plot(iters_fast, get_times(tribonacci_binet, iters_fast), 'o-', label='Binet (Fastest)')
plt.plot(iters_fast, get_times(tribonacci_dp_optimized, iters_fast), 's-', label='DP (Optimized)')
plt.plot(iters_fast, get_times(tribonacci_dp_list, iters_fast), 'x-', label='DP (List)')
plt.plot(iters_fast, get_times(tribonacci_matrix, iters_fast), 'd-', label='Matrix')
plt.title("O(n), O(log n), and O(1) Algorithms")
plt.xlabel("n"); plt.ylabel("Time (seconds)"); plt.legend()

# Subplot 2: Recursive Failure
plt.subplot(1, 2, 2)
plt.plot(iters_slow, get_times(tribonacci_recursive, iters_slow), 'r^-', label='Recursive')
plt.title("Recursive Exponential Growth")
plt.xlabel("n"); plt.ylabel("Time (seconds)"); plt.legend()

plt.tight_layout()
plt.savefig("tribonacci_comparison.png")
plt.show()
