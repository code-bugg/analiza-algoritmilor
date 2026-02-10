import cmath
import time

def tribonacci_binet(n):
    """
    Calculates the n-th Tribonacci number using the exact algebraic closed-form.
    T(n) = sum of (root_i^(n+1) / product of differences with other roots)
    """
    if n <= 1: 
        return 0
    n -= 1

    # Constants from the cubic solution (Cardano's method)
    # x^3 - x^2 - x - 1 = 0
    inner_sqrt = (33**0.5) * 3
    A_val = (19 + inner_sqrt)**(1/3)
    B_val = (19 - inner_sqrt)**(1/3)
    
    # Omega: the primitive complex cube root of unity
    omega = cmath.exp(2j * cmath.pi / 3)
    
    # The three exact roots (t1, t2, t3)
    t1 = (1 + A_val + B_val) / 3
    t2 = (1 + omega * A_val + (omega**2) * B_val) / 3
    t3 = (1 + (omega**2) * A_val + omega * B_val) / 3
    
    # Exact Binet-style formula for Tribonacci:
    # T_n = [t1^(n+1) / (t1-t2)(t1-t3)] + [t2^(n+1) / (t2-t1)(t2-t3)] + [t3^(n+1) / (t3-t1)(t3-t2)]
    
    def calculate_term(root, other1, other2):
        return (root**(n+1)) / ((root - other1) * (root - other2))

    term1 = calculate_term(t1, t2, t3)
    term2 = calculate_term(t2, t1, t3)
    term3 = calculate_term(t3, t1, t2)
    
    total = term1 + term2 + term3
    
    # The result is mathematically guaranteed to be real; 
    # we use .real and round to handle floating point noise.
    return int(round(total.real))

def tribonacci_matrix_power(n: int) -> int:
    if n <= 1:
        return 0
    def matrix_multiply(A, B):
        C = [[0, 0, 0], 
             [0, 0, 0], 
             [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i][j] += A[i][k] * B[k][j]
        return C
    def power(A, n):
        res = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        while n > 0:
            if n % 2 == 1:
                res = matrix_multiply(res, A)
            A = matrix_multiply(A, A)
            n //= 2
        return res
    T = [[1, 1, 1], [1, 0, 0], [0, 1, 0]]
    T = power(T, n-2)
    return T[0][0]

def tribonacci_recursive(n: int) -> int:
    if n <= 1:
        return 0
    if n == 2:
        return 1
    return tribonacci_recursive(n - 1) + tribonacci_recursive(n - 2) + tribonacci_recursive(n - 3)

def tribonacci_dynamic_programming(n: int) -> int:
    tribs = [0, 1, 1]
    if n <= 1:
        return 0
    for i in range(3, n):
        tribs.append(tribs[i - 1] + tribs[i - 2] + tribs[i - 3])
    return tribs[-1]

user_n = int(input("Enter n: "))
print(tribonacci_binet(user_n))
print(tribonacci_recursive(user_n))
print(tribonacci_matrix_power(user_n))
print(tribonacci_dynamic_programming(user_n))

def exec_time(trib_func, iters):
    #iterations = [10, 50, 100, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
    duration = []
    start_time = 0
    for iter in iters:
        start_time = time.time() 
        trib_func(iter)
        duration.append(time.time() - start_time)
    return duration

#iterations = [10, 50, 100, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
iterations = [10, 50, 100, 250, 500, 1000, 1100]
print(exec_time(tribonacci_binet, iterations))
print(exec_time(tribonacci_matrix_power, iterations))
print(exec_time(tribonacci_dynamic_programming, iterations))
