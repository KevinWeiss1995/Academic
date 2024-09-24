import numpy as np
import time
import matplotlib.pyplot as plt

# Rosenbrock function (Objective Function)
# Has a narrow curved valley containing minimum 
# This makes it a good benchmark for optimization algos 
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

# Function evaluation over a grid of points
def evaluate_function(func, x, y):
    z = np.empty((np.size(x), np.size(y)))
    for i in range(np.size(x)):
        for j in range(np.size(y)):
            z[i, j] = func([x[i], y[j]])
    return z

# Gradient evaluation using central difference method
def eval_gradient(func, x, h=1e-7):
    fx = func(x)
    grad = np.zeros_like(x)

    for ix in np.ndindex(x.shape):
        old_value = x[ix]
        x[ix] = old_value + h
        fxh = func(x)
        x[ix] = old_value
        grad[ix] = (fxh - fx) / h

    return grad

# Permutation matrix generator
def generate_permutation_matrix(dim, indices):
    P = np.eye(dim)
    
    if len(indices) != 0:  # Fix: explicitly check if indices is empty
        sorted_indices = np.sort(indices)
        remaining_indices = np.delete(np.arange(dim), sorted_indices)
        new_order = np.concatenate((sorted_indices, remaining_indices))
        P = P[new_order]
    
    return P

# Backtracking line search algorithm
def backtracking_line_search(func, x, grad_f, gamma, max_iter):
    for _ in range(max_iter):
        if func(x - gamma * grad_f) <= func(x) - 0.75 * gamma * np.linalg.norm(grad_f) ** 2:
            break
        gamma *= 0.75
    return gamma

# Hessian approximation (second derivatives)
def eval_hessian(func, x, h, p=[]):
    # Initialize Hessian
    if len(p) == 0:
        p = np.arange(x.shape[0])
    
    hess = np.zeros((len(p), len(p)))  # Adjust Hessian to match length of p
    it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        ix = it.multi_index if len(p) == len(x) else it.value
        ii = it.multi_index

        # Compute gradient at x (left point)
        x1 = x.copy()
        df1 = eval_gradient(func, x1, h)

        # Compute gradient at x + h (right point)
        x2 = x.copy()
        x2[ix] += h
        df2 = eval_gradient(func, x2, h)

        # Compute second derivative (Hessian component)
        d2f = (df2 - df1) / h
        
        # Only extract the relevant portion corresponding to p
        hess[ii] = d2f[p]
        
        it.iternext()

    return hess

# Check if Hessian is singular and correct it if needed
def check_singular(hessian, alpha0=1e-4):
    alpha = alpha0
    while True:
        hessian += alpha * np.eye(len(hessian))
        if np.linalg.det(hessian) != 0:
            break
        alpha *= 1.1
    return hessian, alpha

# Newton method for optimization
def newton_method(func, x0, backtrack=False, gamma=0.001, p0=None, h=1e-7, tol=1e-4, max_iter=10000, verbose=1):
    t0 = time.time()
    
    if verbose >= 1:
        print('--- Newton Method ------------------------------')
    
    x = x0.copy()
    path_x = [x0.copy()]
    p = p0 if p0 is not None else []
    gamma_init = gamma
    alpha = 0.01

    for k in range(max_iter):
        if len(p) == 0:
            grad = eval_gradient(func, x, h)
            hess = eval_hessian(func, x, h)
            hess += alpha * np.eye(hess.shape[0])
            step = np.linalg.solve(hess, grad)
        else:
            grad_full = eval_gradient(func, x, h)
            P = generate_permutation_matrix(x.shape[0], p)
            grad = P @ grad_full
            grad_h, grad_t = grad[:len(p)], grad[len(p):]
            
            hess = eval_hessian(func, x, h, p)
            hess_h = hess + alpha * np.eye(hess.shape[0])
            step_h = np.linalg.solve(hess_h, grad_h)

            sigma = np.linalg.norm(grad_h) / np.linalg.norm(step_h)
            step_combined = np.concatenate((sigma * step_h, grad_t))
            step = P.T @ step_combined

        if backtrack:
            gamma = backtracking_line_search(func, x, grad, gamma, max_iter)

        x -= gamma * step
        path_x.append(x.copy())

        diff = np.linalg.norm(gamma * step)

        if verbose >= 2:
            print(f"{k}: {diff}, {x}")

        if diff < tol:
            elapsed_time = time.time() - t0
            if verbose >= 1:
                print('  Optimization terminated successfully.')
                print(f'    step={h:.1e}, tol={tol:.4f}, itmax={max_iter}')
                print(f'    Backtracking line search = {backtrack}')
                print(f'    Hessian approximation: {p0 if p0 is not None else "None"}')
                print(f'    Current function value = {func(x):.8f}')
                print(f'    Newton method converged in iter = {k}')
                print(f'    Elapsed time : {elapsed_time:.6f}')
                print(f'    Error = {diff:.6e}'+'\n')
                print(f'  Optimized solution is {x}')
                print('-----------------------------------------------')
            return np.array(path_x), k, elapsed_time, 0

    elapsed_time = time.time() - t0
    if verbose >= 1:
        print(f'    Newton method: did not converge, iter = {max_iter}')
        print(f'    Elapsed time : {elapsed_time:.6f}')
        print(f'    Error = {diff:.6e}')
    
    return np.array(path_x), max_iter, elapsed_time, 1


# Example usage
if __name__ == "__main__":
    initial_point = 0.9 * np.ones(8)
    p0 = np.array([0, 1, 2])
    sol = np.ones(8)

    x_path, num_iters, elapsed_time, ierr = newton_method(
        func=rosenbrock, 
        x0=initial_point, 
        backtrack=False, 
        gamma=0.001, 
        p0=p0, 
        verbose=1
    )

    # Post processing
    diff_x = np.linalg.norm(sol - x_path[-1])
    print(f"Solution difference: {diff_x:.6f}")
