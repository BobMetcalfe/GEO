import numpy as np
import math
from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.optimize import differential_evolution

## Reference: https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

# Define the function to be minimized
def f_heat(px_py):
    px, py = px_py[0:len(px_py):2], px_py[1:len(px_py):2]
    sum_heat = 0.0
    N = len(px)
    for i in range(N-1):
        p_x = px[i]
        p_y = py[i]
        for j in range(i+1, N):
            q_x = px[j]
            q_y = py[j]
            sum_heat += math.sqrt((p_x - q_x)**2 + (p_y - q_y)**2)
    return - sum_heat

def bayesian_optimization(f, grid_size, num_wells, acq_func="EI"):
    bounds = [(0.0, grid_size) for _ in range(2*num_wells)]

    res = gp_minimize(f,                  # the function to minimize
                    bounds,      # the bounds on each dimension of x
                    acq_func=acq_func,      # the acquisition function
                    n_calls=15,         # the number of evaluations of f
                    n_random_starts=5,  # the number of random initialization points
                    random_state=1234)   # the random seed
    plt.figure(figsize=(12, 12))
    new_plot = plot_convergence(res)
    plt.title(f"Bayesian Optimization ({acq_func})")
    new_plot.figure.savefig(f"plots/bayesian_convergence_{acq_func}.png")

def bayesian_optimization_with_lhs(f, grid_size, num_wells, l_bound=0.0, acq_func="EI"):
    bounds = [(0.0, grid_size) for _ in range(2*num_wells)]

    l_bounds = [l_bound, l_bound]
    u_bounds = [grid_size, grid_size]
    sampler = qmc.LatinHypercube(d=2, seed=1234)
    samples = sampler.random(n=num_wells)
    scaled_samples = qmc.scale(samples, l_bounds=l_bounds, u_bounds=u_bounds)
    scaled_samples = list(np.ravel(scaled_samples))

    res = gp_minimize(f,                  # the function to minimize
                    bounds,      # the bounds on each dimension of x
                    acq_func=acq_func,      # the acquisition function
                    n_calls=15,         # the number of evaluations of f
                    x0=scaled_samples,
                    n_random_starts=5,  # the number of random initialization points
                    random_state=1234)   # the random seed
    plt.figure(figsize=(12, 12))
    new_plot = plot_convergence(res)
    plt.title(f"Bayesian Optimization with Latin Hypercube Sampling ({acq_func})")
    new_plot.figure.savefig(f"plots/LHS_bayesian_convergence_{acq_func}.png")

def differential_evolution_optimization(objective_function, grid_size, num_wells, strategy='best1bin', l_bound=0.0):
    # TODO: try different strategies
    # Bounds for the optimization problem
    bounds = [(l_bound, grid_size)] * 2 * num_wells
    history = []

    def callback(xk, convergence):
        history.append(objective_function(xk))

    result = differential_evolution(objective_function, bounds, strategy=strategy, callback=callback)

    plt.figure(figsize=(12, 12))
    plt.title("Convergence Plot")
    plt.plot(history, linestyle='-', marker='.', color='blue')
    plt.xlabel("Number of Calls, n")
    plt.ylabel("Objective Function Value")
    plt.title(f"Differential Evolution Optimization ({strategy})")
    plt.grid(True)
    plt.savefig(f"plots/DE_convergence_{strategy}.png")

if __name__ == "__main__":
    grid_size = 100.0
    num_wells = 20

    # bayesian_optimization(f_heat, grid_size, num_wells, acq_func="EI") # LCB, EI, gp_hedge, PI

    # bayesian_optimization_with_lhs(f_heat, grid_size, num_wells, acq_func="EI")

    differential_evolution_optimization(f_heat, grid_size, num_wells, strategy='best2exp')







