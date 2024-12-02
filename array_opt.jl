using BlackBoxOptim
using Plots
using BayesianOptimization, GaussianProcesses, Distributions
using Optim, Hyperopt
using Random
using LatinHypercubeSampling

## TODO
# - Add reference papers below
# --> probabilistic descent: https://arxiv.org/pdf/2204.01275
# --> Upper UpperConfidenceBound: https://www.jmlr.org/papers/volume3/auer02a/auer02a.pdf

## Notes
#  - Cost Reduction: Computational Realm
# - If case study data is available => optimization approach would be more feasible

## Differential Evolution Optimizers
# - metaheuristics, do not guarantee that an optimal solution is ever found
# - Do not Require Differentiability

# Temporary function: Returns the total heat output of all the wells given their locations
function f_heat(px_py)
    px, py = px_py[1:2:end], px_py[2:2:end]
    sum_heat = 0.0
    N = length(px)
    for i in 1:N-1
        p_x = px[i]
        p_y = py[i]
        for j in i+1:N
            q_x = px[j]
            q_y = py[j]
            sum_heat += sqrt((p_x - q_x)^2 + (p_y - q_y)^2)
        end
    end
    return sum_heat
end

# # Bayesian Optimization
# https://arxiv.org/pdf/1807.02811
# https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-terms-1d2bcdeaf12f
function bayesian_helper(px_py, grid_size; lhs=false)
    # TODO: change the structure of the input array to be less confusing
    num_dims = length(px_py)
    model = ElasticGPE(num_dims, 
                        mean = MeanConst(grid_size/2.0), 
                        kernel = SEArd(zeros(num_dims), 5.), 
                        logNoise = -10.0, 
                        capacity = 3000) 
    # set_priors!(model.mean, [Normal(1, 2)]) # if not uniform prior (i.e Normal prior)
    modeloptimizer = MAPGPOptimizer(every = 1, # changed from 10 to 1
                                    # noisebounds = [-10, -10], 
                                    # kernbounds = [[-1, -1, 0], [4, 4, 10]], 
                                    maxeval = 10) # changed from 40 to 10
    if lhs
        init = ScaledLHSIterator
    else
        init = ScaledSobolIterator
    end
    lower_bounds = fill(0, num_dims)
    upper_bounds = fill(grid_size, num_dims)
    init_iterations = 1
    opt = BOpt(f_heat, 
                model, 
                UpperConfidenceBound(), 
                modeloptimizer, 
                lower_bounds, 
                upper_bounds, 
                repetitions = 1, 
                maxiterations = 1, 
                sense = Max, 
                acquisitionoptions = (method = :LD_LBFGS, restarts = 5, maxtime = 0.1, maxeval = 1000), 
                initializer_iterations = init_iterations,
                initializer = init(lower_bounds, upper_bounds, init_iterations),
                verbosity = Progress)
    return opt
end

function bayesian_optimization(px_py, grid_size)
    opt = bayesian_helper(px_py, grid_size)
    result = boptimize!(opt)
    maxiterations!(opt, 50)
    result = boptimize!(opt)
    return result
end

## Track and plot history of optimal answers for each iteration from bayesian Optimization
function plot_bayesian(px_py, grid_size; num_plot_points = 10)
    opt = bayesian_helper(px_py, grid_size)
    value_history = [0.0 for _ in 1:num_plot_points]
    for i in 1:num_plot_points
        result = boptimize!(opt)
        value_history[i] = result.observed_optimum
    end
    x = [i for i in 1:num_plot_points]
    # plot(x, value_history, label="Bayesian Optimization", seriestype=:scatter, xlabel="Iterations", ylabel="Total Heat Output")
    plot(x, value_history, title="Bayesian Optimization", marker=:circle,  ylabel="Total Heat Output", xlabel="Iterations")  # Add markers ('o') at each data point
    savefig("jl_plots/bayesian_optim.png")
end
 
# Bayesian Optimization with Latin Hypercube Sampling
function plot_bayesian_lhs(px_py, grid_size; num_plot_points = 10)
    opt = bayesian_helper(px_py, grid_size; lhs=true)
    value_history = [0.0 for _ in 1:num_plot_points]

    for i in 1:num_plot_points
        result = boptimize!(opt)
        value_history[i] = result.observed_optimum
    end
    x = [i for i in 1:num_plot_points]
    plot(x, value_history, title="Bayesian Optimization with LHS", marker=:circle, xlabel="Iterations", ylabel="Total Heat Output")
    savefig("jl_plots/bayesian_optim_lhs.png")
end

function plot_all(grid_size, num_wells)
    num_dims = 2 * num_wells
    points = [rand(0:grid_size) for _ in 1:num_dims]

    # Adaptive Differential Evolution Optimizer
    adaptive_diff_evo_optim = bboptimize(f_heat; SearchRange = (0.0, grid_size), NumDimensions = num_dims, Method = :adaptive_de_rand_1_bin)
    # adaptive_diff_evo_optim = bboptimize(f_heat; SearchRange = (0.0, grid_size), NumDimensions = num_dims, Method = :probabilistic_descent)
    x1 = best_candidate(adaptive_diff_evo_optim)[1:2:end]
    y1 = best_candidate(adaptive_diff_evo_optim)[2:2:end]

    # # Random Search
    rand_search = bboptimize(f_heat; SearchRange = (0.0, grid_size), NumDimensions = num_dims, Method=:random_search, MaxTime = 30.0)
    x2 = best_candidate(rand_search)[1:2:end]
    y2 = best_candidate(rand_search)[2:2:end]

    # # Bayesian Optimization
    bayesian_optim = bayesian_optimization(fill(0.0, num_dims), Int(grid_size))
    x3 = bayesian_optim.observed_optimizer[1:2:end]
    y3 = bayesian_optim.observed_optimizer[2:2:end]

    # Latin Hypercube Sampling with Gradient Descent
    lhs = latin_hypercube_sampling(f_heat, points, grid_size).minimizer
    x4 = lhs[1:2:end]
    y4 = lhs[2:2:end]

    ## Plotting
    plot(
        plot(x1, y1, label="Adaptive Differential Evolution Optimizer", xlabel="x", ylabel="y", seriestype=:scatter, marker=:circle),
        plot(x2, y2, label="Random Search", seriestype=:scatter, marker=:star),
        plot(x3, y3, label="Bayesian Optimization", seriestype=:scatter, marker=:cross),
        plot(x4, y4, label="Latin Hypercube Sampling with GD", seriestype=:scatter, marker=:x),
        layout=(2, 2), 
        size =(1000, 1000),
        title="Well Locations"
    )
    plot!(legend=:outerbottom)
    savefig("myplot.png") 
end

function main()
    grid_size = 100.0
    num_wells = 40
    # plot_all(grid_size, num_wells)
    plot_bayesian_lhs(fill(0.0, 2*num_wells), Int(grid_size); num_plot_points=100)
end

# main()

# TODO: a new objective function that computes the energy output for the whole array (think about how to handle the Dirichlet boundary conditions)
# TODO: a plot of energy output against time for different optimization algorithms
