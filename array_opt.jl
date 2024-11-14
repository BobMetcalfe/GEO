using BlackBoxOptim
using Plots
using BayesianOptimization, GaussianProcesses, Distributions
using Optim, Hyperopt
using Random

## TODO
# - Add reference papers below
# --> probabilistic descent: https://arxiv.org/pdf/2204.01275
# - Debug the optimization code below to get sensible outputs
# - IMPORTANT: Differentiable Programming approaches in Julia

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
    return -sum_heat
end

## Bayesian Optimization
function bayesian_optimization(px_py, grid_size)
    # TODO: is a normal prior reasonable in this case
    num_dims = length(px_py)
    model = ElasticGPE( num_dims, 
                        mean = MeanConst(grid_size/2.0), 
                        kernel = SEArd(zeros(num_dims), 5.), 
                        logNoise = 0.0, 
                        capacity = 3000) 
    # set_priors!(model.mean, [Normal(1, 2)]) # if not uniform prior (i.e Normal prior)
    modeloptimizer = MAPGPOptimizer(every = 10, 
                                    # noisebounds = [fill(-4, num_dims), fill(3, num_dims)], 
                                    # kernbounds = [[-1, -1, 0], [4, 4, 10]], 
                                    maxeval = 40)
    opt = BOpt(f_heat, 
                model, 
                UpperConfidenceBound(), 
                modeloptimizer, 
                fill(0, num_dims), 
                fill(grid_size, num_dims), 
                repetitions = 5, 
                maxiterations = 400, 
                sense = Min, 
                acquisitionoptions = (method = :LD_LBFGS, restarts = 5, maxtime = 0.1, maxeval = 1000), 
                verbosity = Progress)
    result = boptimize!(opt)
    return result
end
 
## Latin Hypercube Sampling
function latin_hypercube_sampling(f, points, grid_size)
    if length(points) % 2 != 0
        error("The points array must have an even number of elements representing coordinate pairs.")
    end
    candidates = [LinRange(0, grid_size, Int(10 * grid_size)) for _ in points]  
    objective = function (resources, points)
        lower = [0 for _ in 1:length(points)]
        upper = [grid_size for _ in 1:length(points)]
        res = Optim.optimize(f, lower, upper, points, Optim.Fminbox(Optim.GradientDescent()), Optim.Options(time_limit=resources/100))
        Optim.minimum(res), Optim.minimizer(res)
    end
    hohb = hyperband(objective, candidates; R=50, η=3, threads=true, inner=LHSampler())
    return hohb
end

## Differential Programming Approach





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
        title="Well Locations"
    )
    plot!(legend=:outerbottom)
    savefig("myplot.png") 
end

grid_size = 100.0
num_wells = 40
plot_all(grid_size, num_wells)