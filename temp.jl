# using Pkg
# Pkg.add("BlackBoxOptim")
using BlackBoxOptim
using Plots
using BayesianOptimization, GaussianProcesses, Distributions

grid_size = 100.0
num_wells = 20
num_dims = 2 * num_wells

# Temporary function: Returns the total heat output of all the wells given their locations
function f_heat(px_py)
    px, py = px_py[1:2:end], px_py[2:2:end]
    sum_heat = 0.0
    for (i, p_x) in pairs(px)
        p_y = py[i]
        for (j, q_x) in pairs(px[i+1:end])
            q_y = py[j]
            sum_heat += sqrt((p_x - q_x)^2 + (p_y - q_y)^2)
        end
    end
    return -sum_heat
end


function bayesian_optimization(px_py, grid_size)
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
                maxiterations = 200, 
                sense = Min, 
                acquisitionoptions = (method = :LD_LBFGS, restarts = 5, maxtime = 0.1, maxeval = 1000), 
                verbosity = Progress)
    result = boptimize!(opt)
    return result
end

# # Adaptive Differential Evolution Optimizer
adaptive_diff_evo_optim = bboptimize(f_heat; SearchRange = (0.0, grid_size), NumDimensions = num_dims, MaxTime = 30.0)
# print("Adaptive Differential Evolution Optimizer: ", best_candidate(adaptive_diff_evo_optim), "\n")

# # Random Search
rand_search = bboptimize(f_heat; SearchRange = (0.0, grid_size), NumDimensions = num_dims, Method=:random_search, MaxTime = 30.0)
# print("Random Search: ", best_candidate(rand_search), "\n")

# Bayesian Optimization
bayesian_optim = bayesian_optimization(fill(0.0, num_dims), Int(grid_size))

## Plotting

x1 = best_candidate(adaptive_diff_evo_optim)[1:2:end]
y1 = best_candidate(adaptive_diff_evo_optim)[2:2:end]

x2 = best_candidate(rand_search)[1:2:end]
y2 = best_candidate(rand_search)[2:2:end]

x3 = bayesian_optim.observed_optimizer[1:2:end]
y3 = bayesian_optim.observed_optimizer[2:2:end]
plot(x1, y1, label="Adaptive Differential Evolution Optimizer", xlabel="x", ylabel="y", title="Well Locations", seriestype=:scatter)
plot!(x2, y2, label="Random Search", seriestype=:scatter)
plot!(x3, y3, label="Bayesian Optimization", seriestype=:scatter)
savefig("myplot.png") 

