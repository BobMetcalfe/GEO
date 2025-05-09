using Plots
using Optim
using BlackBoxOptim
using Optimization, OptimizationBBO, OptimizationOptimJL

## https://docs.sciml.ai/Optimization/stable/optimization_packages/optim/

function simulated_annealing(original_f, px, py, bounds; 
                    max_iters=20,
                    plot_name="../jl_plots/simulated_annealing_optim.png", 
                    plot_title="Simulated Annealing Optimization"
    )

    pxpy = [px..., py...]
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]

    history = []
    final_config = pxpy
    obj_val = 0.0

    function f(x)
        val = original_f(x)
        push!(history, max(val, obj_val))
        if val >= obj_val
            obj_val = val
            final_config = x
        end
        return -val  
    end

    problem = Optimization.OptimizationProblem((u,_)->f(u), pxpy, lb=lb, ub=ub)
    results = solve(problem, Optim.SAMIN(), maxiters=max_iters)

    plot( history; 
          marker=:circle, 
          ylabel="Total Heat Energy Output from Array (J)", 
          xlabel="Objective Function Evaluations", 
         )  
    savefig(plot_name)
    println("Simulated Annealing Figure saved at $plot_name")
    println(results)

    px = final_config[1:lastindex(px)]
    py = final_config[lastindex(px)+1:end]
    return [(px[i], py[i]) for i in 1:lastindex(px)]
end

function test()
    rosenbrock(x) = (1.0 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    px = [0.0]
    py = [0.0]
    simulated_annealing(rosenbrock, px, py, bounds)
end

# test()
