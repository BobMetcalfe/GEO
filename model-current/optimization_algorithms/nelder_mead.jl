using Plots
using Optim
using BlackBoxOptim
using Optimization, OptimizationBBO

# Julia impl. of NelderMead does not support box constraints
# # https://julianlsolvers.github.io/Optim.jl/v0.9.3/algo/nelder_mead

function nelder_mead(original_f, px, py, bounds; 
                    max_iters=20,
                    plot_name="../jl_plots/nelder_mead_optim.png", 
                    plot_title="Nelder-Mead Optimization"
    )

    pxpy = [px..., py...]
    lbs = [b[1] for b in bounds]
    ubs = [b[2] for b in bounds]

    history = []
    final_config = [(px[i], py[i]) for i in 1:lastindex(px)]
    obj_val = 0.0

    function f(x)
        # if maximum(x) > bounds[1][2] # temp solution
        #     return Inf
        # end
        val = original_f(x)
        push!(history, max(val, obj_val))
        if val >= obj_val
            obj_val = val
            final_config = x
        end
        return -val  
    end

    options = Optim.Options(iterations=max_iters)
    results = optimize(f, lbs, ubs, pxpy, Fminbox(NelderMead()), options)

    println("Best solution found by Nelder-Mead algorithm:", results.minimizer)

    plot( history; 
          marker=:circle, 
          ylabel="Total Heat Energy Output from Array (J)", 
          xlabel="Objective Function Evaluations", 
         )  
    savefig(plot_name)
    println("Nelder-Mead Figure saved at $plot_name")
    println(results)

    px = final_config[1:lastindex(px)]
    py = final_config[lastindex(px)+1:end]
    return [(px[i], py[i]) for i in 1:lastindex(px)]
end

function test()
    rosenbrock(x) = -((1.0 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    px = [0.0]
    py = [0.0]
    nelder_mead(rosenbrock, px, py, bounds)
end

# test()
