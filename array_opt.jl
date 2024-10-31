using Hyperopt
# using Enzyme
using Optim
using ForwardDiff

# Temporary function: Returns the total heat output of all the wells given their locations
function total_heat_output(px_py::Vector{Float64})
   # px, py = px_py[1:2:end], px_py[2:2:end]
   # len = max(length(px), length(py))
   # well_production_heat = 1000 # J, assumed for now
   # return - well_production_heat * len
   return - (px_py[1]^2 - px_py[2]^2 + px_py[3]^2 - px_py[4]^2)
end

function combine_coords(px, py)
   px_py = []
   for (x, y) in zip(px, py)
      push!(px_py, x)
      push!(px_py, y)
   end
   return px_py
end

# Bayesian Optimization
function bayesian_optimization(px, py, grid_size)
   px_py = combine_coords(px, py)
   len = length(px_py)
   objective = function (resources, px_py)
      res = Optim.optimize(total_heat_output, px_py, SimulatedAnnealing(), Optim.Options(time_limit=resources/100))
      Optim.minimum(res), Optim.minimizer(res)
   end
   candidates = [LinRange(0, grid_size, Int(10*grid_size)) for _ in 1:len]
   dims = [Hyperopt.Continuous() for _ in 1:len]
   hohb = hyperband(objective, candidates, inner=BOHB(dims=dims); R=50, η=3, threads=true)
   return hohb
end


function latin_hypercube_sampling(px, py, grid_size)
   px_py = combine_coords(px, py)
   len = length(px_py)
   objective = function (resources, px_py)
      res = Optim.optimize(total_heat_output, px_py, SimulatedAnnealing(), Optim.Options(time_limit=resources/100))
      Optim.minimum(res), Optim.minimizer(res)
   end
   candidates = [LinRange(0, grid_size, 10*grid_size) for _ in 1:len]
   hohb = hyperband(objective, candidates, inner=LHSampler(); R=50, η=3, threads=true)
   return hohb
end

# Random Sampling
function random_sampling(px, py, grid_size)
   px_py = combine_coords(px, py)
   len = length(px_py)
   objective = function (resources, px_py)
      res = Optim.optimize(total_heat_output, px_py, SimulatedAnnealing(), Optim.Options(time_limit=resources/100))
      Optim.minimum(res), Optim.minimizer(res)
   end
   candidates = [LinRange(0, grid_size, 10*grid_size) for _ in 1:len]
   hohb = hyperband(objective, candidates, inner=RandomSampler(); R=50, η=3, threads=true)
   return hohb
end

function combine_coords(px, py)
   px_py = []
   for (x, y) in zip(px, py)
      push!(px_py, x)
      push!(px_py, y)
   end
   return px_py
end

# Gradient-based optimization methods
function gradient_based(f, px, py, grid_size, tol, max_iter, algorithm)
   px_py = Vector{Float64}(combine_coords(px, py))
   lower = [0.0 for _ in 1:length(px_py)]
   upper = [grid_size for _ in 1:length(px_py)]
   res = Optim.optimize(f, lower, upper, px_py, Fminbox(algorithm), Optim.Options(g_tol=tol, iterations=max_iter))
   return Optim.minimizer(res)
end

# Naive Gradient Descent
function gradient_descent(f, px, py, grid_size, tol, max_iter)
   return gradient_based(f, px, py, grid_size, tol, max_iter, GradientDescent())
end

# ADAM
function adam(f, px, py, grid_size, tol, max_iter)
   # TODO: add box constraints instead of projecting the result. Cannot use FminBox with Adam yet in optim
   # return gradient_based(f, px, py, grid_size, tol, max_iter, Adam())
   px_py = Vector{Float64}(combine_coords(px, py))
   res = Optim.optimize(f, px_py, Optim.Adam(epsilon=0.001), Optim.Options(g_tol=tol, iterations=max_iter))
   result = Optim.minimizer(res)
   for i in 1:length(result)
      result[i] = clamp(result[i], 0.0, grid_size)
   end
   return result
end


# bayesian_optimization([1, 2], [2, 1], 20)

# adam(total_heat_output, [1.0, 2.0], [2.0, 1.0], 100.0, 0.001, 10000)
# gradient_descent(total_heat_output, [1.0, 2.0], [2.0, 1.0], 100.0, 0.001, 10000)
bayesian_optimization([1.0, 2.0], [2.0, 1.0], 100.0)



