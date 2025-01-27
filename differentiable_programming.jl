using Zygote 
# include("geothermal.jl")
# include("array_model.jl")

# Example Function
function example_func(px_py)
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

# The high-level function "gradient" supports differentiable programming. 
# More information can be found here: https://fluxml.ai/Zygote.jl/stable/

print(gradient(X -> example_func(X), [2, 2, 4, 4]))



