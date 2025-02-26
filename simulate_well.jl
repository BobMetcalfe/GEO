using LinearAlgebra
using .Threads

########
# Diffusion convection equation for temperature in a pipe-in-pipe geometry
#
# Equation: ρ c ∂ϕ/∂t =  ∂(λ ∂ϕ/∂x)/∂x + ∂(λ ∂ϕ/∂y)/∂y + ∂(λ ∂ϕ/∂z)/∂z
#                        -(ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
#                        + S
########

### Physical parameters ###

sim_time = 240 # Maximum simulation time [s], 40 minutes

ϕs = 293.15 # Earth surface temperature [K], 20 °C

ϕf0 = 293.15 # Fluid Initial temperature [K], 20 °C

## Rock: granite ##
ρr = 2750 # Rock density [kg/m3]
cr = 790 # Rock specific heat [J/(kg K)]
λr = 2.62 # Rock thermal conductivity [W/(m K)]

## Pipes: polyethylene ##
r1 = 0.1 # Inner pipe inside radius [m]
t1 = 0.01 # Inner pipe thickness [m]
h1 = 8.5 # Inner pipe height [m]
t2 = 0.01 # Outer pipe thickness [m]
h2 = 9 # Outer pipe height [m]
ε = 1 # Porosity: ratio of liquid volume to the total volume
ρp = 961 # Pipe density [kg/m3]
cp = 2900 # Pipe specific heat [J/(kg K)]
λp = 0.54 # Pipe thermal conductivity [W/(m K)]

## Fluid: water ##
ρf = 997 # Fluid density [kg/m3]
cf = 4184 # Fluid specific heat capacity [J/(kg K)]
λf = 0.6 # Fluid thermal conductivity [W/(m K)]
uf = 0.01 # Flow speed [m/s]
μf = 0.00089 # Fluid dynamic viscosity at 25 °C [Pa⋅s], 0.0005465 at 50 °C

# See https://gchem.cm.utexas.edu/data/section2.php?target=heat-capacities.php
#     https://en.wikipedia.org/wiki/High-density_polyethylene
#     https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation


## Numerical parameters ##

# Geometry distances [m]
xx = 1
yy = 1
zz = h2 + 5 # compared to the length of outer pipe

# dx, dy, dz [m]
dx = 0.01
dy = 0.01
dz = 0.125

# Solve eq. system
function updateϕ_domain!(ϕ2, ϕ1, d, vx, vy, vz, dx, dy, dz, dt, xc, yc, r1, t1, r2, ε, ii, jj, kk, source=0.0)
    # source: the source term in the heat equation
    for k = 2:kk-1
        @threads for j = 2:jj-1
            for i = 2:ii-1
                # Convective term
                xi, yj = i * dx, j * dy
                r = norm([xi, yj] - [xc, yc])
                if r < r1  # inside inner pipe 
                    conv = (ε * vx[i, j, k] * (ϕ1[i+1, j, k] - ϕ1[i, j, k]) / dx
                            + ε * vy[i, j, k] * (ϕ1[i, j+1, k] - ϕ1[i, j, k]) / dy
                            + ε * vz[i, j, k] * (ϕ1[i, j, k+1] - ϕ1[i, j, k]) / dz)
                elseif r1 + t1 < r < r2 # between inner and outer pipe 
                    conv = (ε * vx[i, j, k] * (ϕ1[i, j, k] - ϕ1[i-1, j, k]) / dx
                            + ε * vy[i, j, k] * (ϕ1[i, j, k] - ϕ1[i, j-1, k]) / dy
                            + ε * vz[i, j, k] * (ϕ1[i, j, k] - ϕ1[i, j, k-1]) / dz)
                else
                    conv = 0
                end
                # Diffusive term
                diff = (((d[i+1, j, k] + d[i, j, k]) / 2 * (ϕ1[i+1, j, k] - ϕ1[i, j, k]) / dx
                         -
                         (d[i, j, k] + d[i-1, j, k]) / 2 * (ϕ1[i, j, k] - ϕ1[i-1, j, k]) / dx) / dx
                        + ((d[i, j+1, k] + d[i, j, k]) / 2 * (ϕ1[i, j+1, k] - ϕ1[i, j, k]) / dy
                           -
                           (d[i, j, k] + d[i, j-1, k]) / 2 * (ϕ1[i, j, k] - ϕ1[i, j-1, k]) / dy) / dy
                        + ((d[i, j, k+1] + d[i, j, k]) / 2 * (ϕ1[i, j, k+1] - ϕ1[i, j, k]) / dz
                           -
                           (d[i, j, k] + d[i, j, k-1]) / 2 * (ϕ1[i, j, k] - ϕ1[i, j, k-1]) / dz) / dz)
                # Update temperature
                ϕ2[i, j, k] = (diff - conv + source) * dt + ϕ1[i, j, k]
            end
        end
    end
end

function updateϕ_boundaries!(ϕ, ii, jj, kk, dx, dy, dz, ϕ0)
    ϕ[1, 2:jj-1, 2:kk-1] .= ϕ[2, 2:jj-1, 2:kk-1]
    ϕ[ii, 2:jj-1, 2:kk-1] .= ϕ[ii-1, 2:jj-1, 2:kk-1]
    ϕ[2:ii-1, 1, 2:kk-1] .= ϕ[2:ii-1, 2, 2:kk-1]
    ϕ[2:ii-1, jj, 2:kk-1] .= ϕ[2:ii-1, jj-1, 2:kk-1]
    ϕ[2:ii-1, 2:jj-1, 1] .= ϕ[2:ii-1, 2:jj-1, 2]
    ϕ[2:ii-1, 2:jj-1, kk] .= ϕ0(kk * dz)
end

function init(ii, jj, kk, dx, dy, dz, r1, t1, r2, dr, dp, df, uf, ϕf0, ϕ1, ϕ2, ϕ0)
    ## Initial and boundary conditions ##
    d = zeros(ii, jj, kk)
    vx = zeros(ii, jj, kk)
    vy = zeros(ii, jj, kk)
    vz = zeros(ii, jj, kk)
    xc, yc = ii ÷ 2 * dx, jj ÷ 2 * dy
    for k in 1:kk
        zk = k * dz
        for j in 1:jj
            yj = j * dy
            for i in 1:ii
                xi = i * dx
                r = norm([xi, yj] - [xc, yc])
                if r < r1  # inside inner pipe
                    d[i, j, k] = df
                    vx[i, j, k] = 0
                    vy[i, j, k] = 0
                    vz[i, j, k] = -uf
                    ϕ2[i, j, k] = ϕf0
                elseif r < r1 + t1  # inner pipe itself
                    d[i, j, k] = dp
                    vx[i, j, k] = 0
                    vy[i, j, k] = 0
                    vz[i, j, k] = 0
                    ϕ2[i, j, k] = ϕs
                elseif r < r2  # between inner and outer pipe 
                    d[i, j, k] = df
                    vx[i, j, k] = 0
                    vy[i, j, k] = 0
                    vz[i, j, k] = uf
                    ϕ2[i, j, k] = ϕf0
                elseif r < r2 + t2  # outer pipe
                    d[i, j, k] = dp
                    vx[i, j, k] = 0
                    vy[i, j, k] = 0
                    vz[i, j, k] = 0
                    ϕ2[i, j, k] = ϕs
                else # rock
                    d[i, j, k] = dr
                    vx[i, j, k] = 0
                    vy[i, j, k] = 0
                    vz[i, j, k] = 0
                    ϕ2[i, j, k] = ϕ0(zk)
                end
            end
        end
    end
    ϕ1 .= ϕ2
    return d, vx, vy, vz, xc, yc
end

function run_well_simulation(sim_time, ϕs, ϕf0, ρr, cr, λr, r1, t1, h1, t2, h2, ε, ρp, cp, λp, ρf, cf, λf, uf, μf, xx, yy, zz, dx, dy, dz)
    # Rock diffusion coefficient
    dr = λr / (ρr * cr)

    # Rock temperature as a function of depth [°C]
    ϕ0(d) = ϕs + 0.5 * d
    r2 = sqrt((r1^2 + (r1 + t1)^2)) #  vol1 = vol2, NOTE: does not have to be this way

    dp = λp / (ρp * cp) # Pipe diffusion coefficient

    df = λf / (ρf * cf)  # Fluid diffusion coefficient

    # Flow speed [m/s]
    vx0 = uf
    vy0 = 0
    vz0 = 0

    # dt using stability condition (check this)
    dtd = (1 / (2 * maximum([dr, dp, df])) * (1 / dx^2 + 1 / dy^2 + 1 / dy^2)^-1)
    dtc = minimum([dx / norm(vx0), dy / norm(vy0), dz / norm(vz0)])
    dt = minimum([dtd, dtc])

    tt = round(Int, sim_time / dt) # No. of time iterations

    # No. of spatial domain nodes
    ii = round(Int, xx / dx)
    jj = round(Int, yy / dy)
    kk = round(Int, zz / dz)
    nn = ii * jj * kk

    # No. of height nodes
    hh1 = round(Int, h1 / dz)
    hh2 = round(Int, h2 / dz)

    # Define temperature matrices: ϕ2, ϕ1
    ϕ2 = ones(ii, jj, kk)
    ϕ1 = ones(ii, jj, kk)

    ## Initial and boundary conditions ##
    d, vx, vy, vz, xc, yc = init(ii, jj, kk, dx, dy, dz, r1, t1, r2, dr, dp, df, uf, ϕf0, ϕ1, ϕ2, ϕ0)

    vels = Array{Float64}(undef, 3, ii, jj, kk)
    vels[1, :, :, :] = vx
    vels[2, :, :, :] = vy
    vels[3, :, :, :] = vz

    cumulative_heat_extracted = 0
    for t = 0:2:tt # run the simulation 
        # Update ϕ
        updateϕ_domain!(ϕ2, ϕ1, d, vx, vy, vz, dx, dy, dz, dt, xc, yc, r1, t1, r2, ε, ii, jj, kk)
        updateϕ_boundaries!(ϕ2, ii, jj, kk, dx, dy, dz, ϕ0)

        # Update ϕ
        updateϕ_domain!(ϕ1, ϕ2, d, vx, vy, vz, dx, dy, dz, dt, xc, yc, r1, t1, r2, ε, ii, jj, kk)
        updateϕ_boundaries!(ϕ1, ii, jj, kk, dx, dy, dz, ϕ0)

        Δϕ = ϕ2[Int(floor((ii+1)/2)), Int(floor((jj+1)/2)), 1] - ϕf0

        cumulative_heat_extracted -= π * r1^2 * vz[Int(floor((ii+1)/2)), Int(floor((jj+1)/2)), 1] * 2 * ρf * cf * Δϕ
    end
    return cumulative_heat_extracted
end

function main()
    total_heat_produced = run_well_simulation(sim_time, ϕs, ϕf0, ρr, cr, λr, r1, t1, h1, t2, h2, ε, ρp, cp, λp, ρf, cf, λf, uf, μf, xx, yy, zz, dx, dy, dz)
    print("Total Heat Produced:", total_heat_produced, "J\n")
    print("Production Capacity:", total_heat_produced / sim_time, "W\n")
end

main()
    