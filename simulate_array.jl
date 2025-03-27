using LinearAlgebra
using .Threads
## TODO: use SUPERPOSITION i.e model for one well after another and update the temperatures
## Diffusion convection equation for temperature in a pipe-in-pipe geometry ##
#
# ρ c ∂ϕ/∂t =  ∂(λ ∂ϕ/∂x)/∂x + ∂(λ ∂ϕ/∂y)/∂y + ∂(λ ∂ϕ/∂z)/∂z
#                        -(ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
#                        + S

### Physical parameters ###

# Maximum simulation time [s]
sim_time = 240 # 4 minutes

ϕs = 293.15 # Earth surface temperature [K], 20 °C

ϕf0 = 293.15 # Fluid Initial temperature [K], 20 °C

## Rock: granite ##
# Rock density [kg/m3]
ρr = 2750.0
# Rock specific heat [J/(kg °C)]
cr = 790.0
# Rock thermal conductivity [W/(m °C)]
λr = 2.62
# Rock diffusion coefficient
dr = λr/(ρr*cr)
# Rock temperature as a function of depth [°C]
ϕ0(d) = ϕs+0.5*d

## Pipes: polyethylene ##
r1 = 0.1 # Inner pipe inside radius [m]
t1 = 0.01 # Inner pipe thickness [m]
h1 = 8.5 # Inner pipe height [m]
# Outer pipe inside radius [m]
r2 = sqrt(r1^2 + (r1+t1)^2) # such that inflow volume == outflow volume. NOTE: This is NOT strictly necessary, discuss in thesis
# Could be interesting to see the impact of having different cross-sectional area between annulus and ring space
t2 = 0.01 # Outer pipe thickness [m]
h2 = 9 # Outer pipe height [m]
ε = 1 # Porosity: ratio of liquid volume to the total volume, [Anything similar in the literature?]
ρp = 961 # Pipe density [kg/m3]
cp = 2900.0 # Pipe specific heat [J/(kg °C)]
λp = 0.54 # Pipe thermal conductivity [W/(m °C)]
dp = λp/(ρp*cp) # Pipe diffusion coefficient

## Fluid: water ##
ρf = 997 # Fluid density [kg/m3]
cf = 4184.0 # Fluid specific heat capacity [J/(kg °C)]
λf = 0.6 # Fluid thermal conductivity [W/(m °C)]
df = λf/(ρf*cf) # Fluid diffusion coefficient
uf = 0.01 # Flow speed [m/s]
vx0 = uf # In-flow velocity
vy0 = 0
vz0 = 0
Lf = 2r1 # Characteristic linear dimension (diameter of the pipe) [m]
μf = 0.00089 # Fluid dynamic viscosity at 25 °C [Pa⋅s], 0.0005465 at 50 °C

# See https://gchem.cm.utexas.edu/data/section2.php?target=heat-capacities.php
#     https://en.wikipedia.org/wiki/High-density_polyethylene
#     https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation


## Numerical parameters ##

# Geometric distances [m]
xx = 100 # [m]
yy = 100 # [m]
zz = 20 # [m]

dx = 1 # [m]
dy = 1 # [m]
dz = 0.125 # [m]

# dt using stability condition (check this)
dtd = (1/(2*maximum([dr,dp,df]))*(1/dx^2+1/dy^2+1/dy^2)^-1)
dtc = minimum([dx/norm(vx0), dy/norm(vy0), dz/norm(vz0)])
dt = minimum([dtd,dtc]) 

# No. of spatial domain nodes
ii = round(Int,xx/dx)
jj = round(Int,yy/dy)
kk = round(Int,zz/dz)

# Define temperature matrices: ϕ2, ϕ1
ϕ2 = ones(ii,jj,kk)
ϕ1 = ones(ii,jj,kk)

## Initial and boundary conditions ##
# Note: we take flow in the annulus to be upwards here (does not have to be the case)
function init!(px, py, ϕ1, ϕ2)
    if h2 + 5 > zz
        throw(ArgumentError("h2 + 10 > zz, well too deep for the simulation height"))
    end
    d = zeros(ii,jj,kk)
    vx = zeros(ii,jj,kk)
    vy = zeros(ii,jj,kk)
    vz = zeros(ii,jj,kk)
    for k in 1:kk
        zk = k*dz
        for j in 1:jj
            yj = j*dy
            for i in 1:ii
                xi = i*dx
                # Rock
                d[i,j,k] = dr
                ϕ2[i,j,k] = ϕ0(zk) 
                # Well
                for (idx0, idx1) in zip(px, py)
                    r = norm([xi,yj]-[idx0,idx1]) 
                    if r < r2 + t2  # inside the confines of some well
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
                        else
                            d[i, j, k] = dp
                            vx[i, j, k] = 0
                            vy[i, j, k] = 0
                            vz[i, j, k] = 0
                            ϕ2[i, j, k] = ϕs
                        end
                        break
                    end
                end
            end
        end
    end
    ϕ1 .= ϕ2
    return d, vx, vy, vz
end

# Solve eq. system
function updateϕ_domain!(ϕ2,ϕ1,d,vx,vy,vz,dx,dy,dz,dt,px,py,source=0.0)
    for k = 2:kk-1 # z-axis
        @threads for j = 2:jj-1 # y-axis
            for i = 2:ii-1 # x-axis
                xi, yj = i * dx, j * dy
                conv = 0
                for (xc, yc) in zip(px, py)
                    r = norm([xi, yj] - [xc, yc])
                    if r <= r2 + t2  # inside the confines of some well
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
                        break 
                    end
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

function updateϕ_boundaries!(ϕ,ii,jj,kk,dx,dy,dz)
     # Addresses the special case of cells without some of its 6 neighbors
    # TODO: Problem: does not correctly address the fluid in the pipe
    # TODO: Is this the best way? How about setting the derivatives to zero in the directions with no neighbors?
    ϕ[1,2:jj-1,2:kk-1] .= ϕ[2,2:jj-1,2:kk-1]
    ϕ[ii,2:jj-1,2:kk-1] .= ϕ[ii-1,2:jj-1,2:kk-1]
    ϕ[2:ii-1,1,2:kk-1] .= ϕ[2:ii-1,2,2:kk-1]
    ϕ[2:ii-1,jj,2:kk-1] .= ϕ[2:ii-1,jj-1,2:kk-1]
    ϕ[2:ii-1,2:jj-1,1] .= ϕ[2:ii-1,2:jj-1,2]
    ϕ[2:ii-1,2:jj-1,kk] .= ϕ0(kk*dz)
end

# Run the simulation
"""
    total_energy(position_array, time)
    Return the total amount of heat energy produced by the array of wells 
    at positions `position_array` for a specified period of time `time`
"""
function run_array_simulation(position_array, sim_time)
    # position_array: [[x1, x2, ...], [y1, y2, ...]]
    # sim_time: simulation time in seconds

    num_iters = round(Int,sim_time/dt) # No. of time iterations
    px = position_array[1]
    py = position_array[2]
    d, vx, vy, vz = init!(px, py, ϕ1, ϕ2)

    cumulative_heat_extracted = 0.0
    for t = 0:2:num_iters
        # Update ϕ
        updateϕ_domain!(ϕ2,ϕ1,d,vx,vy,vz,dx,dy,dz,dt,px,py)
        updateϕ_boundaries!(ϕ2,ii,jj,kk,dx,dy,dz)
        
        # Update ϕ
        updateϕ_domain!(ϕ1,ϕ2,d,vx,vy,vz,dx,dy,dz,dt,px, py)
        updateϕ_boundaries!(ϕ1,ii,jj,kk,dx,dy,dz)
         
        # Compute energy output
        for (xc, yc) in zip(px, py)
            # NOTE: assumes the annulus is the production terminal
            Δϕ = ϕ2[xc, yc, 1] - ϕf0 
            cumulative_heat_extracted -= π * r2^2 * (-uf) * 2 * ρf * cf * Δϕ
        end
    end
    print("Total Heat Produced:", cumulative_heat_extracted, "J\n")
    return cumulative_heat_extracted
end


# Well Positions Array i.e each position is (px[i], py[i])
px = [25, 50, 75]
py = [25, 50, 75]

run_array_simulation([px, py], sim_time)
