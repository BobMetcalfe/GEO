using LinearAlgebra
using .Threads
##### TODO: use SUPERPOSITION i.e model for one well after another and update the temperatures
## Diffusion convection equation for temperature in a pipe-in-pipe geometry ##
#
# ρ c ∂ϕ/∂t =  ∂(λ ∂ϕ/∂x)/∂x + ∂(λ ∂ϕ/∂y)/∂y + ∂(λ ∂ϕ/∂z)/∂z
#                        -(ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
#                        + S

### Physical parameters ###

# Maximum simulation time [s]
sim_time = 240 # 4 minutes

# Earth surface temperature [°C]
ϕs = 20

## Rock: granite ##
# Rock density [g/m3]
ρr = 2750000
# Rock specific heat [J/(g °C)]
cr = 0.790
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
r2 = sqrt(r1^2 + (r1+t1)^2) # such that inflow volume == outflow volume [This is not strictly necessary, discuss this choice in thesis] 
t2 = 0.01 # Outer pipe thickness [m]
h2 = 9 # Outer pipe height [m]
ε = 1 # Porosity: ratio of liquid volume to the total volume, [Anything similar in the literature?]
ρp = 961000 # Pipe density [g/m3]
cp = 2.9 # Pipe specific heat [J/(g °C)]
λp = 0.54 # Pipe thermal conductivity [W/(m °C)]
dp = λp/(ρp*cp) # Pipe diffusion coefficient

# Fluid: water #########################################
ρf = 997000 # Fluid density [g/m3]
cf = 4.184 # Fluid specific heat capacity [J/(g °C)]
λf = 0.6 # Fluid thermal conductivity [W/(m °C)]
df = λf/(ρf*cf) # Fluid diffusion coefficient
uf = 0.01 # Flow speed [m/s]
vx0 = uf
vy0 = 0
vz0 = 0
Lf = 2r1 # Characteristic linear dimension (diameter of the pipe) [m]
μf = 0.00089 # Fluid dynamic viscosity at 25 °C [Pa⋅s], 0.0005465 at 50 °C

# Reynolds number
Re = ρf*uf*Lf/μf

# See https://gchem.cm.utexas.edu/data/section2.php?target=heat-capacities.php
#     https://en.wikipedia.org/wiki/High-density_polyethylene
#     https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation


## Numerical parameters ##

# Geometric distances [m]
xx = 100
yy = 100
zz = h2+1

dx = 1 # [m]
dy = 1 # [m]
dz = 0.125 # [m]
rx = 0:dx:xx
ry = 0:dy:yy
rz = 0:dz:zz

# dt using stability condition (check this)
dtd = (1/(2*maximum([dr,dp,df]))*(1/dx^2+1/dy^2+1/dy^2)^-1)
dtc = minimum([dx/norm(vx0), dy/norm(vy0), dz/norm(vz0)])
dt = minimum([dtd,dtc])

# No. of time iterations
tt = round(Int,sim_time/dt) 

# No. of spatial domain nodes
ii = round(Int,xx/dx)
jj = round(Int,yy/dy)
kk = round(Int,zz/dz)
nn = ii * jj * kk

# No. of height nodes
hh1 = round(Int,h1/dz)
hh2 = round(Int,h2/dz)

# Define temperature matrices: ϕ2, ϕ1
ϕ2 = ones(ii,jj,kk)
ϕ1 = ones(ii,jj,kk)

## Initial and boundary conditions ##
function init(px, py, ϕ1, ϕ2)
    d = zeros(ii,jj,kk)
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
                    if r < r2  # inside the well
                        d[i,j,k] = 0 # Dirichlet boundary
                        ϕ2[i,j,k] = ϕ0(zk)
                        break
                    end
                end
            end
        end
    end
    ϕ1 .= ϕ2
end

# Solve eq. system
function updateϕ_domain!(ϕ2,ϕ1,d,dx,dy,dz,dt,source=0.0)
    @threads for k = 2:kk-1
        for j = 2:jj-1
            for i = 2:ii-1
                if d[i,j,k] > 0  # in the rock formation
                    # Diffusive term
                    diff = (((d[i+1,j,k]+d[i,j,k])/2*(ϕ1[i+1,j,k]-ϕ1[i,j,k])/dx 
                            -(d[i,j,k]+d[i-1,j,k])/2*(ϕ1[i,j,k]-ϕ1[i-1,j,k])/dx)/dx
                        +((d[i,j+1,k]+d[i,j,k])/2*(ϕ1[i,j+1,k]-ϕ1[i,j,k])/dy
                            -(d[i,j,k]+d[i,j-1,k])/2*(ϕ1[i,j,k]-ϕ1[i,j-1,k])/dy)/dy
                        +((d[i,j,k+1]+d[i,j,k])/2*(ϕ1[i,j,k+1]-ϕ1[i,j,k])/dz
                            -(d[i,j,k]+d[i,j,k-1])/2*(ϕ1[i,j,k]-ϕ1[i,j,k-1])/dz)/dz)
                    # Update temperature
                    ϕ2[i,j,k] = (diff+source)*dt+ϕ1[i,j,k]
                end
            end
         end
    end
end

function updateϕ_boundaries!(ϕ,ii,jj,kk,dx,dy,dz)
    ϕ[1,2:jj-1,2:kk-1] .= ϕ[2,2:jj-1,2:kk-1]
    ϕ[ii,2:jj-1,2:kk-1] .= ϕ[ii-1,2:jj-1,2:kk-1]
    ϕ[2:ii-1,1,2:kk-1] .= ϕ[2:ii-1,2,2:kk-1]
    ϕ[2:ii-1,jj,2:kk-1] .= ϕ[2:ii-1,jj-1,2:kk-1]
    ϕ[2:ii-1,2:jj-1,1] .= ϕ[2:ii-1,2:jj-1,2]
    ϕ[2:ii-1,2:jj-1,kk] .= ϕ0(kk*dz)
end

# Run simulation
"""
    total_energy(position_array, time)
    Return the total amount of heat energy produced by the array of wells 
    at positions `position_array` for a specified period of time `time`
"""
function run_array_simulation(position_array::AbstractArray{Float64, 2}, time_elapsed::Int)
    # position_array = [[x1, x2, ...], [y1, y2, ...]]
    # time = time in seconds
    px = position_array[1]
    py = position_array[2]
    init(px, py)
    for t = 0:2:tt
        # Update ϕ
        updateϕ_domain!(ϕ2,ϕ1,d,dx,dy,dz,dt)
        updateϕ_boundaries!(ϕ2,ii,jj,kk,dx,dy,dz)
        
        # Update ϕ
        updateϕ_domain!(ϕ1,ϕ2,d,dx,dy,dz,dt)
        updateϕ_boundaries!(ϕ1,ii,jj,kk,dx,dy,dz)
        
        # Save ϕ
        if t % 10 == 0
            println("Iteration:$t, time:$(round(t*dt,digits=2))s, bottom temp:$(round(ϕ2[ii÷2,jj÷2,kk-1],digits=4))°C")
        end
    end
end


# Well Positions Array i.e each position is (px[i], py[i])
px = [25, 50, 75]
py = [25, 50, 75]

run_array_simulation([px, py], sim_time)



## TODO ##
# - Modify the code to model interference in the well array and compute the total heat energy output
# - 