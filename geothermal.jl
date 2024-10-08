using LinearAlgebra
using .Threads
using WriteVTK

################################################################################
# Diffusion convection equation for temperature in a pipe-in-pipe geometry
#
# Equation: ρ c ∂ϕ/∂t =  ∂(λ dϕ/∂x)/∂x + ∂(λ ∂ϕ/∂y)/∂y + ∂(λ ∂ϕ/∂z)/∂z
#                        -(ε vx ∂ϕ/∂x + ε vy ∂ϕ/∂y + ε vz ∂ϕ/∂z)
#                        + S
# Initial conditions:
#
# Boundary conditions:
#

# Create experiment folder #####################################################
path = "results/"
rm(path, recursive=true, force=true)
mkpath(path)

# Auxiliary functions  ##########################################################
function save(path,name,var,rx,ry,rz,t)
    vtk_grid("$path/$name-$t", rx, ry, rz; compress = false, append = false, ascii = false) do vtk
        vtk[name] = var
    end
end

# Physical parameters ##########################################################

# Maximum simulation time [s]
sim_time = 120

# Earth surface temperature [°C]
ϕs = 20

# Rock: granite #########################################
# Rock density [g/m3]
ρr = 2750000
# Rock specific heat [J/(g °C)]
cr = 0.790
# Rock thermal conductivity [W/(m °C)]
λr =  2.62
# Rock diffusion coefficient
dr = λr/(ρr*cr)
# Rock temperature as a function of depth [°C]
ϕ0(d) = ϕs+1.0*d

# Pipes: polyethylene ####################################
# Inner pipe inside radius [m]
r1 = 0.1
# Inner pipe thickness [m]
t1 = 0.01
# Inner pipe height [m]
h1 = 8.5
# Outer pipe inside radius [m]
vol1 = π*r1^2*h1
r2 = sqrt((vol1 + π*(r1+t1)^2*h1)/(h1*π)) # <= vol1 = vol2 = π*r2^2*h1-π*(r1+t1)^2*h1
# Outer pipe thickness [m]
t2 = 0.01
# Outer pipe height [m]
h2 = 9
# Porosity: ratio of liquid volume to the total volume
ε = 1
# Pipe density [g/m3]
ρp = 961000 
# Pipe specific heat [J/(g °C)]
cp = 2.9
# Pipe thermal conductivity [W/(m °C)]
λp = 0.54 
# Pipe diffusion coefficient
dp = λp/(ρp*cp)

# Fluid: water #########################################
# Fluid density [g/m3]
ρf = 997000
# Fluid specific heat capacity [J/(g °C)]
cf = 4.184 
# Fluid thermal conductivity [W/(m °C)]
λf = 0.6
# Fluid diffusion coefficient
df = λf/(ρf*cf)
# Flow speed [m/s]
uf = 0.01
vx0 = uf
vy0 = 0
vz0 = 0
# Characteristic linear dimension (diameter of the pipe) [m]
Lf = 2r1
# Fluid dynamic viscosity at 25 °C [Pa⋅s]
μf = 0.00089 # 0.0005465 at 50 °C
# Reynolds number
Re = ρf*uf*Lf/μf

# See https://gchem.cm.utexas.edu/data/section2.php?target=heat-capacities.php
#     https://en.wikipedia.org/wiki/High-density_polyethylene
#     https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation


# Numerical parameters #########################################################

# Geometry distances [m]
xx = 1
yy = 1
zz = h2 + 1

# dx, dy, dz [m]
dx = 0.0166
dy = 0.0166
dz = 0.2
rx = 0:dx:xx
ry = 0:dy:yy
rz = 0:dz:zz

# dt using stability condition
dtd = (1/(2*maximum([dr,dp,df]))*(1/dx^2+1/dy^2+1/dy^2)^-1)
dtc= minimum([dx/norm(vx0), dy/norm(vy0), dz/norm(vz0)])
dt = minimum([dtd,dtc]) # 0.00005
println("dt:$dt")

# No. of time iterations
tt = round(Int,sim_time/dt) 
println("tt:$tt")

# No. of spatial domain nodes
ii = round(Int,xx/dx)
jj = round(Int,yy/dy)
kk = round(Int,zz/dz)
nn = ii * jj * kk
println("ii:$ii, jj:$jj, kk:$kk. nn:$nn.")

# No. of height nodes
hh1 = round(Int,h1/dz)
hh2 = round(Int,h2/dz)

# Define temperature matrices: ϕ2, ϕ1
ϕ2 = ones(ii,jj,kk)
ϕ1 = ones(ii,jj,kk)

# Initial and boundary conditions #############################################
d = zeros(ii,jj,kk)
vx = zeros(ii,jj,kk)
vy = zeros(ii,jj,kk)
vz = zeros(ii,jj,kk)
xc, yc = ii÷2*dx, jj÷2*dy
for k in 1:kk
    zk = k*dz
    for j in 1:jj
        yj = j*dy
        for i in 1:ii
            xi = i*dx
            r = norm([xi,yj]-[xc,yc])
            if r < r1  # inside inner pipe
                d[i,j,k] = df
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = -uf
                ϕ2[i,j,k] = ϕs
            elseif r < r1+t1  # inner pipe
                d[i,j,k] = dp
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = 0
                ϕ2[i,j,k] = ϕs
            elseif r < r2  # inside outer pipe
                d[i,j,k] = df
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = uf
                ϕ2[i,j,k] = ϕs
            elseif r < r2+t2  # outer pipe
                d[i,j,k] = dp
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = 0
                ϕ2[i,j,k] = ϕs
            else # rock
                d[i,j,k] = dr
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = 0
                ϕ2[i,j,k] = ϕ0(zk)
            end
        end
    end
end
ϕ1 .= ϕ2
vels = Array{Float64}(undef, 3, ii, jj, kk)
vels[1,:,:,:] = vx
vels[2,:,:,:] = vy
vels[3,:,:,:] = vz
save(path,"velocity",vels,rx,ry,rz,0)
save(path,"diff_coeff",d,rx,ry,rz,0)

# Simulation ##################################################################

# Solve eq. system
function updateϕ_domain!(ϕ2,ϕ1,d,vx,vy,vz,dx,dy,dz,dt)
    @threads for k = 2:kk-1
        for j = 2:jj-1
            for i = 2:ii-1
                diff = (((d[i+1,j,k]+d[i,j,k])/2*(ϕ1[i+1,j,k]-ϕ1[i,j,k])/dx 
                        -(d[i,j,k]+d[i-1,j,k])/2*(ϕ1[i,j,k]-ϕ1[i-1,j,k])/dx)/dx
                       +((d[i,j+1,k]+d[i,j,k])/2*(ϕ1[i,j+1,k]-ϕ1[i,j,k])/dy
                        -(d[i,j,k]+d[i,j-1,k])/2*(ϕ1[i,j,k]-ϕ1[i,j-1,k])/dy)/dy
                       +((d[i,j,k+1]+d[i,j,k])/2*(ϕ1[i,j,k+1]-ϕ1[i,j,k])/dz
                        -(d[i,j,k]+d[i,j,k-1])/2*(ϕ1[i,j,k]-ϕ1[i,j,k-1])/dz)/dz)
                conv = ( ε*vx[i,j,k]*(ϕ1[i+1,j,k]-ϕ1[i-1,j,k])/2dx
                        +ε*vy[i,j,k]*(ϕ1[i,j+1,k]-ϕ1[i,j-1,k])/2dy
                        +ε*vz[i,j,k]*(ϕ1[i,j,k+1]-ϕ1[i,j,k-1])/2dz) # TODO: use upwind
                source = 0.0
                ϕ2[i,j,k] = (diff-conv+source)*dt+ϕ1[i,j,k]
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
for t = 0:2:tt
    # Update ϕ
    updateϕ_domain!(ϕ2,ϕ1,d,vx,vy,vz,dx,dy,dz,dt)
    updateϕ_boundaries!(ϕ2,ii,jj,kk,dx,dy,dz)
    
    # Update ϕ
    updateϕ_domain!(ϕ1,ϕ2,d,vx,vy,vz,dx,dy,dz,dt)
    updateϕ_boundaries!(ϕ1,ii,jj,kk,dx,dy,dz)
    
    # Save ϕ
    if t % 10 == 0
        println("Iteration:$t, time:$(round(t*dt,digits=2))s, bottom temp:$(round(ϕ2[ii÷2,jj÷2,kk-1],digits=4))°C")
        save(path,"temperature",ϕ2,rx,ry,rz,t)
    end
end

