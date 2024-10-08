using LinearAlgebra
using .Threads
using WriteVTK

################################################################################
# Diffusion convection equation for temperature ϕ
#
# Equation: dϕ/dt =   - ( d(D*dϕ/dx)/dx + d(D*dϕ/dy)/dy + d(D*dϕ/dz)/dz )
#                     + vx*dϕ/dx + vy*dϕ/dy + vz*dϕ/dz
#                     + S
#
# Boundary conditions
# 
# Initial conditions
#

# Create experiment folder #####################################################
path = "results/"
rm(path, recursive=true, force=true)
mkpath(path)

# Physical parameters ##########################################################

# Maximum simulation time [s]
sim_time = 0.1
# Initial temperature as a function of depth [C]
ϕ0(k) = 1+0.01*k 
# Geometry distances [m]
xx = 1
yy = 1
zz = 10
# Inner pipe inside radius [m]
r1 = 0.1
# Inner pipe thickness [m]
t1 = 0.01
# Inner pipe height [m]
h1 = 0.85 * zz
# Outer pipe inside radius [m]
r2 = 0.1
# Outer pipe thickness [m]
t2 = 0.01
# Outer pipe height [m]
h2 = 0.90 * zz
# Diffusion coefficients
dw = 0.5 # water
dp = 1   # pipe
dr = 0.001 # rock

# Numerical parameters #########################################################

# dt, dx, dy, dz [m]
dt = 0.00005
dx = 0.02
dy = 0.02
dz = 0.02
rx = 0:dx:xx
ry = 0:dy:yy
rz = 0:dz:zz

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

# Define diffusion coefficient and velocities
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
                d[i,j,k] = dw
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = -1
                ϕ2[i,j,k] = 1
            elseif r < r1+t1  # inner pipe
                d[i,j,k] = dp
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = 0
                ϕ2[i,j,k] = 1
            elseif r < r1+t1+r2  # inside outer pipe
                d[i,j,k] = dw
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = 1
                ϕ2[i,j,k] = 1
            elseif r < r1+t1+r2+t2  # outer pipe
                d[i,j,k] = dp
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = 0
                ϕ2[i,j,k] = 1
            else # rock
                d[i,j,k] = dr
                vx[i,j,k] = 0
                vy[i,j,k] = 0
                vz[i,j,k] = 0
                ϕ2[i,j,k] = ϕ0(k)
            end
        end
    end
end
#for k in hh1+1:hh2
#    zk = k*dz
#    for j in 1:jj
#        yj = j*dy
#        for i in 1:ii
#            xi = i*dx
#            r = norm([xi,yj]-[xc,yc])
#            if r < r1+t1+r2  # inside outer pipe
#                d[i,j,k] = dw
#                vx[i,j,k] = 0
#                vy[i,j,k] = 0
#                vz[i,j,k] = 1
#            elseif r < r1+t1+r2+t2  # outer pipe
#                d[i,j,k] = dp
#                vx[i,j,k] = 0
#                vy[i,j,k] = 0
#                vz[i,j,k] = 0
#            else # rock
#                d[i,j,k] = dr
#                vx[i,j,k] = 0
#                vy[i,j,k] = 0
#                vz[i,j,k] = 0
#            end
#        end
#    end
#end

# Set initial and boundary conditions
#for k in 1:kk
#    ϕ2[:,:,k] .= ϕ0(k)
#end
ϕ1 .= ϕ2

# Solve eq. system
function updateϕ_domain!(ϕ2, ϕ1, d, vx, vy, vz, dx, dy, dz, dt)
    @threads for k = 2:kk-1
        for j = 2:jj-1
            for i = 2:ii-1
                diff = (((d[i+1,j,k]+d[i,j,k])/2*(ϕ1[i+1,j,k]-ϕ1[i,j,k])/dx 
                        -(d[i,j,k]+d[i-1,j,k])/2*(ϕ1[i,j,k]-ϕ1[i-1,j,k])/dx)/dx
                       +((d[i,j+1,k]+d[i,j,k])/2*(ϕ1[i,j+1,k]-ϕ1[i,j,k])/dy
                        -(d[i,j,k]+d[i,j-1,k])/2*(ϕ1[i,j,k]-ϕ1[i,j-1,k])/dy)/dy
                       +((d[i,j,k+1]+d[i,j,k])/2*(ϕ1[i,j,k+1]-ϕ1[i,j,k])/dz
                        -(d[i,j,k]+d[i,j,k-1])/2*(ϕ1[i,j,k]-ϕ1[i,j,k-1])/dz)/dz)
                conv = ( vx[i,j,k]*(ϕ1[i+1,j,k]-ϕ1[i-1,j,k])/2dx
                        +vy[i,j,k]*(ϕ1[i,j+1,k]-ϕ1[i,j-1,k])/2dy
                        +vz[i,j,k]*(ϕ1[i,j,k+1]-ϕ1[i,j,k-1])/2dz) # TODO: use upwind
                conv = 0
                source = 0.0
                ϕ2[i,j,k] = (diff-conv+source)*dt+ϕ1[i,j,k]
            end
         end
    end
end

function updateϕ_boundaries!(ϕ, ii, jj, kk)
    ϕ[1,2:jj-1,2:kk-1] .= ϕ[2,2:jj-1,2:kk-1]
    ϕ[ii,2:jj-1,2:kk-1] .= ϕ[ii-1,2:jj-1,2:kk-1]
    ϕ[2:ii-1,1,2:kk-1] .= ϕ[2:ii-1,2,2:kk-1]
    ϕ[2:ii-1,jj,2:kk-1] .= ϕ[2:ii-1,jj-1,2:kk-1]
    ϕ[2:ii-1,2:jj-1,1] .= ϕ[2:ii-1,2:jj-1,2]
    ϕ[2:ii-1,2:jj-1,kk] .= ϕ0(kk)
end

function saveϕ(path, ϕ, rx, ry, rz, t)
    vtk_grid("$path/ϕ_$t", rx, ry, rz; compress = false, append = false, ascii = false) do vtk
        vtk["temperature"] = ϕ
    end
end

for t = 0:2:tt
    # Update ϕ
    updateϕ_domain!(ϕ2, ϕ1, d, vx, vy, vz, dx, dy, dz, dt)
    updateϕ_boundaries!(ϕ2, ii, jj, kk)
    
    # Update ϕ
    updateϕ_domain!(ϕ1, ϕ2, d, vx, vy, vz, dx, dy, dz, dt)
    updateϕ_boundaries!(ϕ1, ii, jj, kk)
    
    # Save ϕ
    if t % 10 == 0
        saveϕ(path, ϕ2, rx, ry, rz, t)
    end
end


