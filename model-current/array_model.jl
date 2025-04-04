using LinearAlgebra
using .Threads
using WriteVTK
using CSV
using DataFrames
using Plots
include("utils.jl")

# Create experiment folder #####################################################
path = "dbhe-array-results/"
rm(path, recursive=true, force=true)
mkpath(path)

# Deep borehole heat exchanger (DBHE) model ####################################
#
# DBHE type: coaxial
#
# Equations for each DBHE:
# 
#     1D element representation of the center fluid temperature, ϕc.
#         Cc*dϕc/dt=(ϕa-ϕc)/Rac-m*Cf*dϕc/dz
#
#     1D element representation of the annulus fluid temperature, ϕa.
#         Ca*dϕa/dt=(ϕc-ϕa)/Rac+(ϕbw-ϕa)/Rb-m*Cf*dϕa/dz
#
# Coupling with 3D model representing the ground
#
#     Heat flux at the DBHE wall, q.
#         q = ϕa*ϕbw/(Rb+Rs)
#
#     Temperature at the DBHE wall, ϕbw. 
#         ϕbw = computed by the 3D model, using q as input.
#
# References:
#    10.1016/j.renene.2024.121963
#    10.1016/j.renene.2021.07.086
#    10.1016/j.energy.2019.05.228
#
################################################################################
# DBHE array model #############################################################
# 
# Equation for ground model
#     ρ c ∂ϕ/∂t =  ∂(k ∂ϕ/∂x)/∂x + ∂(k ∂ϕ/∂y)/∂y + ∂(k ∂ϕ/∂z)/∂z
#
################################################################################


# Geological parameters (10.1016/j.renene.2021.07.086) #########################

# Layer 1: Clay
# Length (m)
dl1 = 636
# Thermal conductivity (W/m.K)
kl1 = 1.8
# Density (kg/m3)
ρl1 = 1780
# Speciﬁc heat capacity (J/kg.K)
Cl1 = 1379
# Diffusion coefficient
D1 = kl1/(ρl1*Cl1)

# Layer 2: Mudstone
# Length (m)
dl2 = 562
# Thermal conductivity (W/m.K)
kl2 = 2.6
# Density (kg/m3)
ρl2 = 2030
# Speciﬁc heat capacity (J/kg.K)
Cl2 = 1450
# Diffusion coefficient
D2 = kl2/(ρl2*Cl2)

# Layer 3: Medium sand
# Length (m)
dl3 = 562
# Thermal conductivity (W/m.K)
kl3 = 3.5
# Density (kg/m3)
ρl3 = 1510
# Speciﬁc heat capacity (J/kg.K)
Cl3 = 1300
# Diffusion coefficient
D3 = kl3/(ρl3*Cl3)

# Layer 4: Stand stone
# Length (m)
dl4 = 1090
# Thermal conductivity (W/m.K)
kl4 = 5.3
# Density (kg/m3)
ρl4 = 2600
# Speciﬁc heat capacity (J/kg.K)
Cl4 = 878
# Diffusion coefficient
D4 = kl4/(ρl4*Cl4)

# Diffusion coefficient
function diff_coeff(z)
    if z < dl1 
        return D1
    elseif z < dl1+dl2
        return D2
    elseif z < dl1+dl2+dl3
        return D3
    else 
        return D4
    end
end

# Earth surface temperature (K)
ϕs = 283.15
# Geothermal gradient (K/km). 10.1016/j.energy.2019.05.228
gg = 3/100
# Ground temperature as a function of depth (K). 10.1016/j.renene.2021.07.086
ϕ0(z) = ϕs+gg*z

# DBHE parameters ##############################################################

# Borehole depth (m). 10.1016/j.renene.2021.07.086.
depth = 2000
# Inner center pipe speciﬁcations x thickness (m): 0.125 x 0.0114. 10.1016/j.renene.2021.07.086.
dco = 0.125
dci = dco-2*0.0114
# Outer annulus pipe speciﬁcations x thickness (m): 0.1937 x 0.00833. 10.1016/j.renene.2021.07.086.
dao = 0.1937
dai = dao-2*0.00833
# Thermal conductivity of center pipe (W/m.K). 10.1016/j.renene.2021.07.086
kc = 0.4
# Thermal conductivity of annulus pipe (W/m.K).  10.1016/j.renene.2021.07.086.
ka = 41
# Thermal conductivity of water (W/m.K)
kf = 0.618
# Thermal conductivity of grout (W/m.K). 10.1016/j.renene.2021.07.086.
kg = 1.5
# Speciﬁc heat capacity of water (J/kg.K). 10.1016/j.renene.2021.07.086.
Cf = 4174
# Pipe specific heat capacity (J/(kg⋅K))
Cp = 1735 # TODO:missing value
# Grout specific heat capacity (J/(kg⋅K)). 10.1016/j.renene.2024.121963
Cg = 1735 
# Ground surface temperature (K). 10.1016/j.renene.2021.07.086.
ϕs = 283.15
# Water density (kg/m³). 10.1016/j.renene.2024.121963.
ρf = 998
# Pipe density (kg/m³)
ρp = 2190 # TODO:missing value
# Grout density (kg/m³). 10.1016/j.renene.2024.121963
ρg = 2190
# Mass flow rate (kg/s). 10.1016/j.renene.2024.121963
mfr = 4.88
# Thermal resistance between center and annulus pipe ((K m)/W). 10.1016/j.energy.2019.05.228.
Rac = 0.08
# Thermal resistance between annulus and borehole wall ((K m)/W). 10.1016/j.energy.2019.05.228
Rb = 0.025
# Thermal resistance between the borehole and the surroudning soil ((K m)/W). 10.1016/j.energy.2019.05.228
Rs = 0.01
# Thermal capacity of circulating ﬂuid in the annulus, Ca. 10.1016/j.renene.2024.121963.
Ca = (π/4)*(dai^2-dco^2)*ρf*Cf
    +(π/4)*(dao^2-dai^2)*ρp*Cp
    #+(π/4)*(db^2-dao^2)*ρg*Cg
# Thermal capacity of circulating ﬂuid in the center, Cc. 10.1016/j.renene.2024.121963.
Cc = (π/4)*(dci^2*ρf*Cf)
    +(π/4)*(dco^2-dci^2)*ρp*Cp
# Heat extraction rate Q (W)
Q = 100_000
# DBHE positions
xs = [5, 15, 25]
ys = [5, 15, 25]
# Number of DBHEs
mm = length(xs)
# Get DBHE index
function dbhe_index(x,y)
    for (m,(xb,yb)) in enumerate(zip(xs,ys))
        r = norm([x,y]-[xb,yb]) # Distance to DBHE center
        if r<dao/2
            return m
        end
    end
end

# Numerical parameters #########################################################

# Maximum simulation time (s)
#sim_time = 2400 # 40 minutes
sim_time = 3_960_000 # 1100 hours

# Geometry distances (m)
xx = 30
yy = 30
zz = dl1+dl2+dl3+dl4+10

# dx, dy, dz (m)
dx = dao
dy = dao
dz = 20
rx = 0:dx:xx
ry = 0:dy:yy
rz = 0:dz:zz

# dt using stability condition (check this)
dtd = (1/(2*maximum([D1,D2,D3,D4]))*(1/dx^2+1/dy^2+1/dy^2)^-1)
#dtc = minimum([dx/norm(vx0), dy/norm(vy0), dz/norm(vz0)]) # no convection
#dt = minimum([dtd,dtc])
dt = dtd/10
println("dt:$dt")

# No. of time iterations
tt = round(Int,sim_time/dt) 
println("tt:$tt")

# No. of spatial domain nodes
ii = round(Int,xx/dx)
jj = round(Int,yy/dy)
kk = round(Int,zz/dz)
nn = ii*jj*kk
println("ii:$ii, jj:$jj, kk:$kk. nn:$nn.")

# No. of depth nodes
dd1 = round(Int,depth/dz)
dd2 = round(Int,depth/dz)

# Initial and boundary conditions ##############################################

# Temperature at the center of the DBHEs. 1D model.
ϕc2 = ones(mm,kk)
ϕc1 = ones(mm,kk)
# Temperature at the annulus of the DBHEs. 1D model.
ϕa2 = ones(mm,kk)
ϕa1 = ones(mm,kk)
# Temperature at the DBHE wall. 1D model.
ϕbw = ones(mm,kk)
# Heat flux at the DBHE wall. 1D model.
qbw = ones(mm,kk)
# Ground temperature. 3D model.
ϕ2 = ones(ii,jj,kk)
ϕ1 = ones(ii,jj,kk)
# Diffusion coefficient
D = zeros(ii,jj,kk)
for k in 1:kk
    zk = k*dz
    for j in 1:jj
        yj = j*dy
        for i in 1:ii
            xi = i*dx
            # Ground
            D[i,j,k] = diff_coeff(zk)
            ϕ2[i,j,k] = ϕ0(zk)
            # DBHE
            for (xb, yb) in zip(xs, ys)
                r = norm([xi,yj]-[xb,yb]) # Distance to DBHE center
                if r < dao/2
                    D[i,j,k] = 0
                    ϕ2[i,j,k] = ϕs
                end
            end
        end
    end
end
ϕ1 .= ϕ2
save(path,"diff_coeff",D,rx,ry,rz,0)


# Update functions #############################################################

# Update temperature at the center and annulus of each DBHE, ϕc and ϕa. 1D model.
function update_ϕ_fluid!(ϕa2,ϕa1,ϕc2,ϕc1,ϕbw,mm,kk,dz,dt,Rac,Rb,mfr,Cf,Ca,Cc,Q)
    for m in 1:mm
        ϕc2[1] = ϕa1[1]-Q/mfr*Cf # Alternative: use values from Fig 7.
        for k in 2:kk-1
            # Fluid temperature in the center of the well
            diff = (ϕc1[m,k]-ϕa1[m,k])/Rac+(ϕbw[m,k]-ϕa1[m,k])/Rb
            conv = -mfr*Cf*(ϕa1[m,k+1]-ϕa1[m,k-1])/2dz
            ϕa2[k] = (diff+conv)*dt/Ca+ϕa1[m,k]
            # Fluid temperature in the annulus of the well
            diff = (ϕa1[m,k]-ϕc1[m,k])/Rac
            conv = -mfr*Cf*(ϕc1[m,k+1]-ϕc1[m,k-1])/2dz
            ϕc2[k] = (diff+conv)*dt/Cc+ϕc1[m,k]
        end
        ϕc2[kk] = ϕa2[kk]
    end
end

# Update heat flux at the wall of each DBHE, q. 1D model.
function update_q_dbhe_wall!(qbw,ϕa,ϕbw,mm,kk,dz,dt,Rb,Rs)
    for m in 1:mm
        for k in 2:kk-1
            qbw[m,k] = ϕa[m,k]*ϕbw[m,k]/(Rb+Rs)
        end
    end
end

# Update temperature at the borehole walls, ϕbw. 1D model.
# q = ϕa*ϕbw/(Rb+Rs)
# q = -k * grad(ϕ)
function update_ϕ_dbhe_wall!(ϕbw,ϕ,q,mm,kk,dz,dt)
    for m in 1:mm
        for k in 2:kk-1
            ϕbw[m,k] = -2*dx*q[m,k]/k+ϕ[k]
        end
    end
end

# Update temperature in ground domain. 3D model.
function update_ϕ_ground!(ϕ2,ϕ1,ϕbw,D,dx,dy,dz,dt,ii,jj,kk)
    @threads for k = 2:kk-1
        for j = 2:jj-1
            for i = 2:ii-1
                if D[i,j,k] > 0  # Ground domain
                    # Diffusive term
                    diff = (((D[i+1,j,k]+D[i,j,k])/2*(ϕ1[i+1,j,k]-ϕ1[i,j,k])/dx 
                            -(D[i,j,k]+D[i-1,j,k])/2*(ϕ1[i,j,k]-ϕ1[i-1,j,k])/dx)/dx
                           +((D[i,j+1,k]+D[i,j,k])/2*(ϕ1[i,j+1,k]-ϕ1[i,j,k])/dy
                            -(D[i,j,k]+D[i,j-1,k])/2*(ϕ1[i,j,k]-ϕ1[i,j-1,k])/dy)/dy
                           +((D[i,j,k+1]+D[i,j,k])/2*(ϕ1[i,j,k+1]-ϕ1[i,j,k])/dz
                            -(D[i,j,k]+D[i,j,k-1])/2*(ϕ1[i,j,k]-ϕ1[i,j,k-1])/dz)/dz)
                    # Source term
                    source = 0.0
                    # Update temperature
                    ϕ2[i,j,k] = (diff+source)*dt+ϕ1[i,j,k]
                else  # DBHE domain
                    m = dbhe_index(i*dx,j*dy)
                    ϕ2[i,j,k] = ϕbw[m,k]
                end
            end
         end
    end
end

# Update temperature in ground walls. 3D model.
function update_ϕ_ground_bound!(ϕ,ii,jj,kk,dx,dy,dz)
    ϕ[1,2:jj-1,2:kk-1] .= ϕ[2,2:jj-1,2:kk-1]
    ϕ[ii,2:jj-1,2:kk-1] .= ϕ[ii-1,2:jj-1,2:kk-1]
    ϕ[2:ii-1,1,2:kk-1] .= ϕ[2:ii-1,2,2:kk-1]
    ϕ[2:ii-1,jj,2:kk-1] .= ϕ[2:ii-1,jj-1,2:kk-1]
    ϕ[2:ii-1,2:jj-1,1] .= ϕ[2:ii-1,2:jj-1,2]
    ϕ[2:ii-1,2:jj-1,kk] .= ϕ0(kk*dz)
end


# Simulation ##################################################################

# Run simulation
for t = 0:2:tt
    # Update ϕ
    update_ϕ_fluid!(ϕa2,ϕa1,ϕc2,ϕc1,ϕbw,mm,kk,dz,dt,Rac,Rb,mfr,Cf,Ca,Cc,Q) # ϕa2,ϕa1,ϕc2,ϕc1
    update_q_dbhe_wall!(qbw,ϕa2,ϕbw,mm,kk,dz,dt,Rb,Rs) # qbw
    update_ϕ_dbhe_wall!(ϕbw,ϕ2,qbw,mm,kk,dz,dt) # ϕbw
    update_ϕ_ground!(ϕ2,ϕ1,ϕbw,D,dx,dy,dz,dt,ii,jj,kk) # ϕ2,ϕ1
    update_ϕ_ground_bound!(ϕ2,ii,jj,kk,dx,dy,dz) # ϕ2,ϕ1
    
    # Update ϕ
    update_ϕ_fluid!(ϕa1,ϕa2,ϕc1,ϕc2,ϕbw,mm,kk,dz,dt,Rac,Rb,mfr,Cf,Ca,Cc,Q) # ϕa1,ϕa2,ϕc1,ϕc2
    update_q_dbhe_wall!(qbw,ϕa1,ϕbw,mm,kk,dz,dt,Rb,Rs) # qbw
    update_ϕ_dbhe_wall!(ϕbw,ϕ1,qbw,mm,kk,dz,dt) # ϕbw
    update_ϕ_ground!(ϕ1,ϕ2,ϕbw,D,dx,dy,dz,dt,ii,jj,kk) # ϕ1,ϕ2
    update_ϕ_ground_bound!(ϕ1,ii,jj,kk,dx,dy,dz) # ϕ1,ϕ2
    
    # Save ϕ
    if t % 10 == 0
        println("Iteration:$t, time:$(round(t*dt/60/60,digits=2))hs, bottom temp:$(round(ϕ2[ii÷2,jj÷2,kk-1],digits=4)-273.15)°C")
        #save(path,"ground_temp",ϕ2,rx,ry,rz,t)
        #save(path,"dbhe_annulus_temp",ϕa2,rx,ry,rz,t)
        #save(path,"dbhe_center_temp",ϕc2,rx,ry,rz,t)
    end
end

# Validation ###################################################################

# Read the CSV file
df = CSV.read("temp-vs-time.csv", DataFrame)

# Comparison of inlet and outlet water temperature by measurement and simulation.
# 10.1016/j.renene.2021.07.086
t = df."time (h)"
ϕf = df."water_temp (°C)"
plot(t, ϕf)

