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
# Equations for each DBHE, b:
# 
#     1D element representation of the center fluid temperature, ϕb_c:
#         Cc*∂ϕb_c(t,z)/∂t = (ϕb_a(t,z)-ϕb_c(t,z))/Rac
#                            +m*Cf*∂ϕb_c(t,z)/∂z
#
#     1D element representation of the annulus fluid temperature, ϕb_a:
#         Ca*∂ϕb_a(t,z)/∂t = (ϕb_c(t,z)-ϕb_a(t,z))/Rac
#                           +(ϕb_w(t,z)-ϕb_a(t,z))/Rb
#                           -m*Cf*∂ϕb_a(t,z)/∂z
#
#     Heat flux at the DBHE wall, qb_w:
#         qb_w(t,z) = (ϕb_a(t,z)-ϕb_w(t,z))/(Rb+Rs)
#
#     Boundary and initial conditions:
#         ϕb_c(t,z=0) = ϕb_a(t,z=0)-Q/(m*Cf)
#         ϕb_c(t,z=zb) = ϕb_a(t,z=zb)
#         ϕb_c(t=0,z) = ϕ0(z)
#         ϕb_a(t=0,z) = ϕ0(z)
#         ϕ0(z) = ϕs+gg*z
#
#         z range for borehole b: 1:zb
#
# References:
#    10.1016/j.renene.2024.121963
#    10.1016/j.renene.2021.07.086
#    10.1016/j.energy.2019.05.228
#    10.1016/j.renene.2021.01.036
#    10.1016/j.enbuild.2018.02.013
#
################################################################################
# DBHE array model #############################################################
# 
# Equation for ground model:
#     ρ(x,y,z) c(x,y,z) ∂ϕ(t,x,y,z)/∂t =   ∂(k(x,y,z) * ∂ϕ(t,x,y,z)/∂x)/∂x
#                                        + ∂(k(x,y,z) * ∂ϕ(t,x,y,z)/∂y)/∂y
#                                        + ∂(k(x,y,z) * ∂ϕ(t,x,y,z)/∂z)/∂z
#
# Boundary conditions:
#   Ground upper surface: ks*∂ϕ(t,x,y,z)/∂z = hs*(ϕ-ϕs)
#   Ground lower surface: ϕ(t,x,y,z=zz) = ϕ0(z=zz)
#   Ground lateral sides: ∇ϕ(t,x,y,z),n = 0
#   DBHE walls (1D elements): ϕ(t,x=xb,y=yb,z=1:zb) = ϕb_w(t,z=1:zb)
#
# Coupling of DBHE model (1D) with array model (3D):
#     Temperature at the DBHE wall, ϕb_w, is computed using
#     the heat flux at the DBHE wall, qb_w.
#
# Initial conditions:
#  ϕ(t=0,x,y,z) = ϕ0(z), (x,y,z) ∉ borehole positions.
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

# Ground surface temperature (K). 10.1016/j.renene.2021.07.086.
ϕs = 283.15
# Convective heat transfer coeﬃcient on ground surface (W/(m^22 K)). 10.1016/j.enbuild.2018.02.013
hs = 15
# Thermal conductivity of subsurface (W/(m K)). 10.1016/j.enbuild.2018.02.013.
ks = 2.5
# Geothermal gradient (K/m). 10.1016/j.energy.2019.05.228.
gg = 2/100
# Ground temperature as a function of depth (K). 10.1016/j.renene.2021.07.086.
ϕ0(z) = ϕs+gg*z

# DBHE parameters ##############################################################

# Borehole depth (m). 10.1016/j.renene.2021.07.086.
dbhe_depth = 2000
# Borehole diameter (m). 10.1016/j.enbuild.2018.02.013
db = 0.28
# Inner center pipe diameters (m). # 10.1016/j.enbuild.2018.02.013
dco = 0.14 
dci = 0.124
# Outer annulus pipe  diameters (m). 10.1016/j.enbuild.2018.02.013
dao = 0.2 
dai = 0.188
# Thermal conductivity of center pipe (W/m.K). 10.1016/j.renene.2021.07.086.
kc = 0.4
# Thermal conductivity of annulus pipe (W/m.K).  10.1016/j.renene.2021.07.086.
ka = 41
# Thermal conductivity of water (W/m.K).
kf = 0.618
# Thermal conductivity of grout (W/m.K). 10.1016/j.renene.2021.07.086.
kg = 1.5
# Speciﬁc heat capacity of water (J/kg.K). 10.1016/j.renene.2021.07.086.
Cf = 4174
# Pipe specific heat capacity (J/(kg⋅K)). 10.1016/j.enbuild.2018.12.006.
Cp = 2100 
# Grout specific heat capacity (J/(kg⋅K)). 10.1016/j.renene.2024.121963.
Cg = 1735 
# Water density (kg/m³). 10.1016/j.renene.2024.121963.
ρf = 998
# Pipe density (kg/m³). 10.1016/j.enbuild.2018.12.006.
ρp = 930
# Grout density (kg/m³). 10.1016/j.renene.2024.121963.
ρg = 2190
# Mass flow rate (kg/s). 10.1016/j.enbuild.2018.02.013
mfr = 11.65
# Thermal resistance between center and annulus pipe ((K m)/W). 10.1016/j.energy.2019.05.228.
Rac = 0.08
# Thermal resistance between annulus and borehole wall ((K m)/W). 10.1016/j.energy.2019.05.228.
Rb = 0.025
# Thermal resistance between the borehole and the surroudning soil ((K m)/W). 10.1016/j.energy.2019.05.228.
Rs = 0.01
# Thermal capacity of circulating ﬂuid in the annulus, Ca. 10.1016/j.renene.2024.121963.
Ca = (π/4)*(dai^2-dco^2)*ρf*Cf
    +(π/4)*(dao^2-dai^2)*ρp*Cp
    +(π/4)*(db^2-dao^2)*ρg*Cg
# Thermal capacity of circulating ﬂuid in the center, Cc. 10.1016/j.renene.2024.121963.
Cc = (π/4)*(dci^2*ρf*Cf)
    +(π/4)*(dco^2-dci^2)*ρp*Cp
# Heat extraction rate Q (W)
Q = 200_000
# DBHE positions
xs = [5]
ys = [5]
# Number of DBHEs
bb = length(xs)
# Get DBHE index
function dbhe_index(x,y)
    for (b,(xb,yb)) in enumerate(zip(xs,ys))
        r = norm([x,y]-[xb,yb]) # Distance to DBHE center
        if r<dao/2
            return b
        end
    end
end
function dbhe_indexes(b)
    return 5, 5 # TODO:
end

#A1 = π * ((dai/2)^2 - (dco/2)^2) # cross sectional area of annular space
#A2 = π * (dci/2)^2 # cross sectional area of center space

# Numerical parameters #########################################################

# Maximum simulation time (s)
sim_time = 3_960_000 # 1100 hours

# Geometry distances (m)
xx = 10
yy = 10
zz = dl1+dl2+dl3+dl4+30
zzb = dbhe_depth

# dx, dy, dz (m)
dx = db
dy = db
dz = 40
rx = 0:dx:xx
ry = 0:dy:yy
rz = 0:dz:zz
rzb = 0:dz:zzb

# dt using stability condition (check this)
dtd = (1/(2*maximum([D1,D2,D3,D4]))*(1/dx^2+1/dy^2+1/dz^2)^-1)
#dtc = minimum([dx/norm(vx0), dy/norm(vy0), dz/norm(vz0)]) # no convection
#dt = minimum([dtd,dtc])
dt = 10
println("dt:$dt")

# No. of time iterations
tt = round(Int,sim_time/dt)
println("tt:$tt")

# Save step
st = 10_000

# No. of spatial domain nodes
ii = round(Int,xx/dx)
jj = round(Int,yy/dy)
kk = round(Int,zz/dz)
kkb = round(Int,zzb/dz+1)
nn = ii*jj*kk
println("ii:$ii, jj:$jj, kk:$kk. nn:$nn.")


# Update functions #############################################################

# Update temperature at the center and annulus of each DBHE, ϕc and ϕa. 1D model.
function update_ϕ_fluid!(ϕa2,ϕa1,ϕc2,ϕc1,ϕbw,bb,kkb,dz,dt,nt,Rac,Rb,mfr0,Cf,Ca,Cc,Q0)
    mfr = mfr0 # get_mfr(dt*nt)
    Q = Q0 # get_Q(dt*nt)
    for b in 1:bb
        #ϕa2[b,1] = get_ϕin(dt*nt) # 10.1016/j.renene.2021.07.086.
        ϕa2[b,1] = ϕc1[b,1]-Q/(mfr*Cf) # 10.1016/j.renene.2021.07.086.
        for k in 2:kkb-1
            # Fluid temperature in the annulus of the well
            diff = (ϕc1[b,k]-ϕa1[b,k])/Rac+
                   (ϕbw[b,k]-ϕa1[b,k])/Rb
                   #A1*kf*(ϕa1[b,k+1]-2*ϕa1[b,k]+ϕa1[b,k-1])/dz^2
            conv = -mfr*Cf*(ϕa1[b,k]-ϕa1[b,k-1])/dz
            ϕa2[b,k] = (diff+conv)*dt/Ca+ϕa1[b,k]
            # Fluid temperature in the center of the well
            diff = (ϕa1[b,k]-ϕc1[b,k])/Rac
                    #A2*kf*(ϕc1[b,k+1]-2*ϕc1[b,k]+ϕc1[b,k-1])/dz^2
            conv = +mfr*Cf*(ϕc1[b,k+1]-ϕc1[b,k])/dz
            ϕc2[b,k] = (diff+conv)*dt/Cc+ϕc1[b,k]
        end
        ϕa2[b,kkb] = ϕa2[b,kkb-1]
        ϕc2[b,kkb] = ϕa2[b,kkb]
        ϕc2[b,1] = ϕc2[b,2]
    end
end

# Update heat flux at the wall of each DBHE, qbw. 1D model.
function update_q_dbhe_wall!(qbw,ϕa,ϕbw,bb,kkb,dz,dt,Rb,Rs)
    for b in 1:bb
        for k in 1:kkb
            # TODO: check
            if dz*k < 30 # Depth of insulated section of borehole. 10.1016/j.enbuild.2018.02.013
                qbw[b,k] = 0
            else
                qbw[b,k] = (ϕa[b,k]-ϕbw[b,k])/(Rb+Rs)
            end
        end
    end
end

# Update temperature at the borehole walls, ϕbw. 1D model.
function update_ϕ_dbhe_wall!(ϕbw,ϕ,qbw,ka,bb,kkb,dx,dy,dz,dt)
    l = 2*π*dao #2(dx+dy)
    for b in 1:bb
        i,j = dbhe_indexes(b)
        for k in 1:kkb
            # TODO: check
            #q1 = -ka*(ϕ[i+1,j,k]-ϕbw[b,k])/dx
            #q2 = ka*(ϕbw[b,k]-ϕ[i-1,j,k])/dx
            #q3 = -ka*(ϕ[i,j+1,k]-ϕbw[b,k])/dy
            #q4 = ka*(ϕbw[b,k]-ϕ[i,j-1,k])/dy
            #l*qbw[b,k] = q1*dx+q2*dx+q3*dy+q4*dy => ϕbw[b,k]
            ϕbw[b,k] = (ka*dy/dx*(ϕ[i+1,j,k]+ϕ[i-1,j,k])+
                        ka*dx/dy*(ϕ[i,j+1,k]+ϕ[i,j-1,k])+
                        l*qbw[b,k])/(2ka*dy/dx+2ka*dx/dy)
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
                    b = dbhe_index(i*dx,j*dy)
                    ϕ2[i,j,k] = ϕbw[b,k]
                end
            end
         end
    end
end

# Update temperature in ground walls. 3D model.
function update_ϕ_ground_bound!(ϕ,ii,jj,kk,dx,dy,dz,hs,ks,ϕs)
    ϕ[1,2:jj-1,2:kk-1] .= ϕ[2,2:jj-1,2:kk-1]
    ϕ[ii,2:jj-1,2:kk-1] .= ϕ[ii-1,2:jj-1,2:kk-1]
    ϕ[2:ii-1,1,2:kk-1] .= ϕ[2:ii-1,2,2:kk-1]
    ϕ[2:ii-1,jj,2:kk-1] .= ϕ[2:ii-1,jj-1,2:kk-1]
    for j = 2:jj-1
        for i = 2:ii-1
            #10.1016/j.enbuild.2018.02.013
            ϕ[i,j,1] = (ϕ[i,j,2]*ks/dz-ϕs*hs)/(ks/dz-hs)
        end
    end
    nothing
end


# Initial and boundary conditions ##############################################

# Temperature at the center of the DBHEs. 1D model.
ϕc2 = ones(bb,kkb).*ϕ0.(rzb)'
ϕc1 = ones(bb,kkb).*ϕ0.(rzb)'
# Temperature at the annulus of the DBHEs. 1D model.
ϕa2 = ones(bb,kkb).*ϕ0.(rzb)'
ϕa1 = ones(bb,kkb).*ϕ0.(rzb)'
# Temperature at the DBHE wall. 1D model.
ϕbw = ones(bb,kkb).*ϕ0.(rzb)'
# Heat flux at the DBHE wall. 1D model.
qbw = zeros(bb,kkb)
update_q_dbhe_wall!(qbw,ϕa2,ϕbw,bb,kkb,dz,dt,Rb,Rs) # qbw

# Ground temperature. 3D model.
ϕ2 = zeros(ii,jj,kk)
ϕ1 = zeros(ii,jj,kk)
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
            if zk<=dbhe_depth
                for (xb,yb) in zip(xs,ys)
                    r = norm([xi,yj]-[xb,yb]) # Distance to DBHE center
                    if r<=dao/2
                        D[i,j,k] = 0
                        ϕ2[i,j,k] = ϕ0(zk)
                    end
                end
            end
        end
    end
end
ϕ1 .= ϕ2
save(path,"diff_coeff",D,rx,ry,rz,0)

# Inlet water temperature. 10.1016/j.renene.2021.07.086.
df = CSV.read("inlet-temp-vs-time.csv", DataFrame)
tin = (df."time (h)")*3600 # s
ϕin = df."water_temp (°C)".+273.15 # K
function get_ϕin(t;ϕin=ϕin,tin=tin)
    ind = maximum([findfirst(x->x>=t,tin)-1, 1])
    return ϕin[ind]
end

# Outlet water temperature. 10.1016/j.renene.2021.07.086.
df = CSV.read("outlet-temp-vs-time.csv", DataFrame)
tout = (df."time (h)")*3600 # s
ϕout = df."water_temp (°C)".+273.15 # K
function get_ϕout(t;ϕout=ϕout,tout=tout)
    ind = maximum([findfirst(x->x>=t,tout)-1, 1])
    return ϕout[ind]
end

dtt = tout[2]-tout[1]
tin = [[dtt*(i-1) for i in 1:0]; tin.+100*dtt]
tout = [[dtt*(i-1) for i in 1:0]; tout.+100*dtt]
ϕin = [[ϕin[1] for _ in 1:0]; ϕin]
ϕout = [[ϕout[1] for _ in 1:0]; ϕout]

# Heat extraction rate Q (W)
Qs = (get_ϕout.(tin)-get_ϕin.(tin))*(mfr*Cf)
function get_Q(t;Qs=Qs,tin=tin)
    ind = maximum([findfirst(x->x>=t,tin)-1, 1])
    return Qs[ind]
end

# Mass flow rate
mfrs = Q/((get_ϕout.(tin)-get_ϕin.(tin))*Cf)
function get_mfr(t;mfrs=mfrs,tin=tin)
    ind = maximum([findfirst(x->x>=t,tin)-1, 1])
    return mfrs[ind]
end

# Predicted inlet and outlet water temperature. 
ϕin_pred = zeros(bb,ceil(Int,tt/st)+1)
ϕout_pred = zeros(bb,ceil(Int,tt/st)+1)

# Saved times
ts = collect(0:st*dt:tt*dt)

# Simulation ###################################################################

# Run simulation
for nt = 0:2:tt

    # Save ϕ
    if nt % st == 0
        println("Iteration:$nt, time:$(round(nt*dt/60/60,digits=2))hs, " *
                "inlet temp:$(round(ϕa1[1,1].-273.15,digits=4))°C, " *
                "outlet temp:$(round(ϕc1[1,1].-273.15,digits=4))°C")
        plot_ϕ(ϕa1,ϕc1,ϕbw,rzb,nt*dt)
        plot_ϕ_ground(rx,ϕ1,nt*dt)

        ϕin_pred[:,(nt÷st)+1] = ϕa1[:,1]
        ϕout_pred[:,(nt÷st)+1] = ϕc1[:,1]
        #save(path,"ground_temp",ϕ2,rx,ry,rz,t)
        #save(path,"dbhe_annulus_temp",ϕa2,rx,ry,rz,t)
        #save(path,"dbhe_center_temp",ϕc2,rx,ry,rz,t)
    end

    # Update ϕ
    update_ϕ_fluid!(ϕa2,ϕa1,ϕc2,ϕc1,ϕbw,bb,kkb,dz,dt,nt,Rac,Rb,mfr,Cf,Ca,Cc,Q) # 1D: ϕa2,ϕa1,ϕc2,ϕc1
    update_q_dbhe_wall!(qbw,ϕa2,ϕbw,bb,kkb,dz,dt,Rb,Rs) # 1D: qbw
    update_ϕ_dbhe_wall!(ϕbw,ϕ2,qbw,ka,bb,kkb,dx,dy,dz,dt) # 1D: ϕbw
    update_ϕ_ground!(ϕ2,ϕ1,ϕbw,D,dx,dy,dz,dt,ii,jj,kk) # 3D: ϕ2,ϕ1
    update_ϕ_ground_bound!(ϕ2,ii,jj,kk,dx,dy,dz,hs,ks,ϕs) # 3D: ϕ2,ϕ1
    
    # Update ϕ
    update_ϕ_fluid!(ϕa1,ϕa2,ϕc1,ϕc2,ϕbw,bb,kkb,dz,dt,nt,Rac,Rb,mfr,Cf,Ca,Cc,Q) # 1D: ϕa1,ϕa2,ϕc1,ϕc2
    update_q_dbhe_wall!(qbw,ϕa1,ϕbw,bb,kkb,dz,dt,Rb,Rs) # 1D: qbw
    update_ϕ_dbhe_wall!(ϕbw,ϕ1,qbw,ka,bb,kkb,dx,dy,dz,dt) # 1D: ϕbw
    update_ϕ_ground!(ϕ1,ϕ2,ϕbw,D,dx,dy,dz,dt,ii,jj,kk) # 3D: ϕ1,ϕ2
    update_ϕ_ground_bound!(ϕ1,ii,jj,kk,dx,dy,dz,hs,ks,ϕs) # 3D: ϕ1,ϕ2
    
end


# Plots and validation #########################################################

plot_ϕ_inlet_outlet(ts,ϕin_pred,ϕout_pred)

