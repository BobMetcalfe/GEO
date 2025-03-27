## Main Validation Paper: A slightly inclined deep borehole heat exchanger array behaves better than vertical installation
## Second Paper: Proposing stratiﬁed segmented ﬁnite line source (SS-FLS) method for dynamic 
#                simulation of medium-deep coaxial borehole heat exchanger in multiple ground layers

# TODO: CFL condition

### Parameters ###
ϕ0 = 25.0 # [°C] surface temperature
ϕf0 = 20.0 # [°C] initial temperature of the fluid
m = 25/3600 # [m^3/s] mass flow rate
d_b = 0.2159 # [m] well diameter
H = 2000 # [m] well depth
d_ao = 0.1778 # [m] annular pipe outer diameter (d1,o)
th_a = 0.0092 # [m] thickness of the annular pipe
d_co = 0.1100 # [m] center pipe outer diameter (d2,o)
th_c = 0.01 # [m] thickness of the center pipe
ρ_f = 998 # [kg/m³] fluid density
λ_f = 0.6 # [W/mK] fluid thermal conductivity
c_f = 4190 # [J/kgK] fluid specific heat
μ_f = 0.000931 # [Pa.s] fluid dynamic viscosity
ρ_g = 2190 # [kg/m³] grout density
λ_g = 0.63 # [W/mK] grout thermal conductivity
c_g = 1735 # [J/kgK] grout specific heat
g = 33 # [°C/km] geothermal gradient

### NOTE: The following parameter values are not provided in the validation paper
# Pipe parameters
ρ_p = 7700 # [kg/m³] pipe density
λ_1p = 0.40 # [W/mK] thermal conductivity of the annular(outer) pipe, (from second paper)
λ_2p = 0.40 # [W/mK] thermal conductivity of the center(inner) pipe, (from second paper)
c_p = 420 # [J/kgK] specific heat of the pipe

# Rock parameters (assumed granite)
c_r = 790 # [J/kgK], specific heat of the rock
ρ_r = 2700 # [kg/m³], density of the rock
λ_r =  3.12 # [W/mK], thermal conductivity of the rock

α_r = λ_r/(ρ_r*c_r) # [m²/s] thermal diffusivity of the rock

## Computing the convection heat Coefficients
# Assumptions: Forced convection, turbulent internal flow through a cylinder
# Dittus-Boelter correlation: Nu = 0.023 * Re^0.8 * Pr^0.4, source: https://www.sciencedirect.com/topics/engineering/dittus-boelter-correlation
# Re = ρvD/μ (density, velocity, diameter, dynamic viscosity - in that order): https://www.sciencedirect.com/topics/physics-and-astronomy/reynolds-number
# Pr = cμ/λ (s.h.c, dynamic viscosity, thermal conductivity): source: wikipedia, https://en.wikipedia.org/wiki/Prandtl_number

###
d_ai = d_ao - 2*th_a # [m] annular pipe inner diameter (d1,i)
d_ci = d_co - 2*th_c # [m] center pipe inner diameter (d2,i)
A1 = π * ((d_ai/2)^2 - (d_co/2)^2) # cross sectional area of annular space
A2 =  π * (d_ci/2)^2 # cross sectional area of center space
C1 = π/4 * (d_ai^2 - d_co^2)*ρ_f*c_f + π/4 * (d_b^2 - d_ao^2)*ρ_g*c_g

# # Convective Heat Transfer Coefficients
# h3: between water and inner wall of the annular pipe
d_ch3 = (d_ai - d_co) # hydraulic diameter
Re3 = ρ_f * (m / A2) * d_ch3
Pr = c_f * μ_f / λ_f
Nu = 0.023 * Re3^0.8 * Pr^0.4
h3 = Nu * λ_f/d_ch3
# h2: between water and outer wall of center pipe
h2 = h3 # TODO: double-check 
# h1: between water and inner wall of center pipe
d_ch1 = d_ci # hydraulic diameter
Re1 = ρ_f * (m / A1) * d_ch1
Nu = 0.023 * Re1^0.8 * Pr^0.4
h1 = Nu * λ_f/d_ch1
##

R_12 = 1/(π*d_ci*h1) + log(d_co/d_ci)/(2*π*λ_2p) + 1/(π*d_co*h2) 
R_b = 1/(π*d_ai*h3) + log(d_ao/d_ai)/(2*π*λ_1p) + log(d_b/d_ao)/(2*π*λ_g)
C2 = π/4*(d_ci^2)*ρ_f*c_f + π/4 * (d_co^2 - d_ci^2)*ρ_p*c_p

### NOTE: the following values are also not provided in the paper
Δt = 0.1 
Δx, Δy, Δz = 0.1, 0.1, 0.1

# Grid Dimensions 
L, W, D = 10, 10, 5 # [m]
Nx, Ny, Nz = ceil(Int,L/Δx), ceil(Int,W/Δy), ceil(Int,D/Δz)

# Well Positions
well_positions = [(5, 5)] # [(x1, y1),...(xn, yn)]
###

## Fundamental Assumptions ## => Add to the paper
# - groundwater flow is ignored i.e only conduction is considered
# - thermophysical properties do no change with temperature
# - Inside the well, the fluid temperature distribution is uniform across the cross-section

# Initialize the temperature field
function init_temperature(num_wells)
    ϕ = zeros(Nx, Ny, Nz)
    ϕ_a = fill(ϕf0, (num_wells, Nz)) # annular space
    ϕ_c = fill(ϕf0, (num_wells, Nz)) # center space
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        ϕ[i, j, k] = ϕ0 + g * k * Δz
    end
    return ϕ, ϕ_a, ϕ_c
end

## Heat Transfer Update
function update_temperate!(ϕ, ϕ_a, ϕ_c, well_positions, step)
    ϕ_new = copy(ϕ)
    ϕ_a_new = copy(ϕ_a)
    ϕ_c_new = copy(ϕ_c)

    for (idx, (x,y)) in enumerate(well_positions) # superposition
        Threads.@threads for i in 2:Nx-1
            for j in 2:Ny-1
                for k in 2:Nz-1
                    ## TODO: take care of boundaries
                    # if i == 1
                    #     # TODO
                    # elseif i == Nx
                    #     # TODO
                    # if j == 1

                    # elseif j == Ny

                    # end
                    # if k == Nz

                    # elseif 

                    # end
                    if abs(x-i*Δx) < d_b/2.0 && abs(y-j*Δy) < d_b/2.0 # inside a well
                        ## center space
                        ϕ_c_new[idx, k] = ϕ_c[idx, k] + 
                                        ((ϕ_a[idx,k]-ϕ_c[idx,k])/R_12 + 
                                        m*c_f*(ϕ_c[idx,k+1]-ϕ_c[idx,k-1])/(2*Δz) + 
                                        A2*λ_f*(ϕ_c[idx,k+1]-2*ϕ_c[idx,k]+ϕ_c[idx,k-1])/(Δz^2)) *
                                        (Δt/C2)
                        ## annular space
                        ϕ_a_new[idx,k] = ϕ_a[idx,k] + 
                                    ((ϕ_c[idx,k]-ϕ_a[idx,k])/R_12 + 
                                    (ϕ[i,j,k]-ϕ_a[idx,k])/R_b - 
                                    m*c_f*(ϕ_a[idx,k+1]-ϕ_a[idx,k-1])/(2*Δz) + 
                                    A1*λ_f*(ϕ_a[idx,k+1]-2*ϕ_a[idx,k]+ϕ_a[idx,k-1])/(Δz^2)) * 
                                    (Δt/C1)

                    elseif abs(x-i*Δx) < d_b/2.0 + Δx && abs(y-j*Δy) < d_b/2.0 + Δy
                        ## Rock formation bordering the well
                        # Based on equation 9
                        if i >= x
                            a = (ϕ[i+1,j,k]-ϕ[i,j,k])/Δx
                        else
                            a = (ϕ[i,j,k]-ϕ[i-1,j,k])/Δx
                        end
                        if j >= y
                            b = (ϕ[i,j+1,k]-ϕ[i,j,k])/Δy
                        else
                            b = (ϕ[i,j,k]-ϕ[i,j-1,k])/Δy
                        end
                        c = (ϕ[i,j,k] - ϕ_a[k])/log(d_b/d_ai) # doublecheck

                        sum_ = 0.0
                        a_r = λ_r/(ρ_r*c_r)
                        x_i = i * Δx
                        y_i = j * Δy
                        z_i = k * Δz
                        τ = step * Δt
                        for z_ in 1:Nz
                            q = - λ_r * 
                                (a + b + c +
                                (ϕ[i,j,k+1]-ϕ[i,j,k-1])/(2*Δz)) # q(z_,t)
                            sum_ += q / (8*ρ_r*c_r*π*a_r*τ) * exp(-(x_i^2 + y_i^2 + (z_i - z_*Δz)^2)/(4*a_r*τ))
                        end
                        ϕ_new[i,j,k] = ϕ[i,j,k] + sum_ * Δt * Δz
                                    
                    else 
                        ## Rock formation
                        ## Based on Equation 9
                        sum_ = 0.0
                        a_r = λ_r/(ρ_r*c_r)
                        x_i = i * Δx
                        y_i = j * Δy
                        z_i = k * Δz
                        τ = step * Δt
                        for z_ in 1:Nz
                            q = - λ_r * 
                                ((ϕ[i+1,j,k]-ϕ[i-1,j,k])/(2*Δx) + 
                                (ϕ[i,j+1,k]-ϕ[i,j-1,k])/(2*Δy) + 
                                (ϕ[i,j,k+1]-ϕ[i,j,k-1])/(2*Δz)) # q(z_,t)
                            sum_ += q / (8*ρ_r*c_r*π*a_r*τ) * exp(-(x_i^2 + y_i^2 + (z_i - z_)^2)/(4*a_r*τ))
                        end
                        ϕ_new[i,j,k] = ϕ[i,j,k] + sum_ * Δt * Δz
                    end
                end
            end
        end

        x_i, y_i = floor(Int, (x + d_b/2.0 + Δx)/Δx), floor(Int, y/Δy)
        # Connect the annular column to the center column at the bottom
        ϕ_c_new[idx, Nz] = ϕ_c[idx, Nz] + 
                        ((ϕ_a[idx,Nz]-ϕ_c[idx,Nz])/R_12 + 
                        m*c_f*(ϕ_a[idx,Nz]-ϕ_c[idx,Nz-1])/Δz + 
                        A2*λ_f*(ϕ_a[idx,Nz]-2*ϕ_c[idx,Nz]+ϕ_c[idx,Nz-1])/(Δz^2)) *
                        (Δt/C2)

        ϕ_a_new[idx,Nz] = ϕ_a[idx,Nz] + 
                        ((ϕ_c[idx,Nz]-ϕ_a[idx,Nz])/R_12 + 
                        (ϕ[x_i,y_i,Nz]-ϕ_a[idx,Nz])/R_b - 
                        m*c_f*(ϕ_c[idx,Nz]-ϕ_a[idx,Nz-1])/(2*Δz) + 
                        A1*λ_f*(ϕ_c[idx,Nz]-2*ϕ_a[idx,Nz]+ϕ_a[idx,Nz-1])/(Δz^2)) * 
                        (Δt/C1)

        # Update the top of the well
        # central pipe
        ϕ_c_new[idx, 1] = ϕ_c[idx, 1] + 
                        ((ϕ_a[idx,1]-ϕ_c[idx,1])/R_12 + 
                        m*c_f*(ϕ_c[idx,2]-ϕ_c[idx,1])/Δz + 
                        A2*λ_f*(ϕ_c[idx,3]-2*ϕ_c[idx,2]+ϕ_c[idx,1])/(Δz^2)) *
                        (Δt/C2)

        ## annular space
        ϕ_a_new[idx,1] = ϕ_a[idx,1] + 
                        ((ϕ_c[idx,1]-ϕ_a[idx,1])/R_12 + 
                        (ϕ[x_i,y_i,1]-ϕ_a[idx,1])/R_b - 
                        m*c_f*(ϕ_a[idx,2]-ϕ_a[idx,1])/Δz + 
                        A1*λ_f*(ϕ_a[idx,3]-2*ϕ_a[idx,2]+ϕ_a[idx,1])/(Δz^2)) * 
                        (Δt/C1)
        
        ϕ .= ϕ_new
        ϕ_a .= ϕ_a_new
        ϕ_c .= ϕ_c_new
    end
end

simulation_time = 300 # seconds
num_wells = length(well_positions)

## Simulation Loop
ϕ, ϕ_a, ϕ_c = init_temperature(num_wells)
simulation_steps = ceil(Int, simulation_time / Δt)

heat_output = 0.0
for step in 1:simulation_steps
    update_temperate!(ϕ, ϕ_a, ϕ_c, well_positions, step)
    for idx in 1:length(well_positions)
        Δϕ = ϕ_c[idx, 1] - ϕf0
        global heat_output += m * c_f * Δϕ
    end
    println("Heat Output:", heat_output)
end


## Notes
# - Look for opportunities to parallelize the code 
# - move to cluster (if need be)
# - A plot of the outlet temperature against time

### TODO
# - Validation - possibly email the authors [IMPORTANT]