# Helper functions  ##########################################################

# Save vtk
function save(path,name,var,rx,ry,rz,t)
    vtk_grid("$path/$name-$t", rx, ry, rz; compress = false, append = false, ascii = false) do vtk
        vtk[name] = var
    end
end

# Plot ground temperature
function plot_ϕ_ground(rx,ϕ,t)
    rzrev = reverse(-1*rz[1:end-1])
    @views ϕrev = reverse(ϕ[ii÷2,:,:]'.-273.15, dims=1)
    heatmap(rx, rzrev, ϕrev, colorbar_title = "Temperature (°C)", colormap=:jet1)
    contour!(rx, rzrev, ϕrev, linewidth = 1, linecolor = :black)
    contour!(xlabel="Distance [m]", ylabel="Depth [m]")
    savefig("$path/ground-temp-$t.png")
end

# Plot DBHE temperature
function plot_ϕ(ϕa,ϕc,ϕbw,rzb,t)
    plot(rzb, ϕa[1,:].-273.15, label="Temperature at DBHE annulus fluid [°C].")
    plot!(rzb, ϕc[1,:].-273.15, label="Temperature at DBHE center fluid [°C].")
    plot!(rzb, ϕbw[1,:].-273.15, label="Temperature at DBHE wall [°C].")
    plot!(xlabel="depth [m]", ylabel="Temperature [°C]", ylims=(5, 40))
    savefig("$path/temp-dbhe-$t.png")
end

# Plot ground temperature
function plot_ϕ_ground(rx,ϕ,t)
    rzrev = reverse(-1*rz[1:end-1])
    @views ϕrev = reverse(ϕ[ii÷2,:,:]'.-273.15, dims=1)
    heatmap(rx, rzrev, ϕrev, colorbar_title = "Temperature (°C)", colormap=:jet1)
    contour!(rx, rzrev, ϕrev, linewidth = 1, linecolor = :black)
    contour!(xlabel="Distance [m]", ylabel="Depth [m]")
    savefig("$path/ground-temp-$t.png")
end


# Predicted Inlet and outlet temperature vs measurements
function plot_ϕ_inlet_outlet(ts,ϕin_pred,ϕout_pred)
    plot(ts/3600, get_ϕin.(ts).-273.15,
          label="Inlet temperature [°C]")
    scatter!(ts/3600, ϕin_pred[1,:].-273.15,
          label="Predicted inlet temperature [°C]")
    plot!(ts/3600, get_ϕout.(ts).-273.15,
          label="Outlet temperature [°C]")
    scatter!(ts/3600, ϕout_pred[1,:].-273.15,
          label="Predicted outlet temperature [°C]")
    plot!(xlabel="time [h]", ylabel="Temperature [°C]", ylims=(5, 40))
    savefig("$path/inlet-oulet-temp.png")
end
