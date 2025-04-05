# Auxiliary functions  ##########################################################
function save(path,name,var,rx,ry,rz,t)
    vtk_grid("$path/$name-$t", rx, ry, rz; compress = false, append = false, ascii = false) do vtk
        vtk[name] = var
    end
end

function plot_ϕ(ϕa,ϕc,ϕbw,rzb,t)
    plot(rzb, ϕa[1,:].-273.15, label="Temperature at DBHE annulus fluid [°C].")
    plot!(rzb, ϕc[1,:].-273.15, label="Temperature at DBHE center fluid [°C].")
    plot!(rzb, ϕbw[1,:].-273.15, label="Temperature at DBHE wall [°C].")
    plot!(xlabel="depth [m]", ylabel="Temperature [°C]", ylims=(5, 80))
    savefig("$path/temp-dbhe-$t.png")
end
