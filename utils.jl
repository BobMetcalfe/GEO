# Auxiliary functions  ##########################################################
function save(path,name,var,rx,ry,rz,t)
    vtk_grid("$path/$name-$t", rx, ry, rz; compress = false, append = false, ascii = false) do vtk
        vtk[name] = var
    end
end