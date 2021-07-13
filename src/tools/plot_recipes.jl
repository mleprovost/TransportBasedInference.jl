using RecipesBase
using ColorTypes
import PlotUtils: cgrad
using LaTeXStrings

const mygreen = RGBA{Float64}(151/255,180/255,118/255,1)
const mygreen2 = RGBA{Float64}(113/255,161/255,103/255,1)
const myblue = RGBA{Float64}(74/255,144/255,226/255,1)

"""
        heatmap(M::HermiteMap; start::Int64=1, color, degree)

Plot recipe for an `ExpandedFunction`.  We can either plot
the number of occurences of each variable (columns) in each map component (rows) if `degree = false` (default behavior),
or the maximum multi-index of the features identified for each variable (columns) in each map component (rows) if `degree = true`.
"""
@recipe function heatmap(f::ExpandedFunction; color = cgrad([:white, :teal, :navyblue, :purple]))

    @series begin
    seriestype := :heatmap
    # size --> (600, 600)
    # xticks --> collect(1:f.Nx)
    # yticks --> collect(1:f.Nψ)
    xguide -->  "Dimension"
    yguide -->  "Feature Index"
    yflip --> true
    aspect_ratio --> 1
    colorbar --> true
    # levels --> 1:maximum(f.idx)
    clims --> (0, maximum(f.idx))
    seriescolor --> color
    collect(1:f.Nx), collect(1:f.Nψ), f.idx
    end
end


"""
        heatmap(M::HermiteMap; start::Int64=1, color, degree)

Plot recipe for an `HermiteMap`. We can either plot
the number of occurences of each variable (columns) in each map component (rows) if `degree = false` (default behavior),
or the maximum multi-index of the features identified for each variable (columns) in each map component (rows) if `degree = true`.
"""
@recipe function heatmap(M::HermiteMap; start::Int64=1, color = cgrad([:white, :teal, :navyblue, :purple]), degree::Bool=false)
    Nx = M.Nx
    idx = -1e-4*ones(Int64, Nx-start+1, Nx)

    # Count occurence or maximal degree of each index
    for i=start:Nx
        for j=1:i
        if degree == false
            idx[i-start+1, j] = sum(view(getidx(M[i]),:,j) .> 0)
        else
            idx[i-start+1, j] = maximum(view(getidx(M[i]),:,j))
        end
        end
    end

    @series begin
    seriestype := :heatmap
    # size --> (600, 600)
    xticks --> collect(1:Nx)
    yticks --> collect(start:Nx)
    xlim --> (-Inf, Inf)
    ylim --> (-Inf, Inf)
    xguide -->  "Index"
    yguide -->  "Map index"
    yflip --> true
    aspect_ratio --> 1
    colorbar --> true
    clims --> (0, maximum(idx))
    seriescolor --> color
    collect(1:Nx), collect(start:Nx), idx
    end
end

# @recipe function heatmap(C::SparseRadialMapComponent; color = cgrad([:white, :teal, :navyblue, :purple], categorical = true), degree::Bool=false)
@recipe function heatmap(C::SparseRadialMapComponent; color = cgrad([:white, :teal, :navyblue, :purple]), degree::Bool=false)
    Nx = C.Nx
    maxp = maximum(C.p)

    @series begin
    seriestype := :heatmap
    # xticks --> collect(1:Nx)
    # yticks --> collect(1:1)
    xguide -->  "Index"
    yguide -->  "Map index"
    yflip --> true
    aspect_ratio --> 1
    colorbar --> true
    clims --> (-1, maxp)
    levels --> length(-1:1:maxp)
    # colorbar_title --> "Order"
    seriescolor --> color
    collect(1:Nx), collect(1:1), reshape(C.p, Nx, 1)
    end
end

@recipe function heatmap(M::RadialMap; start::Int64=1, color = cgrad([:white, :teal, :navyblue, :purple]), degree::Bool=false)
    Nx = M.Nx

    # Store order of the different map components
    order = fill(-1, (Nx-start+1, Nx))
    for i=start:Nx
        fill!(view(order,i-start+1,1:i), M.p[i])
    end


    @series begin
    seriestype := :heatmap
    # xticks --> collect(1:Nx)
    # yticks --> collect(start:Nx)
    xguide -->  "Index"
    yguide -->  "Map index"
    yflip --> true
    aspect_ratio --> 1
    colorbar --> true
    # colorbar_title --> "Order"
    seriescolor --> color
    collect(1:Nx), collect(start:Nx), order
    end
end

@recipe function heatmap(M::SparseRadialMap; start::Int64=1, color = cgrad([:white, :teal, :navyblue, :purple]), degree::Bool=false)
    Nx = M.Nx
    idx = -1e-4*ones(Int64, Nx-start+1, Nx)

    # Store order of the different map components
    order = fill(-1, (Nx-start+1, Nx))
    for i=start:Nx
        view(order,i-start+1,1:i) .= M.p[i]
    end


    @series begin
    seriestype := :heatmap
    # xticks --> collect(1:Nx)
    # yticks --> collect(start:Nx)
    xguide -->  "Index"
    yguide -->  "Map index"
    yflip --> true
    aspect_ratio --> 1
    colorbar --> true
    # colorbar_title --> "Order"
    seriescolor --> color
    collect(1:Nx), collect(start:Nx), order
    end
end
