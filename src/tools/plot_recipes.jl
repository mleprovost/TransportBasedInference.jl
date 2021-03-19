using RecipesBase
using ColorTypes
import PlotUtils: cgrad
using LaTeXStrings

const mygreen = RGBA{Float64}(151/255,180/255,118/255,1)
const mygreen2 = RGBA{Float64}(113/255,161/255,103/255,1)
const myblue = RGBA{Float64}(74/255,144/255,226/255,1)

# Recipe to plot a heatmap of the number of terms for each variable,
# or we can alternatively plot the maximal degree for each varaible of each component if degree = true
@recipe function heatmap(M::HermiteMap; start::Int64=1, color = cgrad([:white, :teal, :navyblue, :purple]), degree::Bool=false)
    Nx = M.Nx
    idx = zeros(Int64, Nx-start+1, Nx)

    # Count occurence of each index
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
    xguide -->  "Index"
    yguide -->  "Map index"
    yflip --> true
    aspect_ratio --> 1
    # legend --> :none
    colorbar --> true
    clims --> (0, maximum(idx))
    seriescolor --> color
    collect(1:Nx), collect(start:Nx), idx
    end
end
