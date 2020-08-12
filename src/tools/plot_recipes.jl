using RecipesBase
using ColorTypes
import PlotUtils: cgrad
using LaTeXStrings

const mygreen = RGBA{Float64}(151/255,180/255,118/255,1)
const mygreen2 = RGBA{Float64}(113/255,161/255,103/255,1)
const myblue = RGBA{Float64}(74/255,144/255,226/255,1)

@recipe function heatmap(M::HermiteMap; start::Int64=1)
    Nx = M.Nx
    idx = zeros(Int64, Nx-start+1, Nx)

    # Count occurence of each index
    for i=start:Nx
        for j=1:i
        idx[i-start+1, j] = sum(view(getidx(M[i]),:,j) .> 0)
        end
    end

    @series begin
    seriestype := :heatmap
    # size --> (600, 600)
    # xguide -->  "Index"
    # yguide -->  "Map index"
    aspect_ratio --> 1
    # legend --> :none
    colorbar --> true
    # colorbarlabel --> "Occurences"
    seriescolor --> cgrad([:white, :skyblue, :purple])

    collect(1:Nx), collect(start:Nx), reverse(idx; dims = 1)
    end
end
