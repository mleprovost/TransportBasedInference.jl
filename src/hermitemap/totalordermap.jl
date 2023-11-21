export totalordermapcomponent, totalordermap

"""
$(TYPEDSIGNATURES)

Creates an `HermitemapComponent` of order `order` and dimension `Nx` for the basis `b`.
The features of the created maps are all the tensorial products of the basis elements up to the order `order`.
"""
function totalordermapcomponent(Nx::Int64, order::Int64; withconstant::Bool = false, b::String = "CstProHermiteBasis")
    @assert order >= 0 "Order should be positive"
    if b ∈ ["CstProHermiteBasis"; "CstPhyHermiteBasis"]
        MB = MultiBasis(eval(Symbol(b))(order+2), Nx)
    elseif b ∈ ["CstLinProHermiteBasis"; "CstLinPhyHermiteBasis"]
        MB = MultiBasis(eval(Symbol(b))(order+3), Nx)
    else
        error("Undefined basis")
    end

    idx = totalorder(order*ones(Int64, Nx))

    if withconstant == false
        idx = idx[2:end,:]
    end

    Nψ = size(idx, 1)

    f = ExpandedFunction(MB, idx, zeros(Nψ))
    return HermiteMapComponent(IntegratedFunction(f))
end

"""
$(TYPEDSIGNATURES)

Creates an `Hermitemap` of order `order` for the basis `b`. The size of the map is extracted from the ensemble matrix `X`.
The features of the created maps are all the tensorial products of the basis elements up to the order `order`.
"""
function totalordermap(X::Array{Float64,2}, order::Int64; diag::Bool=true, factor::Float64 = 1.0, withconstant::Bool = false, b::String = "CstProHermite")

    L = LinearTransform(X; diag = diag, factor = factor)

    Nx = size(X,1)
    C = HermiteMapComponent[]
    @inbounds for i=1:Nx
        push!(C, totalordermapcomponent(i, order; withconstant = withconstant, b = b))
    end

    if b ∈ ["CstProHermiteBasis"; "CstPhyHermiteBasis"]
        m = order+2
    elseif b ∈ ["CstLinProHermiteBasis"; "CstLinPhyHermiteBasis"]
        m = order+3
    else
        error("Undefined basis")
    end

    return HermiteMap(m, Nx, L, C)
end
