export totalordermapcomponent, totalordermap


function totalordermapcomponent(Nx::Int64, order::Int64; withconstant::Bool = false, b::String = "CstProHermite")
    @assert order >= 0 "Order should be positive"
    if b ∈ ["CstProHermite"; "CstPhyHermite"]
        B = MultiBasis(eval(Symbol(b))(order+2), Nx)
    elseif b ∈ ["CstLinProHermite"; "CstLinPhyHermite"]
        B = MultiBasis(eval(Symbol(b))(order+3), Nx)
    else
        error("Undefined basis")
    end

    idx = totalorder(order*ones(Int64, Nx))

    if withconstant == false
        idx = idx[2:end,:]
    end

    Nψ = size(idx, 1)

    f = ExpandedFunction(B, idx, zeros(Nψ))
    return HermiteMapComponent(IntegratedFunction(f))
end

function totalordermap(X::Array{Float64,2}, order::Int64; diag::Bool=true, factor::Float64 = 1.0, withconstant::Bool = false, b::String = "CstProHermite")

    L = LinearTransform(X; diag = diag, factor = factor)

    Nx = size(X,1)
    C = HermiteMapComponent[]
    @inbounds for i=1:Nx
        push!(C, totalordermapcomponent(i, order; withconstant = withconstant, b = b))
    end

    if b ∈ ["CstProHermite"; "CstPhyHermite"]
        m = order+2
    elseif b ∈ ["CstLinProHermite"; "CstLinPhyHermite"]
        m = order+3
    else
        error("Undefined basis")
    end

    return HermiteMap(m, Nx, L, C)
end
