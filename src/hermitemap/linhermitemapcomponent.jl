export  LinHermiteMapComponent,
        getcoeff,
        setcoeff!,
        getidx,
        evaluate!,
        evaluate



"""
$(TYPEDEF)

`LinHermiteMapComponent` is a type to hold  the composition of a `LinearTransform`: a linear transformation that rescales
the samples to get zero mean and unitary standard deviation component-wise
and  an `HermiteMapComponent` element.

## Fields

$(TYPEDFIELDS)
"""
struct LinHermiteMapComponent
    # Linear transformation
    L::LinearTransform

    # IntegratedFunction
    C::HermiteMapComponent

    function LinHermiteMapComponent(L::LinearTransform, C::HermiteMapComponent)
        return new(L, C)
    end
end

function LinHermiteMapComponent(X::Array{Float64,2}, C::HermiteMapComponent; diag::Bool=true)
    @assert size(X,1)==C.Nx "Wrong dimension of the input"
    L = LinearTransform(X; diag = diag)
    return LinHermiteMapComponent(L, C)
end

"""
$(TYPEDSIGNATURES)

Returns the number of multi-variate features of `L`.
"""
ncoeff(L::LinHermiteMapComponent) = L.C.Nψ
"""
$(TYPEDSIGNATURES)

Returns the coefficients of `L`.
"""
getcoeff(L::LinHermiteMapComponent) = L.C.I.f.coeff

"""
$(TYPEDSIGNATURES)

Sets the coefficients of `L`  to `coeff`.
"""
function setcoeff!(L::LinHermiteMapComponent, coeff::Array{Float64,1})
        @assert size(coeff,1) == Nψ "Wrong dimension of coeff"
        L.C.I.f.coeff .= coeff
end

"""
$(TYPEDSIGNATURES)

Extracts the multi-indices of `L`.
"""
getidx(L::LinHermiteMapComponent) = L.C.I.f.idx

"""
$(TYPEDSIGNATURES)

Evaluates in-place the `LinHermiteMapComponent` `L` for the ensemble matrix `X`.
"""
function evaluate!(out, L::LinHermiteMapComponent, X)
    @assert L.C.Nx==size(X,1) "Wrong dimension of the sample"
    @assert size(out,1) == size(X,2) "Dimensions of the output and the samples don't match"

    transform!(L.L, X)
    evaluate!(out, L.C.I, X)
    itransform!(L.L, X)
    return out
end

"""
$(TYPEDSIGNATURES)

Evaluates the `LinHermiteMapComponent` `L` for the ensemble matrix `X`.
"""
evaluate(L::LinHermiteMapComponent, X::Array{Float64,2}) = evaluate!(zeros(size(X,2)), L, X)
