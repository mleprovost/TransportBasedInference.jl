export  LinHermiteMapk,
        getcoeff,
        setcoeff!,
        getidx,
        evaluate!,
        evaluate

# LinHermiteMapk is a composition of a linear transformation that rescale
# the samples to get zero mean and unitary standard deviation component-wise
# and a nonlinear transport map described by a HermiteMapk element


struct LinHermiteMapk{m, Nψ, k}
    # Linear transformation
    L::LinearTransform{k}


    # IntegratedFunction
    H::HermiteMapk{m, Nψ, k}

    function LinHermiteMapk(L::LinearTransform{k}, Hk::HermiteMapk{m, Nψ, k}) where {m, Nψ, k}
        return new{m, Nψ, k}(L, Hk)
    end
end

function LinHermiteMapk(X::Array{Float64,2}, Hk::HermiteMapk{m, Nψ, k}; diag::Bool=true) where {m, Nψ, k, Nx}
    @assert size(X,1)==k "Wrong dimension of the input"
    L = LinearTransform(X; diag = diag)
    return LinHermiteMapk(L, Hk)
end


ncoeff(Lk::LinHermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Nψ
getcoeff(Lk::LinHermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Lk.H.I.f.f.coeff

function setcoeff!(Lk::LinHermiteMapk{m, Nψ, k}, coeff::Array{Float64,1}) where {m, Nψ, k}
        @assert size(coeff,1) == Nψ "Wrong dimension of coeff"
        L.Hk.I.f.f.coeff .= coeff
end

getidx(Lk::LinHermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Lk.H.I.f.f.idx


function evaluate!(out, Lk::LinHermiteMapk{m, Nψ, k}, X) where {m, Nψ, k}
    @assert k==size(X,1) "Wrong dimension of the sample"
    @assert size(out,1) == size(X,2) "Dimensions of the output and the samples don't match"

    transform!(Lk.L, X)
    evaluate!(out, Lk.H.I, X)
    itransform!(Lk.L, X)
    return out
end

evaluate(Lk::LinHermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k} =
    evaluate!(zeros(size(X,2)), Lk, X)
