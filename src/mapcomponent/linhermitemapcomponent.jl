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
    L::LinearTransform

    # IntegratedFunction
    H::HermiteMapk{m, Nψ, k}

    function LinHermiteMapk(L::LinearTransform{Nx, Ne}, H::HermiteMapk{m, Nψ, k}) where {m, Nψ, k, Nx, Ne}
        return new{m, Nψ, k}(L, H)
    end
end


ncoeff(Lk::LinHermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Nψ
getcoeff(Lk::LinHermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Lk.H.I.f.f.coeff

function setcoeff!(Lk::LinHermiteMapk{m, Nψ, k}, coeff::Array{Float64,1}) where {m, Nψ, k}
        @assert size(coeff,1) == Nψ "Wrong dimension of coeff"
        L.Hk.I.f.f.coeff .= coeff
end

getidx(Lk::LinHermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Lk.H.I.f.f.idx


function evaluate!(out::Array{Float64,1}, Lk::LinHermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k}
    @assert k==size(X,1) "Wrong dimension of the sample"
    @assert size(out,1) == size(X,2) "Dimensions of the output and the samples don't match"
    Xout = zero(X)
    transform!(Lk.L, Xout, X)
    return evaluate!(out, Hk.I, Xout)
end

evaluate(out::Array{Float64,1}, Lk::LinHermiteMapk{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k} =
    evaluate!(zeros(size(X,2)), Lk, X)
