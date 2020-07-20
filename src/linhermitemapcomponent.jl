export  LinHermiteMapk,
        getcoeff,
        setcoeff!,
        getidx,

# LinHermiteMapk is a composition of a linear transformation that rescale
# the samples to get zero mean and unitary standard deviation component-wise
# and a nonlinear transport map described by a HermiteMapk element


struct LinHermiteMapk{m, Nψ, k}
    # IntegratedFunction
    I::IntegratedFunction{m, Nψ, k}
    # Regularization parameter
    α::Float64

    function LinHermiteMapk(I::IntegratedFunction{m, Nψ, k}; α::Float64 = 1e-6) where {m, Nψ, k}
        return new{m, Nψ, k}(I, α)
    end
end


ncoeff(Hk::LinHermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Nψ
getcoeff(Hk::LinHermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Hk.I.f.f.coeff

function setcoeff!(Hk::LinHermiteMapk{m, Nψ, k}, coeff::Array{Float64,1}) where {m, Nψ, k}
        @assert size(coeff,1) == Nψ "Wrong dimension of coeff"
        Hk.I.f.f.coeff .= coeff
end

getidx(Hk::HermiteMapk{m, Nψ, k}) where {m, Nψ, k} = Hk.I.f.f.idx
