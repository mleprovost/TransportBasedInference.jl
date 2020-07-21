export   LinearTransform,
         transform!,
         transform


struct LinearTransform{Nx, Ne}
    # Flag for using diagonal or dense transformation
    μ::Array{Float64,1}

    L::LinearMap

    diag::Bool
end

function LinearTransform(X::Array{Float64,2}; diag::Bool=true)
    Nx, Ne = size(X)
    μ = mean(X; dims = 2)[:,1]

    if diag == true || Nx == 1
        σ = std(X; dims = 2)[:,1]
        L = LinearMap(Diagonal(1 ./σ))
        diag = true

    else #create a dense transformation from the Cholesky factor
        @assert Nx>1 "Only works for Nx>1, otherwise use first method"
        L = LinearMap(inv(cholesky(cov(X')).L))
    end

    return LinearTransform{Nx, Ne}(μ, L, diag)
end


function transform!(L::LinearTransform{Nx, Ne}, Xout::Array{Float64,1}, Xin::Array{Float64,1}) where {Nx, Ne}
    @assert size(Xout,1) == size(Xin,1) "Input and output dimensions don't match"
    @assert size(Xin,1) == Nx "Input dimension is incorrect"

    copy!(Xout, Xin)
    Xout .-= L.μ
    if L.diag == true
        mul!(Xout, L.L, Xout)
    else
        mul!(Xout, L.L.lmap, Xout)
    end
    return Xout
end



function transform!(L::LinearTransform{Nx, Ne}, X::Array{Float64,1}) where {Nx, Ne}
    @assert size(X,1) == Nx "Input dimension is incorrect"

    X .-= L.μ
    if L.diag == true
        mul!(X, L.L, X)
    else
        mul!(X, L.L.lmap, X)
    end
    return X
end

function transform!(L::LinearTransform{Nx, Ne}, Xout::Array{Float64,2}, Xin::Array{Float64,2}) where {Nx, Ne}
    @assert size(Xout) == size(Xin) "Input and output dimensions don't match"
    @assert size(Xin) == (Nx, Ne) "Input dimension is incorrect"

    copy!(Xout, Xin)
    Xout .-= L.μ
    if L.diag == true
        mul!(Xout, L.L, Xout)
    else
        mul!(Xout, L.L.lmap, Xout)
    end
    return Xout
end

function transform!(L::LinearTransform{Nx, Ne}, X::Array{Float64,2}) where {Nx, Ne}
    @assert size(X) == (Nx, Ne) "Input dimension is incorrect"

    X .-= L.μ
    if L.diag == true
        mul!(X, L.L, X)
    else
        mul!(X, L.L.lmap, X)
    end
    return X
end
