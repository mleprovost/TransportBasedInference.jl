export   LinearTransform,
         transform!,
         transform,
         itransform!,
         itransform

struct LinearTransform
    Nx::Int64

    # Flag for using diagonal or dense transformation
    μ::Array{Float64,1}

    L::Union{Diagonal, LowerTriangular}

    diag::Bool
end

function LinearTransform(X::Array{Float64,2}; diag::Bool=true)
    Nx, Ne = size(X)
    μ = mean(X; dims = 2)[:,1]

    if diag == true || Nx == 1
        σ = std(X; dims = 2)[:,1]
        L = Diagonal(σ)
        diag = true

    else #create a dense transformation from the Cholesky factor
        @assert Nx>1 "Only works for Nx>1, otherwise use first method"
        L = cholesky(cov(X')).L
    end

    return LinearTransform(Nx, μ, L, diag)
end


function transform!(L::LinearTransform, Xout::Array{Float64,2}, Xin::Array{Float64,2})
    @assert size(Xout,1) == size(Xin,1) "Input and output dimensions don't match"
    @assert size(Xin,1) == L.Nx "Input dimension is incorrect"

    copy!(Xout, Xin)
    Xout .-= L.μ

    ldiv!(L.L, Xout)
    # return Xout
end

function transform!(L::LinearTransform, X::Array{Float64,2}, idx::Union{UnitRange{Int64}, Array{Int64,1}})
    @assert size(X,1) == length(idx) "Input dimension is incorrect"
    X .-= view(L.μ,idx)
    if typeof(L.L)<:Diagonal
        ldiv!(Diagonal(view(L.L.diag,idx)), X)
    else
        error("Not yet implemented")
    end
    return X
end

function transform!(L::LinearTransform, X::Array{Float64,2})
    @assert size(X,1) == L.Nx "Input dimension is incorrect"
    X .-= L.μ
    ldiv!(L.L, X)
    return X
end


transform(X::Array{Float64,2}; diag::Bool = true) = transform!(LinearTransform(X; diag = diag), zero(X), X)

function itransform!(L::LinearTransform, Xout::Array{Float64,2}, Xin::Array{Float64,2})
    @assert size(Xout) == size(Xin) "Input and output dimensions don't match"
    @assert size(Xin,1) == L.Nx "Input dimension is incorrect"

    copy!(Xout, Xin)
    mul!(Xout, L.L, Xout)
    Xout .+= L.μ

    return Xout
end

function itransform!(L::LinearTransform, X::Array{Float64,2}, idx::Union{UnitRange{Int64}, Array{Int64,1}})
    @assert size(X,1) == L.Nx "Input dimension is incorrect"

    if typeof(L.L)<:Diagonal
        mul!(X, Diagonal(view(L.L.diag,idx)), X)
    else
        error("Not yet implemented")
    end

    # mul!(X, L.L, X)
    X .+= L.μ

    return X
end

function itransform!(L::LinearTransform, X::Array{Float64,2})
    @assert size(X,1) == L.Nx "Input dimension is incorrect"

    mul!(X, L.L, X)
    X .+= L.μ

    return X
end

itransform(X::Array{Float64,2}) = itransform!(LinearTransform(X), zero(X), X)
