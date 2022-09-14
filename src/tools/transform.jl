export   LinearTransform,
         MinMaxTransform,
         updateLinearTransform,
         transform!,
         transform,
         itransform!,
         itransform

 """
 $(TYPEDEF)

 This type holds an affine transformation of the form x-> μ + Lx, where μ is an offset and L is a linear operator.
 We typically use this type to normalize samples to have zero mean and unit covariance of element-wise standard deviation.


 ## Fields

 $(TYPEDFIELDS)

 """
struct LinearTransform
    Nx::Int64

    # Flag for using diagonal or dense transformation
    μ::Array{Float64,1}

    L::Union{Diagonal, LowerTriangular}

    diag::Bool
end

# Identity transformation
LinearTransform(Nx::Int64) = LinearTransform(Nx, zeros(Nx), Diagonal(ones(Nx)), true)


function LinearTransform(X::AbstractMatrix{Float64}; diag::Bool=true, factor::Float64=1.0)
    # The entry `factor` allows to set the entry-wise standard deviation to a value different from 1 
    Nx, Ne = size(X)
    μ = deepcopy(mean(X; dims = 2)[:,1])

    if diag == true || Nx == 1
        σ = std(X; dims = 2)[:,1]
        L = deepcopy((1.0/factor)*Diagonal(σ))
        diag = true
    else #create a dense transformation from the Cholesky factor
        @assert Nx>1 "Only works for Nx>1, otherwise use first method"
        L = deepcopy((1.0/factor)*cholesky(cov(X')).L)
    end

    return LinearTransform(Nx, μ, L, diag)
end

function updateLinearTransform!(Lin::LinearTransform, X::AbstractMatrix{Float64}; diag::Bool=true)
    Nx, Ne = size(X)
    Lin.μ .= copy(mean(X; dims = 2)[:,1])

    if diag == true || Nx == 1
        σ = copy(std(X; dims = 2)[:,1])
        Lin.L .= Diagonal(σ)

    else #create a dense transformation from the Cholesky factor
        @assert Nx>1 "Only works for Nx>1, otherwise use diagonal scaling"
        Lin.L = copy(cholesky(cov(X')).L)
    end
end

# For scalar vectors
function transform!(L::LinearTransform, xout::AbstractVector{Float64}, xin::AbstractVector{Float64})
    @assert size(xout,1) == size(xin,1) "Input and output dimensions don't match"
    @assert size(xin,1) == L.Nx "Input dimension is incorrect"

    copy!(xout, xin)
    xout -= L.μ

    ldiv!(L.L, xout)
end

function transform!(L::LinearTransform, x::AbstractVector{Float64})
    @assert size(x,1) == L.Nx "Input dimension is incorrect"
    x -= L.μ
    ldiv!(L.L, x)
    return x
end

transform(L::LinearTransform, x::AbstractVector{Float64}) = transform!(L, zero(x), x)


function itransform!(L::LinearTransform, xout::AbstractVector{Float64}, xin::AbstractVector{Float64})
    @assert size(xout) == size(xin) "Input and output dimensions don't match"
    @assert size(xin,1) == L.Nx "Input dimension is incorrect"

    copy!(xout, xin)
    mul!(xout, L.L, xout)
    xout += L.μ

    return xout
end

function itransform!(L::LinearTransform, x::AbstractVector{Float64}, idx::Union{UnitRange{Int64}, Array{Int64,1}})
    @assert size(x,1) == L.Nx "Input dimension is incorrect"

    if typeof(L.L)<:Diagonal
        mul!(x, Diagonal(view(L.L.diag,idx)), x)
    else
        error("Not yet implemented")
    end

    # mul!(X, L.L, X)
    x += L.μ

    return x
end

function itransform!(L::LinearTransform, x::AbstractVector{Float64})
    @assert size(x,1) == L.Nx "Input dimension is incorrect"

    mul!(x, L.L, x)
    x += L.μ

    return x
end

itransform(L::LinearTransform, x::AbstractVector{Float64}) = itransform!(L, zero(x), x)


# For ensemble matrix
function transform!(L::LinearTransform, Xout::AbstractArray{Float64}, Xin::AbstractMatrix{Float64})
    @assert size(Xout,1) == size(Xin,1) "Input and output dimensions don't match"
    @assert size(Xin,1) == L.Nx "Input dimension is incorrect"

    copy!(Xout, Xin)
    Xout .-= L.μ

    ldiv!(L.L, Xout)
    # return Xout
end

function transform!(L::LinearTransform, X::AbstractMatrix{Float64}, idx::Union{UnitRange{Int64}, Array{Int64,1}})
    @assert size(X,1) == length(idx) "Input dimension is incorrect"
    X .-= view(L.μ,idx)
    if typeof(L.L)<:Diagonal
        ldiv!(Diagonal(view(L.L.diag,idx)), X)
    else
        error("Not yet implemented")
    end
    return X
end

function transform!(L::LinearTransform, X::AbstractMatrix{Float64})
    @assert size(X,1) == L.Nx "Input dimension is incorrect"
    X .-= L.μ
    ldiv!(L.L, X)
    return X
end

transform(L::LinearTransform, X::AbstractMatrix{Float64}) = transform!(L, zero(X), X)

transform(X::AbstractMatrix{Float64}; diag::Bool = true) = transform!(LinearTransform(X; diag = diag), zero(X), X)

function itransform!(L::LinearTransform, Xout::AbstractMatrix{Float64}, Xin::AbstractMatrix{Float64})
    @assert size(Xout) == size(Xin) "Input and output dimensions don't match"
    @assert size(Xin,1) == L.Nx "Input dimension is incorrect"

    copy!(Xout, Xin)
    mul!(Xout, L.L, Xout)
    Xout .+= L.μ

    return Xout
end

function itransform!(L::LinearTransform, X::AbstractMatrix{Float64}, idx::Union{UnitRange{Int64}, Array{Int64,1}})
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

function itransform!(L::LinearTransform, X::AbstractMatrix{Float64})
    @assert size(X,1) == L.Nx "Input dimension is incorrect"

    mul!(X, L.L, X)
    X .+= L.μ

    return X
end

itransform(L::LinearTransform, X::AbstractMatrix{Float64}) = itransform!(L, zero(X), X)



# Define a min-max transformation to use with the package c3
struct MinMaxTransform
    Nx::Int64

    L::Diagonal #(maximum(X[i,:] - minimum(X[i,:]))

    b::Array{Float64,1} #minimum(X[i,:])/(maximum(X[i,:] - minimum(X[i,:]))
end


function MinMaxTransform(X::AbstractMatrix{Float64})
    Nx = size(X,1)
     b = zeros(Nx)
     L = Diagonal(zeros(Nx))

     @inbounds for i=1:Nx
         xi = view(X,i,:)
         Mi = maximum(xi)
         mi = minimum(xi)
         @assert Mi>mi "Minimum and maximum are equal"
         L.diag[i] = (Mi - mi)/2
         b[i] = (Mi + mi)/(Mi - mi)
     end
    return MinMaxTransform(Nx, L, b)
end

function transform!(L::MinMaxTransform, Xout::AbstractMatrix{Float64}, Xin::AbstractMatrix{Float64})
    @assert size(Xout,1) == size(Xin,1) "Input and output dimensions don't match"
    @assert size(Xin,1) == L.Nx "Input dimension is incorrect"

    copy!(Xout, Xin)
    ldiv!(L.L, Xout)
    Xout .-= L.b


    # return Xout
end

function transform!(L::MinMaxTransform, X::AbstractMatrix{Float64})
    @assert size(X,1) == L.Nx "Input dimension is incorrect"
    ldiv!(L.L, X)
    X .-= L.b
    return X
end

transform(L::MinMaxTransform, X::AbstractMatrix{Float64}) = transform!(L, zero(X), X)


function itransform!(L::MinMaxTransform, Xout::AbstractMatrix{Float64}, Xin::AbstractMatrix{Float64})
    @assert size(Xout,1) == size(Xin,1) "Input and output dimensions don't match"
    @assert size(Xin,1) == L.Nx "Input dimension is incorrect"

    copy!(Xout, Xin)
    Xout .+= L.b
    mul!(Xout, L.L, Xout)
    return Xout
end

function itransform!(L::MinMaxTransform, X::AbstractMatrix{Float64})
    X .+= L.b
    mul!(X, L.L, X)
    return X
end


itransform(L::MinMaxTransform, X::AbstractMatrix{Float64}) = itransform!(L, zero(X), X)
