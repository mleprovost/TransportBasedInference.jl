
import Statistics: mean, cov


export  InflationType, IdentityInflation, AdditiveInflation,
        MultiplicativeInflation, MultiAddInflation,
        exactn



"""
        exactn(N)

A function to create a 1D sample with exactly mean 0 and covariance 1
(The samples are no longer i.i.d but this can be usueful when the initialization of the problem is challenging.)
"""
function exactn(N)
    a = deepcopy(randn(N))
    return (a .- mean(a))./std(a)
end


"""
#     InflationType
#
# An abstract type for Inflation.
"""
abstract type InflationType end


"""
    IdentityInflation <: InflationType


An type to store identity inflation :

Define additive inflation: x <- x
"""
struct IdentityInflation <: InflationType
end


"""
        (A::IdentityInflation)(X)

Apply an `IdentityInflation` `A` on an ensemble matrix `X`, i.e. xⁱ -> xⁱ
"""
function (A::IdentityInflation)(X)
    nothing
end

"""
        AdditiveInflation <: InflationType

An type to store additive inflation :

Define additive inflation: x <- x + ϵ with ϵ a random vector
drawn from the distribution α

## Fields:
$(TYPEDFIELDS)

## Constructors
- `AdditiveInflation(Nx::Int64, α::ContinuousMultivariateDistribution)`
- `AdditiveInflation(Nx::Int64)`
- `AdditiveInflation(Nx::Int64, m::Array{Float64,1}, Σ::Union{Array{Float64,2}, Diagonal{Float64}})`
- `AdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Array{Float64,1})`
- `AdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Float64)`
- `AdditiveInflation(m::Array{Float64,1}, σ::Float64)`
"""

struct AdditiveInflation <: InflationType
    "Dimension of the state vector"
    Nx::Int64

    "Mean of the additive inflation"
    m::Array{Float64,1}

    "Covariance of the additive inflation"
    Σ::Union{Array{Float64,2}, Diagonal{Float64}}

    "Square-root of the covariance matrix"
    σ::Union{Array{Float64,2}, Diagonal{Float64}}
end

# Some convenient constructors for multivariate Gaussian distributions
# By default, the distribution of the additive inflation α is a multivariate
 # normal distribution with zero mean and identity as the covariance matrix
@inline AdditiveInflation(Nx::Int64) = AdditiveInflation(Nx, zeros(Nx),  Diagonal(ones(Nx)), Diagonal(ones(Nx)))

function AdditiveInflation(Nx::Int64, m::Array{Float64,1}, Σ::Union{Array{Float64,2}, Diagonal{Float64}})
@assert Nx==size(m,1) "Error dimension of the mean"
@assert Nx==size(Σ,1)==size(Σ,2) "Error dimension of the covariance matrix"

return AdditiveInflation(Nx, m, Σ, sqrt(Σ))

end

function AdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Array{Float64,1})
@assert Nx==size(m,1) "Error dimension of the mean"
@assert Nx==size(σ,1) "Error dimension of the std vector"

return AdditiveInflation(Nx, m, Diagonal(σ .^2), Diagonal(σ))

end

function AdditiveInflation(Nx::Int64, m::Array{Float64,1}, σ::Float64)
@assert Nx==size(m,1) "Error dimension of the mean"

return AdditiveInflation(Nx, m, Diagonal(σ^2*ones(Nx)), Diagonal(σ*ones(Nx)))

end

function AdditiveInflation(m::Array{Float64,1}, σ::Float64)
    Nx = size(m,1)
    return AdditiveInflation(Nx, m, Diagonal(σ^2*ones(Nx)), Diagonal(σ*ones(Nx)))
end

Base.size(A::AdditiveInflation)  = A.Nx
mean(A::AdditiveInflation) = A.m
cov(A::AdditiveInflation) = A.Σ

"""
        (A::AdditiveInflation)(X, start::Int64, final::Int64)


Apply the additive inflation `A` to the lines `start` to `final` of an ensemble matrix `X`,
i.e. xⁱ -> xⁱ + ϵⁱ with ϵⁱ ∼ `A.α`.
"""
function (A::AdditiveInflation)(X, start::Int64, final::Int64; laplace::Bool=false)
    Ne = size(X)[end]
    @assert A.Nx == final - start + 1 "final-start + 1 doesn't match the length of the additive noise"
    # @show X[start:final, 1]
    if laplace == false
        @inbounds for i=1:Ne
            col = view(X, start:final, i)
            col .+= A.m + A.σ*randn(A.Nx)
        end
    else
        @inbounds for i=1:Ne
            col = view(X, start:final, i)
            col .+= A.m + sqrt(2.0)*A.σ*randn(Laplace(), A.Nx)
        end
    end
    # @show X[start:final, 1]
end

"""
        (A::AdditiveInflation)(X)


Apply the additive inflation `A` to an ensemble matrix `X`,
i.e. xⁱ -> xⁱ + ϵⁱ with ϵⁱ ∼ `A.α`.
"""
(A::AdditiveInflation)(X; laplace::Bool=false) = A(X, 1, size(X)[1]; laplace = laplace)

"""
        (A::AdditiveInflation)(x::Array{Float64,1})

Apply the additive inflation `A` to the vector `x`,
i.e. x -> x + ϵ with ϵ ∼ `A.α`.
"""
function (A::AdditiveInflation)(x::Array{Float64,1})
    x .+= A.m + A.σ*randn(A.Nx)
    return x
end


"""
    MultiplicativeInflation <: InflationType

An type to store multiplicative inflation :

xⁱ -> xⁱ + β*(xⁱ - x̄) with β a scalar

# Fields:
- 'β' : multiplicative inflation factor
"""
struct MultiplicativeInflation <: InflationType
    "Multiplicative inflation factor β"
    β::Real
end

"""
        (A::MultiplicativeInflation)(X, start::Int64, final::Int64)


Apply the multiplicative inflation `A` to the lines `start` to `final` of an ensemble matrix `X`,
i.e. xⁱ -> x̄ + β*(xⁱ - x̄)  with β scalar, usually ∼ 1.0.
"""
function (A::MultiplicativeInflation)(X, start::Int64, final::Int64)
    Ne = size(X,2)
    X̂ = copy(mean(view(X, start:final,:), dims = 2)[:,1])
    @inbounds for i=1:Ne
        col = view(X, start:final, i)
        col .= (1.0-A.β)*X̂ + A.β*col
    end
end


"""
        (A::MultiplicativeInflation)(X)


Apply the multiplicative inflation `A` to an ensemble matrix `X`,
i.e. xⁱ -> x̄ + β*(xⁱ - x̄)  with β scalar, usually ∼ 1.0.
"""
(A::MultiplicativeInflation)(X) = A(X, 1, size(X, 1))


"""
    MultiAddInflation <: InflationType


An type to store multiplico-additive inflation :

Define multiplico-additive inflation: xⁱ -> x̄ + β*(xⁱ - x̄)  + ϵⁱ with ϵⁱ ∼ α and β a scalar

## Fields:
- `Nx` : dimension of the vector
- 'β' : Multiplicative inflation factor
- 'α' : Distribution of the additive inflation

## Constructors:
- `MultiAddInflation(Nx::Int64, β::Real, α::ContinuousMultivariateDistribution)`
- `MultiAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, Σ)`
- `MultiAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, σ::Array{Float64,1})`
- `MultiAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, σ::Float64)`
"""
struct MultiAddInflation <: InflationType
    "Dimension of the state vector"
    Nx::Int64

    "Multiplicative inflation factor β"
    β::Real

    "Mean of the additive inflation"
    m::Array{Float64,1}

    "Covariance of the additive inflation"
    Σ::Union{Array{Float64,2}, Diagonal{Float64}}

    "Square-root of the covariance matrix"
    σ::Union{Array{Float64,2}, Diagonal{Float64}}
end

# Some convenient constructors for multivariate Gaussian additive distributions
# By default, for a Multiplico-additive inflation, the multiplicative inflation
# factor β is set to 1.0, and  α is a  multivariate
# normal distribution with zero mean and identity as the covariance matrix
function MultiAddInflation(Nx::Int)
    return MultiAddInflation(Nx, 1.0, zeros(Nx), Diagonal(ones(Nx)), Diagonal(ones(Nx)))
end


function MultiAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, Σ)
    @assert β>0.0 "The multiplicative inflation must be >0.0"
    @assert Nx==size(m,1) "Error dimension of the mean"
    @assert Nx==size(Σ,1)==size(Σ,2) "Error dimension of the covariance matrix"

    return MultiAddInflation(Nx, β, m, Σ, sqrt(Σ))
end

function MultiAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, σ::Array{Float64,1})
    @assert β>0.0 "The multiplicative inflation must be >0.0"
    @assert Nx==size(m,1) "Error dimension of the mean"
    @assert Nx==size(σ,1) "Error dimension of the std vector"

    return MultiAddInflation(Nx, β, m, Diagonal(σ .^2), Diagonal(σ))
end

function MultiAddInflation(Nx::Int64, β::Float64, m::Array{Float64,1}, σ::Float64)
    @assert β>0.0 "The multiplicative inflation must be >0.0"
    @assert Nx==size(m,1) "Error dimension of the mean"

    return MultiAddInflation(Nx, β, m, Diagonal(σ^2*ones(Nx)), Diagonal(σ*ones(Nx)))
end

"""
    size(A::MultiAddInflation)

Return the dimension of the additive inflation of `A`.
"""
Base.size(A::MultiAddInflation) = A.Nx
mean(A::MultiAddInflation) = A.m
cov(A::MultiAddInflation)  = A.Σ

"""
        (A::MultiAddInflation)(X, start::Int64, final::Int64)


Apply the multiplicat inflation `A` to the lines `start` to `final` of an ensemble matrix `X`,
i.e. xⁱ -> x̄ + β*(xⁱ - x̄)  + ϵⁱ with ϵⁱ ∼ `A.α` and β a scalar, usually ∼ 1.0.
"""
function (A::MultiAddInflation)(X, start::Int64, final::Int64)
    @assert A.Nx = final - start +1 "Dimension does not match"
    Ne = size(X,2)
    X̂ = copy(mean(view(X, start:final,:), dims = 2)[:,1])
    @inbounds for i=1:Ne
        col = view(X, start:final, i)
        col .= (1.0-A.β)*X̂ + A.β*col + A.m + A.σ*randn(A.Nx)
    end
end

"""
        (A::MultiAddInflation)(X, start::Int64, final::Int64)


Apply the multiplico-additive inflation `A` to the ensemble matrix `X`,
i.e. xⁱ -> x̄ + β*(xⁱ - x̄)  + ϵⁱ with ϵⁱ ∼ `A.α` and β a scalar, usually ∼ 1.0.
"""
(A::MultiAddInflation)(X) = A(X, 1, size(X,1))
