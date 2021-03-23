
import Statistics: mean, cov


export  InflationType, IdentityInflation, AdditiveInflation,
        MultiplicativeInflation, MultiAddInflation,
        exactn



# """
# exactn
#
# A function to create a 1D sample with exactly mean 0 and covariance 1
# (The samples are no longer i.i.d but this can be usueful when the initialization of the problem is challenging.)
#
# """
function exactn(N)
    a = deepcopy(rand(N))
    return (a .- mean(a))./std(a)
end


# """
#     InflationType
#
# An abstract type for Inflation.
# """
abstract type InflationType end

# """
#     IdentityInflation
#
#
# An type to store identity inflation :
#
# Define additive inflation: x <- x
#
# """
struct IdentityInflation <: InflationType

end


# " Define action of IdentityInflation on an EnsembleState : x <- x  "
function (A::IdentityInflation)(X)
    nothing
end

# """
#     AdditiveInflation
#
# An type to store additive inflation :
#
# Define additive inflation: x <- x + α with α a N-dimensional vector
# drawn from a random distribution
#
# # Fields:
# - 'α' : Distribution of the additive inflation
#
# """

# We only consider Gaussian distribution for the additive noise
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

# """
#     size(A::AdditiveInflation) -> Tuple{Int...}
#
# Return the dimension of the additive inflation
#
# """
Base.size(A::AdditiveInflation)  = A.Nx
mean(A::AdditiveInflation) = A.m
cov(A::AdditiveInflation) = A.Σ



# """
# Define action of AdditiveInflation on an array X : x <- x + α
# """

# For 2D array of stacked measurement and state components
function (A::AdditiveInflation)(X, start::Int64, final::Int64; laplace::Bool=false)
    Ne = size(X,2)
    @assert A.Nx == final - start + 1 "final-start + 1 doesn't match the length of the additive noise"
    # @show X[start:final, 1]
    if laplace == false
        @inbounds for i=1:Ne
            col = view(X, start:final, i)
            col .+= A.m + A.σ*rand(A.Nx)
        end
    else
        @inbounds for i=1:Ne
            col = view(X, start:final, i)
            col .+= A.m + sqrt(2.0)*A.σ*rand(Laplace(), A.Nx)
        end
    end
    # @show X[start:final, 1]
end

(A::AdditiveInflation)(X; laplace::Bool=false) = A(X, 1, size(X,1); laplace = laplace)

# Only for 1D array
function (A::AdditiveInflation)(x::Array{Float64,1})
    x .+= A.m + A.σ*rand(A.Nx)
    return x
end


# """
#     MultiplicativeInflation
#
# An type to store multiplicative inflation :
#
# Define multiplicative inflation: x <- x + β*(x - x̂) with β a scalar
#
# # Fields:
# - 'β' : multiplicative inflation factor
#
# """
struct MultiplicativeInflation <: InflationType
    "Multiplicative inflation factor β"
    β::Real
end


# """
# Define action of MultiplicativeInflation : x <- x̂ + β*(x - x̂)
# """
# function (A::MultiplicativeInflation)(ens::EnsembleState{Nx, Ne}) where {Nx, Ne}
#     Ŝ = deepcopy(mean(ens))
#     @inbounds for i=1:Ne
#         ens.S[:,i] .= Ŝ + A.β*(ens.S[:,i]-Ŝ)
#     end
# end


# For 2D array of stacked measurement and state components
function (A::MultiplicativeInflation)(X, start::Int64, final::Int64)
    Ne = size(X,2)
    X̂ = copy(mean(view(X, start:final,:), dims = 2)[:,1])
    @inbounds for i=1:Ne
        col = view(X, start:final, i)
        col .= (1.0-A.β)*X̂ + A.β*col
    end
end

(A::MultiplicativeInflation)(X) = A(X, 1, size(X, 1))


# """
#     MultiAddInflation
#
#
# An type to store multiplico-additive inflation :
#
# Define multiplico-additive inflation: x̃⁻ <- x̂⁻ + β*(x̃⁻ - x̂⁻)  + α with β a scalar
#
# # Fields:
# - 'β' : Multiplicative inflation factor
# - 'α' : Distribution of the additive inflation
#
# """
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

# """
#     size(A::MultiAddInflation) -> Tuple{Int...}
#
# Return the dimension of the additive inflation
#
# """
Base.size(A::MultiAddInflation) = A.Nx

mean(A::MultiAddInflation) = A.m
cov(A::MultiAddInflation)  = A.Σ

# """
# Define action of Multiplicative and Additive Inflation : x <- x̂ + β*(x - x̂)
# """
# function (A::MultiAddInflation)(ens::EnsembleState{Nx, Ne}) where {Nx, Ne}
#     Ŝ = deepcopy(mean(ens))
#     @inbounds for i=1:Ne
#         s = view(ens.S,:,i)
#         # s = deepcopy(ens.S[:,i])
#         rmul!(s, A.β)
#         s .+= rmul!(Ŝ, 1.0-A.β) + A.m + A.σ*randn(A.Nx)
#         # ens.S[:,i] .= deepcopy(s)
#     end
# end

# For 2D array of stacked measurement and state components
function (A::MultiAddInflation)(X, start::Int64, final::Int64)
    @assert A.Nx = final - start +1 "Dimension does not match"
    Ne = size(X,2)
    X̂ = copy(mean(view(X, start:final,:), dims = 2)[:,1])
    @inbounds for i=1:Ne
        col = view(X, start:final, i)
        col .= (1.0-A.β)*X̂ + A.β*col + A.m + A.σ*randn(A.Nx)
    end
end

(A::MultiAddInflation)(X) = A(X, 1, size(X,1))
