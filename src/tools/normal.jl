# Define the Hessian of the logpdf of a Normal distribution
# The logpdf and gradient of the logpdf are already defined in Distributions.jl

export hesslogpdf

# gradlogpdf(Normal(), 2.0)
# logpdf(Normal(1.0, 2.0),randn(10))

hesslogpdf(N::Normal{Float64}, x::T) where {T<:Real} = -1/N.σ^2
