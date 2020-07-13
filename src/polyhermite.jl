export AbstractPhyHermite, AbstractProHermite

struct AbstractPhyHermite <: ParamFcn
    P::ImmutablePolynomial
    scaled::Bool
end

(P::AbstractPhyHermite)(x::T) where {T<:Real} = P.P(x)*exp(-x^2/2)


struct AbstractProHermite <: ParamFcn
    P::ImmutablePolynomial
    scaled::Bool
end

(P::AbstractProHermite)(x) = P.P(x)*exp(-x^2/4)
