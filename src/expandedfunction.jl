

export ExpandedFunction

# ExpandedFunction decomposes a multi-dimensional function f:Rᴹ → R onto
# a basis of MultiFunctions ψ_α where c_α are scalar coefficients
# for each MultiFunction:
# f(x1, x2, ..., xNx) = ∑_α c_α ψ_α(x1, x2, ..., xNx)
# Nψ is the number of MultiFunctions used,
# Nx is the dimension of the input vector x

struct ExpandedFunction{m, Nx}
    B::MultiBasis{m, Nx}
    α::Array{Array{Int64,1}}
    c::Array{Float64,1}
    function ExpandedFunction(B::MultiBasis{m, Nx}, α::Array{Array{Int64,1}}, c::Array{Float64,1}) where {m, Nx}
            @assert size(α,1) == size(c,1) "The dimension of the basis functions don't
                                            match the number of coefficients"
            for i=1:Nx
                @assert size(α[i],i) == m "Size of one of the multi-index αi is wrong"
            end

        return new{m, Nx}(B, α, c)
    end
end

# # Naive approach of taking the full product
# function (f::ExpandedFunction{Nψ, m, Nx})(x::Array{T,1}) where {Nψ, m, Nx, T <: Real}
#     @assert Nx==size(x,1) "Wrong dimension of the input vector"
#     Nα =  size(α, 1)
#     out = 0.0
#     @inbounds for i=1:Nα
#         if f.c[i] != 0.0
#             out += f.c[i]*(f.B.B[f.α[i]])(x)
#         end
#     end
#     return out
# end
#
# # function
