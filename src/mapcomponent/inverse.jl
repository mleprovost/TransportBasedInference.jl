
export   functionalf!,
         functionalg!,
         inverse!

# function functionalf!(F, xk, cache, cache_vander, ψoff, output::Array{Float64,1}, R::IntegratedFunction)

function functionalf!(F, xk, cache, cache_vander, ψoff, output, R::IntegratedFunction)
    function integrand!(v::Vector{Float64}, t::Float64)
        repeated_grad_xk_basis!(cache, cache_vander, R.f.f,  t*xk)
        # cache .= repeated_grad_xk_basis(R.f.f,  t*xk)
        @avx @. v = (cache .* ψoff) *ˡ R.f.f.coeff
        evaluate!(v, R.g, v)
        # v .= R.g((repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff)*R.f.f.coeff)
    end

    quadgk!(integrand!, F, 0.0, 1.0; rtol = 1e-3)
    F .*= xk
    F .-= output
    nothing
end

# functionalf!(cache, cache_vander, ψoff, output::Array{Float64,1}, R::IntegratedFunction) =

functionalf!(cache, cache_vander, ψoff, output, R::IntegratedFunction) =
    (F, xk) -> functionalf!(F, xk, cache, cache_vander, ψoff, output, R)

function functionalg!(J, xk, cache, cache_vander, ψoff, output, R::IntegratedFunction)
    # cache .= repeated_grad_xk_basis(R.f.f,  xk)
    repeated_grad_xk_basis!(cache, cache_vander, R.f.f,  xk)
    @avx @. J.diag = (cache .* ψoff) *ˡ R.f.f.coeff
    evaluate!(J.diag, R.g, J.diag)
    nothing
end


functionalg!(cache, cache_vander, ψoff, output, R::IntegratedFunction) =
    (J, xk) -> functionalg!(J, xk, cache, cache_vander, ψoff, output, R)


# The state is modified in-place
# function inverse!(X::Array{Float64,2}, F, R::IntegratedFunction, S::Storage)
function inverse!(X, F, R::IntegratedFunction, S::Storage)
    Nψ = R.Nψ
    Nx = R.Nx
    NxX, Ne = size(X)

    @assert NxX == R.Nx "Wrong dimension of the sample X"

    cache  = zeros(Ne, Nψ)
    cache_vander = zeros(Ne, maximum(R.f.f.idx[:,Nx])+1)
    f0 = zeros(Ne)

    # Remove f(x_{1:k-1},0) from the output F
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψoffψd0[i,j])*R.f.f.coeff[j]
        end
        F[i] -= f0i
    end

    df_inverse = OnceDifferentiable(functionalf!(cache, cache_vander, S.ψoff, F, R),
                                    functionalg!(cache, cache_vander, S.ψoff, F, R),
                                    f0, f0, Diagonal(f0))

    # Start minimization from the prior value
    result = nlsolve(df_inverse, X[end,:]; method = :newton);

    # Check convergence
    @assert converged(result) "Optimization hasn't converged"

    X[end,:] .= result.zero
end


inverse!(X, F, C::MapComponent, S::Storage) = inverse!(X, F, C.I, S)


function inverse!(X::Array{Float64,2}, F, L::LinMapComponent, S::Storage)
    # Pay attention that S is computed in the renormalized space for improve performance !!!
    transform!(L.L, X)
    inverse!(X, F, L.C, S)
    itransform!(L.L, X)
end






# function functionalfg!(F, J, xk, cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx}
#
#     if !(F == nothing)
#         function integrand!(v::Vector{Float64}, t::Float64)
#             v .= R.g((repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff)*R.f.f.coeff)
#         end
#         F .= xk .* quadgk!(integrand!, cache, 0, 1)[1]
#         F .-= output
#     end
#
#     if !(J == nothing)
#         J .= Diagonal(R.g((repeated_grad_xk_basis(R.f.f,  xk) .* ψoff)*R.f.f.coeff))
#     end
# end
#
#
# functionalfg!(cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx} =
#     (F, J, xk) -> functionalfg!(F, J, xk, cache, ψoff, output, R)
