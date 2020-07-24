
export   functionalf!,
         functionalg!,
         inverse!

function functionalf!(F, xk, cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx}
    function integrand!(v::Vector{Float64}, t::Float64)
        # repeated_grad_xk_basis(cacheF, R.f.f,  t*xk)
        cache .= repeated_grad_xk_basis(R.f.f,  t*xk)
        @avx @. v = (cache .* ψoff) *ˡ R.f.f.coeff
        evaluate!(v, R.g, v)
        # v .= R.g((repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff)*R.f.f.coeff)
    end

    quadgk!(integrand!, F, 0, 1; rtol = 1e-3)
    F .*= xk
    F .-= output
    nothing
end

functionalf!(cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx} =
    (F, xk) -> functionalf!(F, xk, cache, ψoff, output, R)

function functionalg!(J, xk, cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx}
    cache .= repeated_grad_xk_basis(R.f.f,  xk)
    @avx @. J.diag = (cache .* ψoff) *ˡ R.f.f.coeff
    evaluate!(J.diag, R.g, J.diag)
    nothing
end


functionalg!(cache, ψoff, output::Array{Float64,1}, R::IntegratedFunction{m, Nψ, Nx}) where {m, Nψ, Nx} =
    (J, xk) -> functionalg!(J, xk, cache, ψoff, output, R)


# The state is modified in-place
function inverse!(X::Array{Float64,2}, F, R::IntegratedFunction{m, Nψ, Nx}, S::Storage{m, Nψ, Nx}) where {m, Nψ, Nx}
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the sample X"

    cache  = zeros(Ne, Nψ)
    f0 = zeros(Ne)

    # Remove f(x_{1:k-1},0) from the output F
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψd0[i,j] * S.ψoff[i,j])*R.f.f.coeff[j]
        end
        F[i] -= f0i
    end

    df_inverse = OnceDifferentiable(functionalf!(cache, S.ψoff, F, R),
                                    functionalg!(cache, S.ψoff, F, R),
                                    f0, f0, Diagonal(f0))

    # Start minimization from the prior value
    result = nlsolve(df_inverse, X[end,:]; method = :newton);

    # Check convergence
    @assert converged(result) "Optmization hasn't converged"

    X[end,:] .= result.zero
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
