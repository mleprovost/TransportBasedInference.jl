
export   functionalf!,
         functionalg!,
         functionalg1D!,
         inverse!,
         inversehybrid!,
         inverse1D!

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

# In this version, the Jacobian is stored as a vector (it is diagonal in this case).
function functionalg1D!(J::AbstractVector{Float64}, xk, cache, cache_vander, ψoff, output, R::IntegratedFunction)
    # cache .= repeated_grad_xk_basis(R.f.f,  xk)
    repeated_grad_xk_basis!(cache, cache_vander, R.f.f,  xk)
    @avx @. J = (cache .* ψoff) *ˡ R.f.f.coeff
    evaluate!(J, R.g, J)
    nothing
end

functionalg1D!(cache, cache_vander, ψoff, output, R::IntegratedFunction) =
    (J, xk) -> functionalg1D!(J, xk, cache, cache_vander, ψoff, output, R)

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
    result = nlsolve(df_inverse, X[end,:]; method = :newton, linesearch = LineSearches.MoreThuente());

    # Check convergence
    # @show converged(result)
    if converged(result) == true
    # @assert converged(result) "Optimization hasn't converged"
        X[end,:] .= result.zero
    end
end


inverse!(X, F, C::MapComponent, S::Storage) = inverse!(X, F, C.I, S)


function inverse!(X::Array{Float64,2}, F, L::LinMapComponent, S::Storage)
    # Pay attention that S is computed in the renormalized space for improve performance !!!
    transform!(L.L, X)
    inverse!(X, F, L.C, S)
    itransform!(L.L, X)
end

#Expands the range of valued searched geometrically until a root is bracketed
# Julia's adaptation of zbrac Numerical Recipes in Fortran 77 p.345
# function bracket(f, a, b)
#     F = 1.6
#     Niter = 100
#     outa = deepcopy(a)
#     outb = deepcopy(b)
#     counter = 0
#     # Root is not bracketed
#     while f(outa)*f(outb)>0 && counter<Niter
#         Δ = F*deepcopy(outa-outb)
#         if abs(f(outa))<abs(f(outb))
#         outa += Δ
#         else
#         outb += Δ
#         end
#         counter += 1
#     end
#     return outa, outb
#     @assert counter<=(Niter-1) "Maximal number of iterations reached"
# end

# function repeated_grad_xk_basis1D!(out, cache, f::ExpandedFunction, x, idx::Array{Int64,2})
#     # Compute the k=th order deriviative of an expanded function along the direction grad_dim
#     Nx = f.Nx
#     # ∂ᵏf/∂x_{grad_dim} = ψ
#     k = 1
#     grad_dim = Nx
#     dims = Nx
#
#     midxj = idx[:, Nx]
#     maxj = maximum(midxj)
#     #   Compute the kth derivative along grad_dim
#     vander!(cache, f.B.B, maxj, k, x)
#     Nψreduced = size(idx, 1)
#     @avx for l = 1:Nψreduced
#             out[l] = cache[midxj[l] + 1]
#     end
#     return out
# end

# function functionalf1D!(F, xk, cache, cache_vander, ψoff, output, R::IntegratedFunction)
#     function integrand!(v::Vector{Float64}, t::Float64)
#         repeated_grad_xk_basis!(cache, cache_vander, R.f.f,  t*xk)
#         # cache .= repeated_grad_xk_basis(R.f.f,  t*xk)
#         @avx @. v = (cache .* ψoff) *ˡ R.f.f.coeff
#         evaluate!(v, R.g, v)
#         # v .= R.g((repeated_grad_xk_basis(R.f.f,  t*xk) .* ψoff)*R.f.f.coeff)
#     end
#
#     quadgk!(integrand!, F, 0.0, 1.0; rtol = 1e-3)
#     F .*= xk
#     F .-= output
#     nothing
# end

#
# function inverse1D!(x, R::IntegratedFunction, ψoff::AbstractVector{Float64})
#
#
#
#
#
#
#
#
#
#
#
# end
#
# # Bissection + Newton-Raphson because Newton-Raphsoin has poor global properties
# # but good provided that we are sufficiently close
# function inversehybrid!(X, F, R::IntegratedFunction, S::Storage; xtol::Float64 = 0.0, iterations::Int64 = 1000, )
#     Nψ = R.Nψ
#     Nx = R.Nx
#     NxX, Ne = size(X)
#
#     @assert NxX == R.Nx "Wrong dimension of the sample X"
#
#     cache  = zeros(Ne, Nψ)
#     cache_vander = zeros(Ne, maximum(R.f.f.idx[:,Nx])+1)
#     f0 = zeros(Ne)
#
#     # Remove f(x_{1:k-1},0) from the output F
#     @avx for i=1:Ne
#         f0i = zero(Float64)
#         for j=1:Nψ
#             f0i += (S.ψoffψd0[i,j])*R.f.f.coeff[j]
#         end
#         F[i] -= f0i
#     end
#
#     # ensemble members to invert
#     idxinvert = collect(1:Ne)
#
#     # lower and upper brackets of the ensemble members
#     brackets = zeros(2, Nx)
#
#     cachexold = zeros(Ne)
#     copy!(cachexold, view(X,:,end))
#     σ = std(cachexold)
#     # Compute the brackets
#     @inbounds for i=1:Ne
#         x₋ = cachexold[i] - 10*σ
#         x₊ = cachexold[i] + 10*σ
#         x₋, x₊ =
#         brackets[1,i] = x₋
#         brackets[2,i] = x₊
#     end
#
#
#     Jcache = Diagonal(f0)
#
#     # Iterate
#     # while
#     #
#     #
#     # end
#
#     functionalg!(J, xk, cache, cache_vander, ψoff, output, R)
#
#     # Xcache = X[:,end] - function
#
#
#
#
#
#     df_inverse = OnceDifferentiable(functionalf!(cache, cache_vander, S.ψoff, F, R),
#                                     functionalg!(cache, cache_vander, S.ψoff, F, R),
#                                     f0, f0, Diagonal(f0))
#
#     # Start minimization from the prior value
#     result = nlsolve(df_inverse, X[end,:]; method = :newton, linesearch = LineSearches.HagerZhang());
#
#     # Check convergence
#     @show converged(result)
#     # @show "hello 2.0"
#     if converged(result) == true
#     # @assert converged(result) "Optimization hasn't converged"
#         X[end,:] .= result.zero
#     end
# end






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
