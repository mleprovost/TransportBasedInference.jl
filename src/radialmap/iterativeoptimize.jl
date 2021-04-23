export iterative

# Efficient least-square solution for affine diagonal component
# using least-square formulation
# function iterative(C::SparseRadialMapComponent, ens::EnsembleState{Nx, Ne}, λ, δ) where {Nx, Ne}
#     @get C (Nx,p)
#     # Compute weights
#     ψ_off, ψ_mono, _ = weights(C, ens)
#
#     # no = size(ψ_off,1)
#     # nd = size(ψ_mono,1)
#     nx = size(ψ_off,1)+size(ψ_mono,1)+1
#     # nlog = size(dψ_mono,1)
#
#     # @assert nd==nlog "Error size of the diag and ∂k weights"
#     # @assert nx==no+nlog+1 "Error size of the weights"
#
#     # Cache for the solution
#     x = zeros(nx)
#
#     # Normalize monotone basis functions
#     μψ = mean(ψ_mono, dims=2)[1,1]
#     σψ = std(ψ_mono, dims=2, corrected=false)[1,1]
#     ψ_mono .-= μψ
#     ψ_mono ./= σψ
#
#     # Normalize off-diagonal covariates
#     σψ_off = std(ψ_off, dims = 2, corrected = false)[:,1]
#     ψ_off ./= σψ_off
#     # Solve for the off-diagonal coefficients and the constant
#     # We can use dψ_mono since it contains only one
#     ψ_at = hcat(ψ_off', ones(Ne))
#
#     # Solve least-square problem
#     # x[1:nx-1] = ψ_a' \ view(ψ_mono,1,:)
#     x0 = view(x,1:nx-1)
#     lsmr!(x0, ψ_at, view(ψ_mono,1,:), λ = λ, atol = 1e-9, btol = 1e-9)
#     # lsmr!(x0, S, view(ψ_mono,1,:), λ = λ)
#
#     @inbounds for i=1:nx-2
#     x[i] *= (-σψ/σψ_off[i])
#     end
#
#     x[nx-1] *= -σψ
#     x[nx-1] -= μψ
#
#     x[end] = √(Ne)/norm(ψ_at*(x[1:nx-1] .* vcat(σψ_off, 1.0)) .+ μψ + σψ*ψ_mono[1,:])
#     x[1:nx-1] .*= x[end]
#     return x
# end

function iterative(C::SparseRadialMapComponent, X, λ, δ)
    NxX, Ne = size(X)
    @get C (Nx,p)

    @assert NxX  == Nx "Wrong dimension of the ensemble matrix `X`"
    # Compute weights
    ψ_off, ψ_mono, _ = weights(C, X)

    # no = size(ψ_off,1)
    # nd = size(ψ_mono,1)
    nx = size(ψ_off,1)+size(ψ_mono,1)+1
    # nlog = size(dψ_mono,1)

    # @assert nd==nlog "Error size of the diag and ∂k weights"
    # @assert nx==no+nlog+1 "Error size of the weights"

    # Cache for the solution
    x = zeros(nx)

    # Normalize monotone basis functions
    μψ = mean(ψ_mono, dims=2)[1,1]
    σψ = std(ψ_mono, dims=2, corrected=false)[1,1]
    ψ_mono .-= μψ
    ψ_mono ./= σψ

    # Normalize off-diagonal covariates
    σψ_off = std(ψ_off, dims = 2, corrected = false)[:,1]
    ψ_off ./= σψ_off
    # Solve for the off-diagonal coefficients and the constant
    # We can use dψ_mono since it contains only one
    ψ_at = hcat(ψ_off', ones(Ne))

    # Solve least-square problem
    # x[1:nx-1] = ψ_a' \ view(ψ_mono,1,:)
    x0 = view(x,1:nx-1)
    lsmr!(x0, ψ_at, view(ψ_mono,1,:), λ = λ, atol = 1e-9, btol = 1e-9)
    # lsmr!(x0, S, view(ψ_mono,1,:), λ = λ)

    @inbounds for i=1:nx-2
    x[i] *= (-σψ/σψ_off[i])
    end

    x[nx-1] *= -σψ
    x[nx-1] -= μψ

    x[end] = √(Ne)/norm(ψ_at*(x[1:nx-1] .* vcat(σψ_off, 1.0)) .+ μψ + σψ*ψ_mono[1,:])
    x[1:nx-1] .*= x[end]
    return x
end
