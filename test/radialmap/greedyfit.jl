

@testset "Verify Jacobian of off-diagonal entries" begin
    Nx = 3
    Ne = 100

    X = randn(Nx, Ne) .* randn(Nx, Ne)

    p = 2
    order = p*ones(Int64, Nx)
    order[end] = p

    λ = 0.1
    δ = 0.0
    γ = 2.0

    C = SparseRadialMapComponent(Nx, order)

    center_std!(C, X; γ = γ)



end
