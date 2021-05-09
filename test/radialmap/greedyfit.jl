

@testset "Verify Jacobian of off-diagonal entries" begin
    Nx = 4
    Ne = 100

    X = randn(Nx, Ne) .* randn(Nx, Ne)
    λ = 0.0
    δ = 0.0
    γ = 2.0

    for p in [0; 1; 2; 3]
        if p == 0
            coeff_off = randn((Nx-1)*(p+1))
            coeff_diag = rand(p+2)
        else
            coeff_off = randn((Nx-1)*(p+1))
            coeff_diag = rand(p+3)
        end

        coeff = vcat(coeff_off, coeff_diag)
        dJ = zero(coeff_off)
        order = p*ones(Int64, Nx)
        Cfull = SparseRadialMapComponent(Nx, order)
        center_std!(Cfull, sort(X; dims = 2); γ = γ)
        modify_a!(Cfull, coeff)
        ψ_off, ψ_diag, ψ_∂k = compute_weights(Cfull, X)

        J(x) = (1/(2*Ne))*norm((ψ_off*x) + ψ_diag*coeff_diag[2:end] .+ coeff_diag[1])^2

        dJautodiff = ForwardDiff.gradient(x-> J(x), coeff_off)

        rhs = zeros(Ne)
        cache = zeros(Ne)
        mul!(rhs, ψ_diag, coeff_diag[2:end])
        rhs .+= coeff_diag[1]
        rmul!(rhs, -1.0)
        gradient_off!(dJ, cache, ψ_off, coeff_off, rhs, Ne)

        @test isapprox(dJ, dJautodiff, atol = 1e-10)
    end
end

@testset "Test greedyfit for a fixed number of family of features" begin

    for Nx=1:5
        Ne = 100

        X = randn(Nx, Ne) .* randn(Nx, Ne)

        λ = 0.0
        δ = 1e-8
        γ = 2.0

        for pdiag=0:3
            for poff=0:3
                for k=0:Nx-1
                    Cgreedy, ϵgreedy = greedyfit(Nx, pdiag, poff, deepcopy(X), k, λ, δ, γ)
                    C = SparseRadialMapComponent(Nx, Cgreedy.p)
                    center_std!(C, sort(deepcopy(X); dims = 2); γ = γ)
                    x_opt = optimize(C, X, λ, δ)
                    modify_a!(C, x_opt)
                    @test isapprox(ϵgreedy[end], negative_likelihood(C, X), atol = 1e-10)
                    @test norm(Cgreedy.a - C.a)<1e-10
                end
            end
        end

        for poff=0:3
            for pdiag=0:3
                for k=0:Nx-1
                    Cgreedy, ϵgreedy = greedyfit(Nx, pdiag, poff, deepcopy(X), k, λ, δ, γ)
                    C = SparseRadialMapComponent(Nx, Cgreedy.p)
                    center_std!(C, sort(deepcopy(X); dims = 2); γ = γ)
                    x_opt = optimize(C, X, λ, δ)
                    modify_a!(C, x_opt)
                    @test isapprox(ϵgreedy[end], negative_likelihood(C, X), atol = 1e-10)
                    @test norm(Cgreedy.a - C.a)<1e-10
                end
            end
        end
    end
end
