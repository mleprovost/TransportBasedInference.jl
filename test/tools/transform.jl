@testset "Linear rescaling of multi-dimensional sample" begin

    # With diagonal rescaling
    Nx = 1
    Ne = 100

    X = 1.0 .+ 5.0*randn(Nx, Ne)

    L = LinearTransform(X; diag = true)

    AdaptiveTransportMap.transform!(L, X)

    @test norm(mean(X; dims = 2)[:,1])<1e-10
    @test norm(cov(X') .- 1.0)<1e-10


    Nx = 10
    Ne = 100

    X = randn(Nx) .+ 5.0* randn(Nx, Ne) .* randn(Nx, Ne)


    L = LinearTransform(X; diag = true)

    AdaptiveTransportMap.transform!(L, X)

    @test norm(mean(X; dims = 2)[:,1])<1e-10
    @test norm(diag(cov(X')) .- 1.0)<1e-10


    Nx = 10
    Ne = 500
    ens = EnsembleState(Nx, Ne)

    ens.S .= randn(Nx).+ randn(Nx, Ne) .* randn(Nx, Ne)

    L = LinearTransform(ens.S; diag = false)

    AdaptiveTransportMap.transform!(L, ens.S)

    @test norm(mean(ens))<1e-10
    @test norm(cov(ens)  - I)<1e-10

end
