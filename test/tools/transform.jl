@testset "Linear rescaling of multi-dimensional sample" begin

    # Diagonal rescaling
    Nx = 1
    Ne = 500
    ens = EnsembleState(Nx, Ne)

    ens.S .= randn(Nx).+ randn(Nx, Ne) .* randn(Nx, Ne)
    X = deepcopy(ens.S)
    L = LinearTransform(X; diag = true)

    AdaptiveTransportMap.transform!(L, X);

    @test norm(mean(X;dims = 2))<1e-10
    @test norm(std(X; dims = 2)[:,1] - ones(Nx))<1e-10

    AdaptiveTransportMap.itransform!(L, X)

    @test norm(mean(X;dims = 2) - mean(ens))<1e-10
    @test norm(cov(X')  - cov(ens))<1e-10

    # Diagonal rescaling
    Nx = 100
    Ne = 500
    ens = EnsembleState(Nx, Ne)

    ens.S .= randn(Nx).+ randn(Nx, Ne) .* randn(Nx, Ne)
    X = deepcopy(ens.S)
    L = LinearTransform(X; diag = true)

    AdaptiveTransportMap.transform!(L, X);

    @test norm(mean(X;dims = 2))<1e-10
    @test norm(std(X; dims = 2)[:,1] - ones(Nx))<1e-10

    AdaptiveTransportMap.itransform!(L, X)

    @test norm(mean(X;dims = 2) - mean(ens))<1e-10
    @test norm(cov(X')  - cov(ens))<1e-10

    # Dense rescaling
    Nx = 100
    Ne = 500
    ens = EnsembleState(Nx, Ne)

    ens.S .= randn(Nx).+ randn(Nx, Ne) .* randn(Nx, Ne)
    X = deepcopy(ens.S)
    L = LinearTransform(X; diag = false)

    AdaptiveTransportMap.transform!(L, X);

    @test norm(mean(X;dims = 2))<1e-10
    @test norm(cov(X')  - I)<1e-10

    AdaptiveTransportMap.itransform!(L, X)

    @test norm(mean(X;dims = 2) - mean(ens))<1e-10
    @test norm(cov(X')  - cov(ens))<1e-10



end
