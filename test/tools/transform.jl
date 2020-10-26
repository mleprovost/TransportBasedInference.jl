@testset "Linear rescaling of multi-dimensional sample" begin

    # Diagonal rescaling
    Nx = 1
    Ne = 500

    X = randn(Nx).+ randn(Nx, Ne) .* randn(Nx, Ne)
    X0 = deepcopy(X)
    L = LinearTransform(X; diag = true)

    AdaptiveTransportMap.transform!(L, X);

    @test norm(mean(X;dims = 2))<1e-10
    @test norm(std(X; dims = 2)[:,1] - ones(Nx))<1e-10

    AdaptiveTransportMap.itransform!(L, X)

    @test norm(mean(X;dims = 2) - mean(X0;dims = 2))<1e-10
    @test norm(cov(X')  - cov(X0'))<1e-10

    # Diagonal rescaling
    Nx = 100
    Ne = 500
    X = zeros(Nx, Ne)

    X .= randn(Nx).+ randn(Nx, Ne) .* randn(Nx, Ne)
    X0 = deepcopy(X)
    L = LinearTransform(X; diag = true)

    AdaptiveTransportMap.transform!(L, X);

    @test norm(mean(X;dims = 2))<1e-10
    @test norm(std(X; dims = 2)[:,1] - ones(Nx))<1e-10

    AdaptiveTransportMap.itransform!(L, X)

    @test norm(mean(X; dims = 2) - mean(X0; dims = 2))<1e-10
    @test norm(cov(X')  - cov(X0'))<1e-10

    # Dense rescaling
    Nx = 100
    Ne = 500
    X = zeros(Nx, Ne)

    X .= randn(Nx).+ randn(Nx, Ne) .* randn(Nx, Ne)
    X0 = deepcopy(X)
    L = LinearTransform(X; diag = false)

    AdaptiveTransportMap.transform!(L, X);

    @test norm(mean(X;dims = 2))<1e-10
    @test norm(cov(X')  - I)<1e-10

    AdaptiveTransportMap.itransform!(L, X)

    @test norm(mean(X;dims = 2) - mean(X0; dims = 2))<1e-10
    @test norm(cov(X')  - cov(X0'))<1e-10
end


@testset "MinMaxTransform in-place rescaling of multi-dimensional sample" begin
    # Nx = 1
    Nx = 1
    Ne = 500
    X = randn(Nx) .+ randn(Nx, Ne) .* randn(Nx, Ne)
    X0 = deepcopy(X)

    L = MinMaxTransform(X)

    transform!(L, X)

    for i=1:Nx
        @test abs(minimum(X[i,:]) - (-1.0))<1e-10
        @test abs(maximum(X[i,:]) - ( 1.0))<1e-10
    end

    itransform!(L, X)

    @test norm(X - X0)<1e-10

    # Nx = 5
    Nx = 5
    Ne = 500
    X = randn(Nx) .+ randn(Nx, Ne) .* randn(Nx, Ne)
    X0 = deepcopy(X)

    L = MinMaxTransform(X)

    transform!(L, X)

    for i=1:Nx
        @test abs(minimum(X[i,:]) - (-1.0))<1e-10
        @test abs(maximum(X[i,:]) - ( 1.0))<1e-10
    end

    itransform!(L, X)

    @test norm(X - X0)<1e-10
end

@testset "MinMaxTransform out of place rescaling of multi-dimensional sample" begin
    # Nx = 1
    Nx = 1
    Ne = 500
    Xin = randn(Nx) .+ randn(Nx, Ne) .* randn(Nx, Ne)
    Xout = zero(Xin)
    Xinverse = zero(Xin)
    X0 = deepcopy(Xin)

    L = MinMaxTransform(Xin)

    transform!(L, Xout, Xin)

    for i=1:Nx
        @test abs(minimum(Xout[i,:]) - (-1.0))<1e-10
        @test abs(maximum(Xout[i,:]) - ( 1.0))<1e-10
    end

    itransform!(L, Xinverse, Xout)

    @test norm(Xinverse - X0)<1e-10
    @test norm(Xin - X0)<1e-10

    # Nx = 5
    Nx = 5
    Ne = 500
    Xin = randn(Nx) .+ randn(Nx, Ne) .* randn(Nx, Ne)
    Xout = zero(Xin)
    Xinverse = zero(Xin)
    X0 = deepcopy(Xin)

    L = MinMaxTransform(Xin)

    transform!(L, Xout, Xin)

    for i=1:Nx
        @test abs(minimum(Xout[i,:]) - (-1.0))<1e-10
        @test abs(maximum(Xout[i,:]) - ( 1.0))<1e-10
    end

    itransform!(L, Xinverse, Xout)

    @test norm(Xinverse - X0)<1e-10
    @test norm(Xin - X0)<1e-10
end
