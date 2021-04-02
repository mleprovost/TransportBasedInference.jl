@testset "Identity inflation" begin
    A = IdentityInflation()

    Nx = 10
    Ne = 20
    X = zeros(Nx, Ne)
    X .= randn(Nx, Ne)

    Xinflation = deepcopy(X)
    A(X)

    @test norm(X - Xinflation)<1e-14
end

@testset "Additive inflation I" begin
    Ny = 10
    Nx= 10
    Ne = 20
    A = AdditiveInflation(Nx)

    @test A.Nx==Nx
    @test mean(A) == zeros(Nx)
    @test cov(A) == Diagonal(ones(Nx))


    X = zeros(Ny + Nx, Ne)
    X .= randn(Ny + Nx, Ne)

    Xadd = deepcopy(X)
    A(Xadd, Ny+1, Ny+Nx)
    @test size(Xadd) == (Ny + Nx, Ne)
    @test Xadd != X
    @test norm(Xadd[1:Ny,:] -  X[1:Ny,:]) < 10*eps()
    @test norm(Xadd[Ny+1:Ny+Nx,:] -  X[Ny+1:Ny+Nx,:]) > 10*eps()



    Σ = zeros(Nx,Nx)

    for i=1:Nx
        for j=1:Nx
            Σ[i,j] = 1.0/(i + j)
        end
    end

    A = AdditiveInflation(Nx, zeros(Nx), Σ)

    @test A.Nx==10
    @test mean(A) == zeros(Nx)
    @test cov(A) == Σ
end

@testset "Additive inflation II" begin
    Nx= 10
    Ne = 20
    A = AdditiveInflation(Nx)

    @test A.Nx==10
    @test mean(A) == zeros(Nx)
    @test cov(A) == Diagonal(ones(Nx))


    X = zeros(Nx, Ne)
    X .= randn(Nx, Ne)

    Xadd = deepcopy(X)
    A(Xadd)
    @test size(Xadd) == (Nx, Ne)
    @test Xadd != X


    Σ = zeros(Nx,Nx)

    for i=1:Nx
        for j=1:Nx
            Σ[i,j] = 1.0/(i + j)
        end
    end

    A = AdditiveInflation(Nx, zeros(Nx), Σ)

    @test A.Nx==10
    @test mean(A) == zeros(Nx)
    @test cov(A) == Σ
end

@testset "Multiplicative inflation I" begin
    Ny = 10
    Nx= 10
    Ne = 20
    A = MultiplicativeInflation(1.2)

    @test A.β==1.2

    X = zeros(Ny + Nx, Ne)
    X .= randn(Ny + Nx, Ne)

    Xmul = deepcopy(X)
    A(Xmul, Ny+1, Ny+Nx)

    for i=1:Ne
        x̄ = deepcopy(mean(X, dims = 2))[Ny+1:Ny+Nx,1]
        @test norm(Xmul[1:Ny,i] - X[1:Ny,i])<1e-10
        @test norm(Xmul[Ny+1:Ny+Nx,i] - (x̄ + A.β*(X[Ny+1:Ny+Nx,i] - x̄)))<1e-10
    end
end

@testset "Multiplicative inflation II" begin
    Nx= 10
    Ne = 20
    A = MultiplicativeInflation(1.2)

    @test A.β==1.2

    X = zeros(Nx, Ne)
    X .= randn(Nx, Ne)

    Xmul = deepcopy(X)
    A(Xmul)

    for i=1:Ne
        x̄ = deepcopy(mean(X, dims = 2))[:,1]
        @test norm(Xmul[:,i] - (x̄ + A.β*(X[:,i] - x̄)))<1e-10
    end
end

@testset "MultiAdditive Inflation" begin

    T = MultiAddInflation(10)

    @test size(T)== 10
    @test T.β == 1.0
    
    @test mean(T) == zeros(10)
    @test cov(T) ==  Diagonal(ones(10))


    T = MultiAddInflation(10, 1.01, zeros(10), 1.01)

    @test T.β == 1.01
    @test mean(T) == zeros(10)
    @test cov(T) == Diagonal(1.01^2*ones(10))
end
