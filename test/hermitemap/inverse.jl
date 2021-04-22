using AdaptiveTransportMap: evaluate

@testset "Test inverse function" begin

    Ne = 200
    Nx = 3

    idx = [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 0 1; 0 1 1; 0 2 1; 0 3 1; 1 3 1]

    Nψ = size(idx,1)

    coeff = randn(Nψ)

    B = MultiBasis(CstProHermite(6), Nx)
    f = ExpandedFunction(B, idx, coeff)
    R = IntegratedFunction(f)

    X = randn(Nx, Ne) .* randn(Nx, Ne)

    F = evaluate(R, X)

    Xmodified = deepcopy(X)

    Xmodified[end,:] .+= cos.(2*π*randn(Ne)) .* exp.(-randn(Ne).^2/2)

    @test norm(Xmodified[end,:]-X[end,:])>1e-6

    S = Storage(f, Xmodified)

    inverse!(Xmodified, F, R, S)

    @test norm(Xmodified[end,:] - X[end,:])<2e-8

    @test norm(Xmodified[1:end-1,:] - X[1:end-1,:])<2e-8
end

@testset "Test inverse HermiteHermiteMapComponent" begin

    Ne = 200
    Nx = 3

    idx = [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 0 1; 0 1 1; 0 2 1; 0 3 1; 1 3 1]

    Nψ = size(idx,1)

    coeff = randn(Nψ)

    C = HermiteMapComponent(20, Nx, idx, coeff)

    X = randn(Nx, Ne) .* randn(Nx, Ne)

    F = evaluate(C.I, X)

    Xmodified = deepcopy(X)

    Xmodified[end,:] .+= cos.(2*π*randn(Ne)) .* exp.(-randn(Ne).^2/2)

    @test norm(Xmodified[end,:]-X[end,:])>1e-6

    S = Storage(C.I.f, Xmodified)

    inverse!(Xmodified, F, C, S)

    @test norm(Xmodified[end,:] - X[end,:])<1e-5

    @test norm(Xmodified[1:end-1,:] - X[1:end-1,:])<1e-5
end

@testset "Test inverse LinHermiteHermiteMapComponent" begin

    Ne = 200
    Nx = 3

    idx = [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 0 1; 0 1 1; 0 2 1; 0 3 1; 1 3 1]

    Nψ = size(idx,1)

    coeff = randn(Nψ)

    C = HermiteMapComponent(20, Nx, idx, coeff)
    X = randn(Nx, Ne) .* randn(Nx, Ne)

    X0 = deepcopy(X)
    Xmodified = deepcopy(X)

    L = LinHermiteMapComponent(X, C)
    F = evaluate(L, X)

    @test norm(X -X0)<1e-8

    Xmodified[end,:] .+= cos.(2*π*randn(Ne)) .* exp.(-randn(Ne).^2/2)

    Lmodified = LinHermiteMapComponent(Xmodified, C)

    @test norm(Xmodified[end,:]-X[end,:])>1e-6

    transform!(Lmodified.L, Xmodified)
    S = Storage(Lmodified.C.I.f, Xmodified);
    itransform!(Lmodified.L, Xmodified)

    inverse!(Xmodified, F, L, S)

    @test norm(Xmodified[end,:] - X[end,:])<1e-5

    @test norm(Xmodified[1:end-1,:] - X[1:end-1,:])<1e-5
end
