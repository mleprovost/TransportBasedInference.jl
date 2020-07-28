

@testset "Test inverse function" begin

    Ne = 200
    Nx = 3

    idx = [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 0 1; 0 1 1; 0 2 1; 0 3 1; 1 3 1]

    Nψ = size(idx,1)

    coeff = randn(Nψ)

    B = MultiBasis(CstProHermite(6; scaled =true), Nx)
    f = ParametricFunction(ExpandedFunction(B, idx, coeff))
    R = IntegratedFunction(f)

    X = randn(Nx, Ne) .* randn(Nx, Ne)

    F = evaluate(R, X)

    Xmodified = deepcopy(X)

    Xmodified[end,:] .+= cos.(2*π*randn(Ne)) .* exp.(-randn(Ne).^2/2)

    @test norm(Xmodified[end,:]-X[end,:])>1e-6

    S = Storage(f, Xmodified)

    inverse!(Xmodified, F, R, S)

    @test norm(Xmodified[end,:] - X[end,:])<1e-8

    @test norm(Xmodified[1:end-1,:] - X[1:end-1,:])<1e-8
end

@testset "Test inverse HermiteMapComponent" begin

    Ne = 200
    Nx = 3

    idx = [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 0 1; 0 1 1; 0 2 1; 0 3 1; 1 3 1]

    Nψ = size(idx,1)

    coeff = randn(Nψ)

    Hk = HermiteMapk(20, Nx, idx, coeff)

    X = randn(Nx, Ne) .* randn(Nx, Ne)

    F = evaluate(Hk.I, X)

    Xmodified = deepcopy(X)

    Xmodified[end,:] .+= cos.(2*π*randn(Ne)) .* exp.(-randn(Ne).^2/2)

    @test norm(Xmodified[end,:]-X[end,:])>1e-6

    S = Storage(Hk.I.f, Xmodified)

    inverse!(Xmodified, F, Hk, S)

    @test norm(Xmodified[end,:] - X[end,:])<1e-8

    @test norm(Xmodified[1:end-1,:] - X[1:end-1,:])<1e-8
end

@testset "Test inverse LinHermiteMapComponent" begin

    Ne = 200
    Nx = 3

    idx = [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 0 1; 0 1 1; 0 2 1; 0 3 1; 1 3 1]

    Nψ = size(idx,1)

    coeff = randn(Nψ)

    Hk = HermiteMapk(20, Nx, idx, coeff)
    X = randn(Nx, Ne) .* randn(Nx, Ne)

    X0 = deepcopy(X)
    Xmodified = deepcopy(X)

    Lk = LinHermiteMapk(X, Hk)
    F = evaluate(Lk, X)

    @test norm(X -X0)<1e-8

    Xmodified[end,:] .+= cos.(2*π*randn(Ne)) .* exp.(-randn(Ne).^2/2)

    Lkmodified = LinHermiteMapk(Xmodified, Hk)

    @test norm(Xmodified[end,:]-X[end,:])>1e-6

    transform!(Lkmodified.L, Xmodified)
    S = Storage(Lkmodified.H.I.f, Xmodified);
    itransform!(Lkmodified.L, Xmodified)

    inverse!(Xmodified, F, Lk, S)

    @test norm(Xmodified[end,:] - X[end,:])<1e-8

    @test norm(Xmodified[1:end-1,:] - X[1:end-1,:])<1e-8
end
