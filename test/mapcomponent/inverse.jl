

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

    @test norm(Xmodified[end,:] - X[end,:])<1e-10

    @test norm(Xmodified[1:end-1,:] - X[1:end-1,:])<1e-10
end
