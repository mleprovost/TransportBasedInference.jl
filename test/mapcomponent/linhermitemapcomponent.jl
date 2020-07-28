
@testset "Verify evaluation of LinHermiteMapComponent" begin

    Nx = 2
    Ne = 500
    X = randn(Nx, Ne)
    B = MultiBasis(CstProHermite(6; scaled =true), Nx)

    idx = [0 0; 0 1; 1 0; 1 1; 1 2]
    truncidx = idx[1:2:end,:]
    Nψ = 5

    coeff = randn(Nψ)

    f = ExpandedFunction(B, idx, coeff)
    fp = ParametricFunction(f);
    R = IntegratedFunction(fp)
    Hk = HermiteMapk(R)
    Lk = LinHermiteMapk(X, Hk)

    X0 = deepcopy(X)

    # Test evaluate
    ψt = zeros(Ne)
    transform!(Lk.L, X)

    for i=1:Ne
      x = view(X,:,i)
      ψt[i] = R.f.f(vcat(x[1:end-1], 0.0)) + quadgk(t->R.g(ForwardDiff.gradient(y->R.f.f(y), vcat(x[1:end-1],t))[end]), 0, x[end])[1]
    end

    itransform!(Lk.L, X)

    ψ = evaluate(Lk, X0)

    @test norm(X - X0)<1e-10
    @test norm(ψ - ψt)<1e-10
end


@testset "Test optimization for LinHermiteMapk when maxterms = nothing" begin
    Nx = 2

    m = 20
    Ne = 500
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)
    Xlin = deepcopy(X)
    L = LinearTransform(X; diag = true)
    transform!(L, X);

    idx = [0 0; 0 1; 0 2; 0 3; 1 1]
    coeff =  [   -0.7708538710735897;
          0.1875006767230035;
          1.419396079869706;
          0.2691388018878241;
         -2.9879004455954723];

    Hk = HermiteMapk(m, Nx, idx, coeff);
    Hklin = deepcopy(Hk)

    Hk_new, error_new = AdaptiveTransportMap.optimize(Hk, X, nothing; verbose = false);

    itransform!(L, X)

    Lk = LinHermiteMapk(Xlin, Hklin)

    Lk_new, error_Lknew = AdaptiveTransportMap.optimize(Lk, Xlin, nothing; verbose = false);

    @test norm(error_new - error_Lknew)<1e-8
    @test norm(getcoeff(Hk_new) - getcoeff(Lk_new))<1e-8

    @test norm(X - Xlin)<1e-8
end


@testset "Test optimization for LinHermiteMapk when maxterms is an integer" begin
    Nx = 2

    m = 20
    Ne = 500
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)
    Xlin = deepcopy(X)
    L = LinearTransform(X; diag = true)
    transform!(L, X);

    idx = [0 0; 0 1; 0 2; 0 3; 1 1]
    coeff =  [ -0.7708538710735897;
                0.1875006767230035;
                1.419396079869706;
                0.2691388018878241;
               -2.9879004455954723];

    Hk = HermiteMapk(m, Nx, idx, coeff);
    Hklin = deepcopy(Hk)

    Hk_new, error_new = AdaptiveTransportMap.optimize(Hk, X, 4; verbose = false);

    itransform!(L, X)

    Lk = LinHermiteMapk(Xlin, Hklin)

    Lk_new, error_Lknew = AdaptiveTransportMap.optimize(Lk, Xlin, 4; verbose = false);

    @test norm(error_new - error_Lknew)<1e-8
    @test norm(getcoeff(Hk_new) - getcoeff(Lk_new))<1e-8

    @test norm(X - Xlin)<1e-8
end


@testset "Test optimization for LinHermiteMapk when maxterms is kfold" begin
    Nx = 2

    m = 20
    Ne = 500
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)
    Xlin = deepcopy(X)
    L = LinearTransform(X; diag = true)
    transform!(L, X);

    idx = [0 0; 0 1; 0 2; 0 3; 1 1]
    coeff =  [ -0.7708538710735897;
                0.1875006767230035;
                1.419396079869706;
                0.2691388018878241;
               -2.9879004455954723];

    Hk = HermiteMapk(m, Nx, idx, coeff);
    Hklin = deepcopy(Hk)

    Hk_new, error_new = AdaptiveTransportMap.optimize(Hk, X, "kfold"; verbose = false);

    itransform!(L, X)

    Lk = LinHermiteMapk(Xlin, Hklin)

    Lk_new, error_Lknew = AdaptiveTransportMap.optimize(Lk, Xlin, "kfold"; verbose = false);

    @test norm(error_new - error_Lknew)<1e-8
    @test norm(getcoeff(Hk_new) - getcoeff(Lk_new))<1e-8

    @test norm(X - Xlin)<1e-8
end

@testset "Test optimization for LinHermiteMapk when maxterms is split" begin
    Nx = 2

    m = 20
    Ne = 500
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)
    Xlin = deepcopy(X)
    L = LinearTransform(X; diag = true)
    transform!(L, X);

    idx = [0 0; 0 1; 0 2; 0 3; 1 1]
    coeff =  [ -0.7708538710735897;
                0.1875006767230035;
                1.419396079869706;
                0.2691388018878241;
               -2.9879004455954723];

    Hk = HermiteMapk(m, Nx, idx, coeff);
    Hklin = deepcopy(Hk)

    Hk_new, error_new = AdaptiveTransportMap.optimize(Hk, X, "split"; verbose = false);

    itransform!(L, X)

    Lk = LinHermiteMapk(Xlin, Hklin)

    Lk_new, error_Lknew = AdaptiveTransportMap.optimize(Lk, Xlin, "split"; verbose = false);

    @test norm(error_new[1] - error_Lknew[1])<1e-8
    @test norm(error_new[2] - error_Lknew[2])<1e-8
    @test norm(getcoeff(Hk_new) - getcoeff(Lk_new))<1e-8

    @test norm(X - Xlin)<1e-8
end
