
@testset "Verify evaluation of LinHermiteHermiteMapComponent" begin

    Nx = 2
    Ne = 500
    X = randn(Nx, Ne)
    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]
    for b in Blist
        B = MultiBasis(b, Nx)

        idx = [0 0; 0 1; 1 0; 1 1; 1 2]
        truncidx = idx[1:2:end,:]
        Nψ = 5

        coeff = randn(Nψ)

        f = ExpandedFunction(B, idx, coeff)
        R = IntegratedFunction(f)
        C = HermiteMapComponent(R)
        L = LinHermiteMapComponent(X, C)

        X0 = deepcopy(X)

        # Test evaluate
        ψt = zeros(Ne)
        transform!(L.L, X)

        for i=1:Ne
          x = view(X,:,i)
          ψt[i] = R.f(vcat(x[1:end-1], 0.0)) + quadgk(t->R.g(ForwardDiff.gradient(y->R.f(y), vcat(x[1:end-1],t))[end]), 0, x[end])[1]
        end

        itransform!(L.L, X)

        ψ = evaluate(L, X0)

        @test norm(X - X0)<1e-10
        @test norm(ψ - ψt)<1e-10
    end
end


@testset "Test optimization for LinHermiteMapComponent when maxterms = nothing" begin
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

    C = HermiteMapComponent(m, Nx, idx, coeff);
    Clin = deepcopy(C)

    C_new, error_new = AdaptiveTransportMap.optimize(C, X, nothing; verbose = false);

    itransform!(L, X)

    L = LinHermiteMapComponent(Xlin, Clin)

    L_new, error_Lnew = AdaptiveTransportMap.optimize(L, Xlin, nothing; verbose = false);

    @test norm(error_new - error_Lnew)<1e-8
    @test norm(getcoeff(C_new) - getcoeff(L_new))<1e-8

    @test norm(X - Xlin)<1e-8
end


@testset "Test optimization for LinHermiteMapComponent when maxterms is an integer" begin
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

    C = HermiteMapComponent(m, Nx, idx, coeff);
    Clin = deepcopy(C)

    C_new, error_new = AdaptiveTransportMap.optimize(C, X, 4; verbose = false);

    itransform!(L, X)

    L = LinHermiteMapComponent(Xlin, Clin)

    L_new, error_Lnew = AdaptiveTransportMap.optimize(L, Xlin, 4; verbose = false);

    @test norm(error_new - error_Lnew)<1e-8
    @test norm(getcoeff(C_new) - getcoeff(L_new))<1e-8

    @test norm(X - Xlin)<1e-8
end


@testset "Test optimization for LinHermiteMapComponent when maxterms is kfold" begin
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

    C = HermiteMapComponent(m, Nx, idx, coeff);
    Clin = deepcopy(C)

    C_new, error_new = AdaptiveTransportMap.optimize(C, X, "kfold"; verbose = false);

    itransform!(L, X)

    L = LinHermiteMapComponent(Xlin, Clin)

    L_new, error_Lnew = AdaptiveTransportMap.optimize(L, Xlin, "kfold"; verbose = false);

    @test norm(error_new - error_Lnew)<1e-8
    @test norm(getcoeff(C_new) - getcoeff(L_new))<1e-8

    @test norm(X - Xlin)<1e-8
end

@testset "Test optimization for LinHermiteMapComponent when maxterms is split" begin
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

    C = HermiteMapComponent(m, Nx, idx, coeff);
    Clin = deepcopy(C)

    C_new, error_new = AdaptiveTransportMap.optimize(C, X, "split"; verbose = false);

    itransform!(L, X)

    L = LinHermiteMapComponent(Xlin, Clin)

    L_new, error_Lnew = AdaptiveTransportMap.optimize(L, Xlin, "split"; verbose = false);

    @test norm(error_new[1] - error_Lnew[1])<1e-8
    @test norm(error_new[2] - error_Lnew[2])<1e-8
    @test norm(getcoeff(C_new) - getcoeff(L_new))<1e-8

    @test norm(X - Xlin)<1e-8
end
