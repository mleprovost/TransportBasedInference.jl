

@testset "Test evaluation of HermiteMap" begin

    Nx = 3
    Ne = 100
    m = 10
    X = randn(Nx, Ne) .* randn(Nx, Ne) .+ rand(Nx);
    X0 = deepcopy(X)

    B = CstProHermite(m-2; scaled =true)

    idx1   = reshape([0; 1; 2], (3,1))
    coeff1 = randn(3)
    MB1 = MultiBasis(B, 1)
    H1 = HermiteMapk(IntegratedFunction(ExpandedFunction(MB1, idx1, coeff1)))

    idx2   = reshape([0 0 ;0 1; 1 0; 2 0], (4,2))
    coeff2 = randn(4)
    MB2 = MultiBasis(B, 2)
    H2 = HermiteMapk(IntegratedFunction(ExpandedFunction(MB2, idx2, coeff2)))


    idx3   = reshape([0 0 0; 0 0 1; 0 1 0; 1 0 1], (4,3))
    coeff3 = randn(4)
    MB3 = MultiBasis(B, 3)
    H3 = HermiteMapk(IntegratedFunction(ExpandedFunction(MB3, idx3, coeff3)))

    L = LinearTransform(X; diag = true)
    M = HermiteMap{m, Nx}(L, HermiteMapk[H1; H2; H3])

    out = zero(X)
    # Without rescaling of the variables

    evaluate!(out, M, X; apply_rescaling = false)


    @test norm(out[1,:] - evaluate(H1, X[1:1,:]))<1e-10
    @test norm(out[2,:] - evaluate(H2, X[1:2,:]))<1e-10
    @test norm(out[3,:] - evaluate(H3, X[1:3,:]))<1e-10

    # With rescaling of the variables

    evaluate!(out, M, X; apply_rescaling = true)

    transform!(L, X)
    @test norm(out[1,:] - evaluate(H1, X[1:1,:]))<1e-10
    @test norm(out[2,:] - evaluate(H2, X[1:2,:]))<1e-10
    @test norm(out[3,:] - evaluate(H3, X[1:3,:]))<1e-10

    itransform!(L, X)

    @test norm(X0 - X)<1e-12
end
