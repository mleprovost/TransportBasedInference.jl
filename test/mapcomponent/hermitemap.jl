

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
    H1 = MapComponent(IntegratedFunction(ExpandedFunction(MB1, idx1, coeff1)))

    idx2   = reshape([0 0 ;0 1; 1 0; 2 0], (4,2))
    coeff2 = randn(4)
    MB2 = MultiBasis(B, 2)
    H2 = MapComponent(IntegratedFunction(ExpandedFunction(MB2, idx2, coeff2)))


    idx3   = reshape([0 0 0; 0 0 1; 0 1 0; 1 0 1], (4,3))
    coeff3 = randn(4)
    MB3 = MultiBasis(B, 3)
    H3 = MapComponent(IntegratedFunction(ExpandedFunction(MB3, idx3, coeff3)))

    L = LinearTransform(X; diag = true)
    M = HermiteMap(L, MapComponent[H1; H2; H3])

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

@testset "Validate evalution of HermiteMap with multi-threading" begin

    Nx = 3
    Ne = 100
    m = 10
    X = randn(Nx, Ne) .* randn(Nx, Ne) .+ rand(Nx);
    X0 = deepcopy(X)

    B = CstProHermite(m-2; scaled =true)

    idx1   = reshape([0; 1; 2], (3,1))
    coeff1 = randn(3)
    MB1 = MultiBasis(B, 1)
    H1 = MapComponent(IntegratedFunction(ExpandedFunction(MB1, idx1, coeff1)))

    idx2   = reshape([0 0 ;0 1; 1 0; 2 0], (4,2))
    coeff2 = randn(4)
    MB2 = MultiBasis(B, 2)
    H2 = MapComponent(IntegratedFunction(ExpandedFunction(MB2, idx2, coeff2)))


    idx3   = reshape([0 0 0; 0 0 1; 0 1 0; 1 0 1], (4,3))
    coeff3 = randn(4)
    MB3 = MultiBasis(B, 3)
    H3 = MapComponent(IntegratedFunction(ExpandedFunction(MB3, idx3, coeff3)))

    L = LinearTransform(X; diag = true)
    M = HermiteMap(L, MapComponent[H1; H2; H3])

    out = zero(X)
    out_thread = zero(X)
    # Without rescaling of the variables

    evaluate!(out, M, X; apply_rescaling = false, P = serial)
    evaluate!(out_thread, M, X; apply_rescaling = false, P = thread)


    @test norm(out[1,:] - evaluate(H1, X[1:1,:]))<1e-10
    @test norm(out[2,:] - evaluate(H2, X[1:2,:]))<1e-10
    @test norm(out[3,:] - evaluate(H3, X[1:3,:]))<1e-10

    @test norm(out_thread[1,:] - evaluate(H1, X[1:1,:]))<1e-10
    @test norm(out_thread[2,:] - evaluate(H2, X[1:2,:]))<1e-10
    @test norm(out_thread[3,:] - evaluate(H3, X[1:3,:]))<1e-10

    # With rescaling of the variables

    evaluate!(out, M, X; apply_rescaling = true, P = serial)
    evaluate!(out_thread, M, X; apply_rescaling = true, P = thread)

    transform!(L, X)
    @test norm(out[1,:] - evaluate(H1, X[1:1,:]))<1e-10
    @test norm(out[2,:] - evaluate(H2, X[1:2,:]))<1e-10
    @test norm(out[3,:] - evaluate(H3, X[1:3,:]))<1e-10

    @test norm(out_thread[1,:] - evaluate(H1, X[1:1,:]))<1e-10
    @test norm(out_thread[2,:] - evaluate(H2, X[1:2,:]))<1e-10
    @test norm(out_thread[3,:] - evaluate(H3, X[1:3,:]))<1e-10

    itransform!(L, X)

    @test norm(X0 - X)<1e-12
end


@testset "Verify that Serial and Multi-threading optimization are working properly" begin
    Nx = 50
    Ne = 1000
    m = 10


    X = randn(Nx, Ne) .* randn(Nx, Ne) .+ randn(Nx)
    M = HermiteMap(m, X)
    Mthread = HermiteMap(m,X)

    optimize(M, X, 10; P = serial)
    optimize(Mthread, X, 10; P = thread)

    for i=1:Nx
        @test norm(getcoeff(M.C[i]) - getcoeff(Mthread.C[i]))<1e-8
    end

    @test norm(evaluate(M, X; P = serial) - evaluate(Mthread, X; P = thread))<1e-8
end


@testset "Verify log_pdf function" begin
    # For diagonal rescaling
    Nx = 2
    m = 10
    X  =  Matrix([0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]');
    X0 = deepcopy(X)

    B = MultiBasis(CstProHermite(m; scaled =true), Nx);

    M = HermiteMap(m, X);


    idx1 = reshape([0; 1; 2;3],(4,1))
    coeff1 =[-7.37835916713439;
             16.897974063168977;
             -0.20935914261414929;
              8.116032140446544];

    M[1] = MapComponent(m, 1,  idx1, coeff1)

    idx2 = [ 0  0; 0 1; 1 0; 2 0];
    coeff2 = [2.995732886294909; -2.2624558703623903; -3.3791974895855486; 1.3808989978516617]

    M[2] = MapComponent(m, 2,  idx2, coeff2);

    @test norm(log_pdf(M, X) -   [-1.849644462509894;
                                  -1.239419791201162;
                                  -3.307440928766559;
                                  -2.482615159224171;
                                  -2.392017816948268;
                                  -1.331903266052277;
                                  -3.638709417555485;
                                  -2.242238529589172])<1e-6

    @test norm(X - X0)<1e-10

    # For Cholesky refactorization

    Nx = 2
    m = 10
    X  =  Matrix([0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]');
    X0 = deepcopy(X)

    B = MultiBasis(CstProHermite(m; scaled =true), Nx);

    M = HermiteMap(m, X; diag = false);


    idx1 = reshape([0; 1; 2;3],(4,1))
    coeff1 =[-7.37835916713439;
             16.897974063168977;
             -0.20935914261414929;
              8.116032140446544];

    M[1] = MapComponent(m, 1,  idx1, coeff1)

    idx2 = [ 0  0; 0 1; 1 0; 2 0];
    coeff2 = [2.995732886294909; -2.2624558703623903; -3.3791974895855486; 1.3808989978516617]

    M[2] = MapComponent(m, 2,  idx2, coeff2);

    @test norm(log_pdf(M, X) - [  -2.284420690193831;
                                  -1.113693643793066;
                                  -3.927922581187332;
                                  -2.461816757918692;
                                  -2.328286973844927;
                                  -1.529996011165051;
                                  -3.444940697346834;
                                  -1.999419335266747])<1e-6

    @test norm(X - X0)<1e-10

end
