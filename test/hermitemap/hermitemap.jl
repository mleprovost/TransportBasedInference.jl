import AdaptiveTransportMap: optimize

@testset "Verify clearcoeff!" begin
  m = 5
  Nx = 10
  Nψ = 5

  coeff = randn(Nψ)

  C = HermiteMapComponent[]

  for i=1:Nx
      idxi = rand(0:m, Nψ, i)
      coeffi = randn(Nψ)
      push!(C, HermiteMapComponent(m, i, deepcopy(idxi), deepcopy(coeffi)))
  end

  M = HermiteMap(m, Nx, LinearTransform(Nx), C)

  clearcoeff!(M)

  for i=1:Nx
      @test norm(getcoeff(M.C[i]) - zeros(Nψ))<1e-12
  end
end


@testset "Test evaluation of HermiteMap" begin

    Nx = 3
    Ne = 100
    m = 10
    X = randn(Nx, Ne) .* randn(Nx, Ne) .+ rand(Nx);
    X0 = deepcopy(X)

    B = CstProHermite(m-2)

    idx1   = reshape([0; 1; 2], (3,1))
    coeff1 = randn(3)
    MB1 = MultiBasis(B, 1)
    H1 = HermiteMapComponent(IntegratedFunction(ExpandedFunction(MB1, idx1, coeff1)))

    idx2   = reshape([0 0 ;0 1; 1 0; 2 0], (4,2))
    coeff2 = randn(4)
    MB2 = MultiBasis(B, 2)
    H2 = HermiteMapComponent(IntegratedFunction(ExpandedFunction(MB2, idx2, coeff2)))


    idx3   = reshape([0 0 0; 0 0 1; 0 1 0; 1 0 1], (4,3))
    coeff3 = randn(4)
    MB3 = MultiBasis(B, 3)
    H3 = HermiteMapComponent(IntegratedFunction(ExpandedFunction(MB3, idx3, coeff3)))

    L = LinearTransform(X; diag = true)
    M = HermiteMap(L, HermiteMapComponent[H1; H2; H3])

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

    B = CstProHermite(m-2)

    idx1   = reshape([0; 1; 2], (3,1))
    coeff1 = randn(3)
    MB1 = MultiBasis(B, 1)
    H1 = HermiteMapComponent(IntegratedFunction(ExpandedFunction(MB1, idx1, coeff1)))

    idx2   = reshape([0 0 ;0 1; 1 0; 2 0], (4,2))
    coeff2 = randn(4)
    MB2 = MultiBasis(B, 2)
    H2 = HermiteMapComponent(IntegratedFunction(ExpandedFunction(MB2, idx2, coeff2)))


    idx3   = reshape([0 0 0; 0 0 1; 0 1 0; 1 0 1], (4,3))
    coeff3 = randn(4)
    MB3 = MultiBasis(B, 3)
    H3 = HermiteMapComponent(IntegratedFunction(ExpandedFunction(MB3, idx3, coeff3)))

    L = LinearTransform(X; diag = true)
    M = HermiteMap(L, HermiteMapComponent[H1; H2; H3])

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


@testset "Verify that Serial and Multi-threading optimization are working properly without/with QR" begin
    Nx = 20
    Ne = 300
    m = 10


    X = randn(Nx, Ne) .* randn(Nx, Ne) .+ randn(Nx)
    M_noqr = HermiteMap(m, X)
    Mthread_noqr = HermiteMap(m,X)

    M_qr = HermiteMap(m, X)
    Mthread_qr = HermiteMap(m,X)

    optimize(M_noqr, X, 10; withqr = false, P = serial)
    optimize(Mthread_noqr, X, 10; withqr = false, P = thread)

    optimize(M_qr, X, 10; withqr = true, P = serial)
    optimize(Mthread_qr, X, 10; withqr = true, P = thread)

    for i=1:Nx
        @test norm(getcoeff(M_noqr.C[i]) - getcoeff(Mthread_noqr.C[i]))<1e-4
        @test norm(getcoeff(M_noqr.C[i]) - getcoeff(M_qr.C[i]))<1e-4
        @test norm(getcoeff(M_noqr.C[i]) - getcoeff(Mthread_qr.C[i]))<1e-4
    end

    @test norm(evaluate(M_noqr, X; P = serial) - evaluate(Mthread_noqr, X; P = thread))<1e-4
    @test norm(evaluate(M_noqr, X; P = serial) - evaluate(M_qr, X; P = thread))<1e-4
    @test norm(evaluate(M_noqr, X; P = serial) - evaluate(Mthread_qr, X; P = thread))<1e-4

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

    B = MultiBasis(CstProHermite(m), Nx);

    M = HermiteMap(m, X);


    idx1 = reshape([0; 1; 2;3],(4,1))
    coeff1 =[-7.37835916713439;
             16.897974063168977;
             -0.20935914261414929;
              8.116032140446544];

    M[1] = HermiteMapComponent(m, 1,  idx1, coeff1)

    idx2 = [ 0  0; 0 1; 1 0; 2 0];
    coeff2 = [2.995732886294909; -2.2624558703623903; -3.3791974895855486; 1.3808989978516617]

    M[2] = HermiteMapComponent(m, 2,  idx2, coeff2);

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

    B = MultiBasis(CstProHermite(m), Nx);

    M = HermiteMap(m, X; diag = false);


    idx1 = reshape([0; 1; 2;3],(4,1))
    coeff1 =[-7.37835916713439;
             16.897974063168977;
             -0.20935914261414929;
              8.116032140446544];

    M[1] = HermiteMapComponent(m, 1,  idx1, coeff1)

    idx2 = [ 0  0; 0 1; 1 0; 2 0];
    coeff2 = [2.995732886294909; -2.2624558703623903; -3.3791974895855486; 1.3808989978516617]

    M[2] = HermiteMapComponent(m, 2,  idx2, coeff2);

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


@testset "Verify the different optimization max_terms ∈ Int64" begin

    Nx = 3
    Ne = 8

    X = Matrix([  0.112122    0.371145  -1.09587;
                 -0.124971    0.563166   1.21132;
                 -0.0781768   0.974237   1.16974;
                  0.7536      3.67691    1.31592;
                  0.329414   -1.57155    1.02031;
                  0.0263465   0.445467   1.33992;
                  0.457334    0.20462    1.91469;
                  0.190431   -0.22547    1.25082]');

    m = 30

    M = HermiteMap(m, X)

    optimize(M, X, 5; withconstant = false)

    @test norm(log_pdf(M, X) - [  -1.648733127003676;
                                  -0.524080373136291;
                                   0.466815189258083;
                                  -1.265418705621051;
                                  -4.178674562888346;
                                   0.977833056870988;
                                  -3.449875825411972;
                                   0.460274224407305])<1e-3

    @test getidx(M[1]) == reshape([1; 2; 3; 4; 5],(5,1))
    @test norm(getcoeff(M[1]) - [-1.485021686942572;
                                  1.741440183812295;
                                 -3.814293233298080;
                                  2.150582375812929;
                                 -1.418659400598183])<1e-3

    @test getidx(M[2]) == [1 0; 0 1; 0 2; 0 3; 2 0]
    @test norm(getcoeff(M[2]) - [18.385719748659056 ;
                                -20.079305399569837 ;
                                 5.208177303845926  ;
                                -7.543285947028642  ;
                                 1.043264254165708])<1e-3

    @test getidx(M[3]) == [0 0 1; 0 0 2; 0 0 3; 0 0 4; 0 1 0]
    @test norm(getcoeff(M[3]) - [  9.801288179417845;
                                 -18.297926511077648;
                                  14.512139032442953;
                                 -16.509726071255606;
                                  -0.962812577491049])<1e-2

end


@testset "Verify the different optimization max_terms = Split" begin

    Nx = 3
    Ne = 8

    X = Matrix([  0.112122    0.371145  -1.09587;
                 -0.124971    0.563166   1.21132;
                 -0.0781768   0.974237   1.16974;
                  0.7536      3.67691    1.31592;
                  0.329414   -1.57155    1.02031;
                  0.0263465   0.445467   1.33992;
                  0.457334    0.20462    1.91469;
                  0.190431   -0.22547    1.25082]');

    m = 30

    M = HermiteMap(m, X)

    optimize(M, X, "Split"; withconstant = false)

    @test norm(log_pdf(M, X) - [ -21.745907202610731;
                                  -0.464429325572978;
                                  -0.155330841509541;
                                  -4.329981710326985;
                                  -2.539350574125156;
                                   0.206049495114118;
                                  -3.760956579928015;
                                  -0.463356887488941])<1e-4

    @test getidx(M[1]) == reshape([1; 2; 3],(3,1))
    @test norm(getcoeff(M[1]) - [-0.907122832552077;
                                 -0.471583533261757;
                                 -1.653849103515142])<1e-4

    @test getidx(M[2]) == [1 0; 0 1; 0 2]
    @test norm(getcoeff(M[2]) - [ 6.254057966900727;
                                 -5.202600223050682;
                                  1.829358246492261])<1e-4

    @test getidx(M[3]) == [0 0 1; 0 0 2; 0 1 0]
    @test norm(getcoeff(M[3]) - [ -0.348002361230585;
                                   6.995417803422829;
                                  -2.169327382026170])<1e-4

end


@testset "Verify the different optimization max_terms = kfold" begin

    Nx = 3
    Ne = 16

    X = Matrix([  0.112122    0.371145  -1.09587  ;
                 -0.124971    0.563166   1.21132  ;
                 -0.0781768   0.974237   1.16974  ;
                  0.7536      3.67691    1.31592  ;
                  0.329414   -1.57155    1.02031  ;
                  0.0263465   0.445467   1.33992  ;
                  0.457334    0.20462    1.91469  ;
                  0.190431   -0.22547    1.25082  ;
                 -0.383296    0.838512  -0.687386 ;
                 -1.57094    -0.808855   0.0632401;
                 -0.684854   -0.12027    0.187772 ;
                  0.767601    2.35813   -0.658357 ;
                  0.338965   -0.168985  -0.00250368;
                 -0.27756    -0.790416  -0.133158 ;
                 -0.824008    0.459429  -0.235808 ;
                  0.0145145   0.123892   0.0831142]')

    m = 30

    M = HermiteMap(m, X)

    M = HermiteMap(m, X)

    optimize(M, X, "kfold"; withconstant = false)

    @test norm(log_pdf(M, X) - [  -3.880921471282678;
                                  -2.753134673606892;
                                  -2.839021221004361;
                                  -7.308105457845060;
                                  -3.838700341109559;
                                  -2.845755023406152;
                                  -3.982132693868225;
                                  -2.869826713565865;
                                  -3.474164717698875;
                                  -6.086150330191086;
                                  -3.064117775500836;
                                  -5.349287829523062;
                                  -2.704253030752519;
                                  -3.037988603791519;
                                  -3.557543327559524;
                                  -2.443326419210569])<1e-4

    @test getidx(M[1]) == reshape([1],(1,1))
    @test norm(getcoeff(M[1]) - [-0.118440805265421])<1e-4

    @test getidx(M[2]) == [0 1]
    @test norm(getcoeff(M[2]) - [0.146373981523974])<1e-4

    @test getidx(M[3]) == [1 0 0]
    @test norm(getcoeff(M[3]) - [-0.067872411753720])<1e-4
end


@testset "Test inversion of the Hermite Map I" begin
    Nx = 100
    m = 20
    Ne = 500
    Xprior = randn(Nx, Ne).*randn(Nx, Ne)
    Xpost = deepcopy(Xprior) .+ randn(Nx, Ne)

    M = HermiteMap(m, Xprior; diag = true)
    optimize(M, Xprior, 5; withconstant = false)

    F = evaluate(M, Xprior)
    inverse!(Xpost, F, M)

    @test norm(Xprior - Xpost)/norm(Xpost)<1e-6
    @test norm(Xprior - Xpost)<1e-6
    @test norm(evaluate(M, Xprior) - evaluate(M, Xpost))<1e-6
end


@testset "Test inversion of the Hermite Map II" begin

    Nx = 100
    Ny = 50
    m = 20
    Ne = 500
    Xprior = randn(Nx, Ne).*randn(Nx, Ne)
    Xpost = deepcopy(Xprior) .+ randn(Nx, Ne)

    M = HermiteMap(m, Xprior; diag = true)
    Ystar = deepcopy(Xpost[1:Ny,:])# + 0.05*randn(Ny,Ne);

    M = HermiteMap(m, Xprior)
    optimize(M, Xprior, 5; withconstant = false, start = Ny+1)

    F = evaluate(M, Xpost; start = Ny+1)
    inverse!(Xprior, F, M, Ystar; start = Ny+1, P = serial)

    @test norm(evaluate(M, Xprior; start = Ny+1)-evaluate(M, Xpost; start = Ny+1))/norm(evaluate(M, Xpost; start = Ny+1))<1e-6

    @test norm(Xprior[Ny+1:end,:] - Xpost[Ny+1:end,:])/norm(Xpost[Ny+1:end,:])<1e-6
end

@testset "Test hybrid inversion of the Hermite Map I" begin
    Nx = 100
    m = 20
    Ne = 500
    Xprior = randn(Nx, Ne).*randn(Nx, Ne)
    Xpost = deepcopy(Xprior) .+ randn(Nx, Ne)

    M = HermiteMap(m, Xprior; diag = true)
    optimize(M, Xprior, 5; withconstant = false)

    F = evaluate(M, Xprior)
    hybridinverse!(Xpost, F, M)

    @test norm(Xprior - Xpost)/norm(Xpost)<1e-6
    @test norm(Xprior - Xpost)<1e-6
    @test norm(evaluate(M, Xprior) - evaluate(M, Xpost))<1e-6
end


@testset "Test hybrid inversion of the Hermite Map II" begin

    Nx = 100
    Ny = 50
    m = 20
    Ne = 500
    Xprior = randn(Nx, Ne).*randn(Nx, Ne)
    Xpost = deepcopy(Xprior) .+ randn(Nx, Ne)

    M = HermiteMap(m, Xprior; diag = true)
    Ystar = deepcopy(Xpost[1:Ny,:])# + 0.05*randn(Ny,Ne);

    M = HermiteMap(m, Xprior)
    optimize(M, Xprior, 5; withconstant = false, start = Ny+1)

    F = evaluate(M, Xpost; start = Ny+1)
    inverse!(Xprior, F, M, Ystar; start = Ny+1, P = serial)

    @test norm(evaluate(M, Xprior; start = Ny+1)-evaluate(M, Xpost; start = Ny+1))/norm(evaluate(M, Xpost; start = Ny+1))<1e-6

    @test norm(Xprior[Ny+1:end,:] - Xpost[Ny+1:end,:])/norm(Xpost[Ny+1:end,:])<1e-6
end
