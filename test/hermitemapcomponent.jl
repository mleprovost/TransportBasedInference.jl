

@testset "Verify loss function and its gradient" begin

    Nx = 2
    Ne = 8
    ens = EnsembleState(Nx, Ne)

    ens.S .=  [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    B = MultiBasis(CstProHermite(3; scaled =true), Nx)

    idx = [0 0; 0 1; 1 0; 2 1; 1 2]
    truncidx = idx[1:2:end,:]
    Nψ = 5

    coeff =   [0.6285037650645056;
     -0.4744029092496623;
      1.1405280011620331;
     -0.7217760771894809;
      0.11855056306742319]
    f = ExpandedFunction(B, idx, coeff)

    fp = ParametricFunction(f)
    R = IntegratedFunction(fp)
    H = HermiteMapk(R)
    S = Storage(H.I.f, ens.S);

    J, dJ = negative_log_likelihood!(S,H, ens.S)

    @test abs(J - 2.137129544927313)<1e-8
    @test norm(dJ - [1.254209297811173; 0.752759343086777; 0.669152523388112; -0.073354929658946; 0.071051667979605])<1e-8







end
