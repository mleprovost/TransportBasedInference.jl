
@testset "Verify that initial map is identity" begin

  H = HermiteMapk(3, 2; α = 1e-6)

  x = randn(2,1)
  Hx = evaluate(H.I, x)
  @test abs(Hx[1] - x[2])<1e-10
end



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

    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood!(S, H, ens.S)), coeff, Optim.BFGS())
    coeffopt = Optim.minimizer(res)

    @test norm(coeffopt - [3.015753764546621;
                          -2.929908252283099;
                          -3.672401233483867;
                           2.975554571687243;
                           1.622308437415610])<1e-4

    # Verify with L-2 penalty term

    H = HermiteMapk(R; α = 1e-2)
    S = Storage(H.I.f, ens.S);

    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood!(S, H, ens.S)), coeff, Optim.BFGS())
    coeffopt = Optim.minimizer(res)

    @test norm(coeffopt - [0.550368190586868;
                          -0.823576312038818;
                          -0.974273115827102;
                           1.552794230307129;
                           0.414753530952308])<1e-4
end
