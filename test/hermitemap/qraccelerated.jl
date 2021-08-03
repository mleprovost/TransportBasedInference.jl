import TransportBasedInference: ncoeff

@testset "Test updateQRscaling" begin
    Nx = 3
    Ne = 8
    m = 20

    idx = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 0 1 2; 1 0 0]


    Nψ = 6
    coeff = [ 0.20649582065364197;
             -0.5150990160472986;
              2.630096893080717;
              1.13653076177397;
              0.6725837371023421;
             -1.3126095306624133]
    C = HermiteMapComponent(m, Nx, idx, coeff; α = 1e-6);

    Ne = 100


    # The QR decomposition is not unique!

    X = randn(Nx, Ne) .* randn(Nx, Ne) + cos.(randn(Nx, Ne)) .* exp.(-randn(Nx, Ne).^2)

    L = LinearTransform(X)
    transform!(L, X)
    S = Storage(C.I.f, X)
    F = QRscaling(S)
    newidx = [1 1 1]

    Snew = update_storage(S, X, newidx)
    Fupdated = updateQRscaling(F, Snew)

    Fnew = QRscaling(Snew)

    @test norm(Fupdated.D - Fnew.D)<1e-6
    @test norm(Fupdated.Dinv - Fnew.Dinv)<1e-6

    @test norm(Fupdated.R.data'*Fupdated.R.data - Fnew.R.data'*Fnew.R.data)<1e-6

    @test norm(inv(Fupdated.R) - Fupdated.Rinv.data)<1e-6
    @test norm(inv(Fupdated.U) - Fupdated.Uinv.data)<1e-6

    @test norm(Fupdated.L2Uinv - Fupdated.Uinv'*Fupdated.Uinv)<1e-6

end

@testset "Test evaluation of negative log likelihood with QR basis" begin

    Nx = 3
    Ne = 8
    m = 20

    idx = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 0 1 2; 1 0 0]

    Nψ = 6
    coeff = [ 0.20649582065364197;
             -0.5150990160472986;
              2.630096893080717;
              1.13653076177397;
              0.6725837371023421;
             -1.3126095306624133]
    coeff0 = deepcopy(coeff)

    X =  Matrix([ 1.12488     0.0236348   -0.958426;
             -0.0493163   0.00323509  -0.276744;
              1.11409     0.976117     0.256577;
             -0.563545    0.179956    -0.418444;
              0.0780599   0.371091    -0.742342;
              1.77185    -0.175635     0.32151;
             -0.869045   -0.0570977   -1.06254;
             -0.165249   -2.70636      0.548725]')

    L = LinearTransform(X)
    transform!(L, X);

    # For α = 0.0
    C = HermiteMapComponent(m, Nx, idx, coeff; α = 0.0)

    S = Storage(C.I.f, X)
    S̃ = deepcopy(S)
    F = QRscaling(S̃)


    #Verify loss function
    # Without QR normalization
    J = 0.0
    dJ = zeros(Nψ)
    J = negative_log_likelihood!(J, dJ, coeff, S, C, X)

    # In QR space
    J̃ = 0.0
    dJ̃ = zeros(Nψ)
    c̃oeff0 = F.U*coeff0

    # mul!(S̃.ψoffψd0, S̃.ψoffψd0, F.Uinv)
    # mul!(S̃.ψoffdψxd, S̃.ψoffdψxd, F.Uinv)

    J̃ = qrnegative_log_likelihood!(J̃, dJ̃, c̃oeff0, F, S̃, C, X)

    @test abs(J - J̃)<1e-5
    # Chain's rule applies!
    @test norm(F.Uinv'*dJ - dJ̃)<1e-5


    # For α = 1.0
    C = HermiteMapComponent(m, Nx, idx, coeff; α = 1.0)

    S = Storage(C.I.f, X)
    S̃ = deepcopy(S)
    F = QRscaling(S̃)


    #Verify loss function
    # Without QR normalization
    J = 0.0
    dJ = zeros(Nψ)
    J = negative_log_likelihood!(J, dJ, coeff, S, C, X)

    # In QR space
    J̃ = 0.0
    dJ̃ = zeros(Nψ)
    c̃oeff0 = F.U*coeff0

    # mul!(S̃.ψoffψd0, S̃.ψoffψd0, F.Uinv)
    # mul!(S̃.ψoffdψxd, S̃.ψoffdψxd, F.Uinv)

    J̃ = qrnegative_log_likelihood!(J̃, dJ̃, c̃oeff0, F, S̃, C, X)

    @test abs(J - J̃)<1e-5
    # Chain's rule applies!
    @test norm(F.Uinv'*dJ - dJ̃)<1e-5
end

@testset "Test optimization with QR basis and Hessian preconditioner without/with L2 penalty" begin

    Nx = 3
    Ne = 8
    m = 20

    idx = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 0 1 2; 1 0 0]

    Nψ = 6
    coeff = [ 0.20649582065364197;
             -0.5150990160472986;
              2.630096893080717;
              1.13653076177397;
              0.6725837371023421;
             -1.3126095306624133]
    coeff0 = deepcopy(coeff)

    X =  Matrix([ 1.12488     0.0236348   -0.958426;
             -0.0493163   0.00323509  -0.276744;
              1.11409     0.976117     0.256577;
             -0.563545    0.179956    -0.418444;
              0.0780599   0.371091    -0.742342;
              1.77185    -0.175635     0.32151;
             -0.869045   -0.0570977   -1.06254;
             -0.165249   -2.70636      0.548725]')

    L = LinearTransform(X)
    transform!(L, X);

    # For α = 0.0
    C = HermiteMapComponent(m, Nx, idx, coeff; α = 0.0)

    S = Storage(C.I.f, X)
    S̃ = deepcopy(S)
    F = QRscaling(S̃)

    precond = zeros(ncoeff(C), ncoeff(C))
    precond!(precond, coeff, S, C, X)


    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff, Optim.LBFGS(; m = 20))

    # In QR space
    c̃oeff0 = F.U*coeff0

    # mul!(S̃.ψoffψd0, S̃.ψoffψd0, F.Uinv)
    # mul!(S̃.ψoffdψxd, S̃.ψoffdψxd, F.Uinv)

    r̃es = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S̃, C, X)), c̃oeff0, Optim.LBFGS(; m = 20))

    # With QR and Hessian approximation

    qrprecond = zeros(ncoeff(C), ncoeff(C))
    qrprecond!(qrprecond, c̃oeff0, F, S̃, C, X)

    r̃esprecond = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S̃, C, X)), c̃oeff0,
                           Optim.LBFGS(; m = 20, P = Preconditioner(qrprecond)))

    @test norm(res.minimizer - F.Uinv*r̃es.minimizer)<1e-5
    @test norm(res.minimizer - F.Uinv*r̃esprecond.minimizer)<1e-5
    @test norm(r̃es.minimizer - r̃esprecond.minimizer)<1e-5

    # Verify relation between the preconditioner express in the unscaled and QR spaces
    # Hqr = U^{-T} Hunscaled U^{-1}
    @test norm(qrprecond - F.Uinv'*precond*F.Uinv)<1e-5

    @test norm(coeff - coeff0)<1e-5

    # For α = 1e-1
    C = HermiteMapComponent(m, Nx, idx, coeff; α = 0.1)

    S = Storage(C.I.f, X)
    S̃ = deepcopy(S)
    F = QRscaling(S̃)

    precond = zeros(ncoeff(C), ncoeff(C))
    precond!(precond, coeff, S, C, X)


    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C, X)), coeff, Optim.LBFGS(; m = 20))

    # With QR
    c̃oeff0 = F.U*coeff0

    # mul!(S̃.ψoffψd0, S̃.ψoffψd0, F.Uinv)
    # mul!(S̃.ψoffdψxd, S̃.ψoffdψxd, F.Uinv)

    r̃es = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S̃, C, X)), c̃oeff0, Optim.LBFGS(; m = 20))

    @test norm(res.minimizer - F.Uinv*r̃es.minimizer)<1e-5

    # With QR and Hessian approximation

    qrprecond = zeros(ncoeff(C), ncoeff(C))
    qrprecond!(qrprecond, c̃oeff0, F, S̃, C, X)

    r̃esprecond = Optim.optimize(Optim.only_fg!(qrnegative_log_likelihood(F, S̃, C, X)), c̃oeff0,
                           Optim.LBFGS(; m = 20, P = Preconditioner(qrprecond)))

    @test norm(res.minimizer - F.Uinv*r̃esprecond.minimizer)<1e-5
    @test norm(r̃es.minimizer - r̃esprecond.minimizer)<1e-5

    # Verify relation between the preconditioner express in the unscaled and QR spaces
    # Hqr = U^{-T} Hunscaled U^{-1}
    @test norm(qrprecond - F.Uinv'*precond*F.Uinv)<1e-5

    @test norm(coeff - coeff0)<1e-5
end
