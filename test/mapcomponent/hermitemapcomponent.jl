
@testset "Verify that initial map is identity" begin

  H = MapComponent(3, 2; α = 1e-6)

  x = randn(2,1)
  Hx = evaluate(H.I, x)
  @test abs(Hx[1] - x[2])<1e-10
end



@testset "Verify loss function and its gradient" begin

    Nx = 2
    Ne = 8
    m = 5
    ens = EnsembleState(Nx, Ne)

    ens.S .=  [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    B = MultiBasis(CstProHermite(3), Nx)

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
    H = MapComponent(R; α = 0.0)
    S = Storage(H.I.f, ens.S);

   res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, H, ens.S)), coeff, Optim.BFGS())
   coeff_opt = Optim.minimizer(res)

    @test norm(coeff_opt - [3.015753764546621;
                          -2.929908252283099;
                          -3.672401233483867;
                           2.975554571687243;
                           1.622308437415610])<1e-4

    # Verify with L-2 penalty term

    H = MapComponent(R; α = 0.1)
    S = Storage(H.I.f, ens.S);

    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, H, ens.S)), coeff, Optim.BFGS())
    coeff_opt = Optim.minimizer(res)

    @test norm(coeff_opt - [ -0.11411931034615422;
                             -0.21942146397348522;
                             -0.17368042128948974;
                              0.37348086659548607;
                              0.02434745831060741])<1e-4
end


@testset "Verify evaluation of HermiteMapComponent" begin

  Nx = 2
  Ne = 500
  ens = EnsembleState(Nx, Ne)

  ens.S .= randn(Nx, Ne)

  # ens.S .=  [0.267333   1.43021;
  #           0.364979   0.607224;
  #          -1.23693    0.249277;
  #          -2.0526     0.915629;
  #          -0.182465   0.415874;
  #           0.412907   1.01672;
  #           1.41332   -0.918205;
  #           0.766647  -1.00445]';
  B = MultiBasis(CstProHermite(6), Nx)

  idx = [0 0; 0 1; 1 0; 1 1; 1 2]
  truncidx = idx[1:2:end,:]
  Nψ = 5

  coeff = randn(Nψ)

  # coeff =   [0.6285037650645056;
  #  -0.4744029092496623;
  #   1.1405280011620331;
  #  -0.7217760771894809;
  #   0.11855056306742319]
  f = ExpandedFunction(B, idx, coeff)
  fp = ParametricFunction(f);
  R = IntegratedFunction(fp)

  C = MapComponent(R)

  # Test evaluate
  ψt = zeros(Ne)

  for i=1:Ne
      x = member(ens, i)
      ψt[i] = R.f.f(vcat(x[1:end-1], 0.0)) + quadgk(t->R.g(ForwardDiff.gradient(y->R.f.f(y), vcat(x[1:end-1],t))[end]), 0, x[end])[1]
  end

  ψ = evaluate(C, ens.S)

  @test norm(ψ - ψt)<1e-10
end



@testset "Verify log_pdf function" begin

  Nx = 2
  X  =  Matrix([0.267333   1.43021;
          0.364979   0.607224;
         -1.23693    0.249277;
         -2.0526     0.915629;
         -0.182465   0.415874;
          0.412907   1.01672;
          1.41332   -0.918205;
          0.766647  -1.00445]');
  B = MultiBasis(CstProHermite(6), Nx)

  idx = [0 0; 0 1; 1 0; 0 2; 2 0; 1 1]

  Nψ = 5

  coeff = [ -0.9905841755746164;
        0.6771992097558741;
       -2.243695806805015;
       -0.34591297359447354;
       -1.420159186008486;
       -0.5361337327704369]
  f = ExpandedFunction(B, idx, coeff)
  C = MapComponent(f)

  @test norm(log_pdf(C, X) - [  -1.572509004118956
                                -2.870725221050853
                                -1.285696671132943
                                -1.088115266085997
                                -2.396567575345843
                                -2.238446803642176
                                -5.999143198546611
                                -7.082679248037675])<1e-8
end

@testset "Verify grad_log_pdf function Nx = 1" begin

  Nx = 1
  Ne = 100
  X = randn(Nx, Ne)
  ens = EnsembleState(X)
  m = 10
  B = MultiBasis(CstProHermite(3), Nx)

  idx = reshape([0; 1; 2; 3], (4,1))

  coeff =  randn(size(idx,1))

  C = MapComponent(m, Nx, idx, coeff)

  dxlogC = grad_x_log_pdf(C, X)

  function evaluatef0(x)
    y = copy(x)
    y[end] = 0.0
    return C.I.f.f(y)
  end

  integrand(t,x) = C.I.g(ForwardDiff.gradient(y->C.I.f.f(y), vcat(x[1:end-1],t*x[end]))[end])

  function Ct(x)
      lb = 0.0
      ub = 1.0
      prob = QuadratureProblem(integrand,lb,ub,x)
      out = evaluatef0(x) + x[end]*solve(prob,CubatureJLh(),reltol=1e-6,abstol=1e-6)[1]
      return out
  end

  log_pdfCt(x) = log_pdf(Ct(x)) + log(C.I.g(ForwardDiff.gradient(z->C.I.f.f(z),x)[end]))

  @inbounds for i=1:Ne
    @test norm(ForwardDiff.gradient(log_pdfCt, member(ens,i)) - dxlogC[i,:])<1e-8
  end

end


@testset "Verify grad_log_pdf function Nx = 2" begin

  Nx = 2
  Ne = 100
  X = randn(Nx, Ne)
  ens = EnsembleState(X)
  m = 10
  B = MultiBasis(CstProHermite(3), Nx)

  idx = [0 0; 0 1; 1 0; 1 1; 1 2; 3 2]

  coeff =  randn(size(idx,1))

  C = MapComponent(m, Nx, idx, coeff)

  dxlogC = grad_x_log_pdf(C, X)

  function evaluatef0(x)
    y = copy(x)
    y[end] = 0.0
    return C.I.f.f(y)
  end

  integrand(t,x) = C.I.g(ForwardDiff.gradient(y->C.I.f.f(y), vcat(x[1:end-1],t*x[end]))[end])

  function Ct(x)
      lb = 0.0
      ub = 1.0
      prob = QuadratureProblem(integrand,lb,ub,x)
      out = evaluatef0(x) + x[end]*solve(prob,CubatureJLh(),reltol=1e-6,abstol=1e-6)[1]
      return out
  end

  log_pdfCt(x) = log_pdf(Ct(x)) + log(C.I.g(ForwardDiff.gradient(z->C.I.f.f(z),x)[end]))

  @inbounds for i=1:Ne
    @test norm(ForwardDiff.gradient(log_pdfCt, member(ens,i)) - dxlogC[i,:])<1e-8
  end

end


@testset "Verify grad_log_pdf function Nx = 4" begin

  Nx = 4
  Ne = 100
  X = randn(Nx, Ne)
  ens = EnsembleState(X)
  m = 10
  B = MultiBasis(CstProHermite(3), Nx)

  idx = [0 0 0 0; 0 2 0 1; 0 2 3 0; 2 0 2 1; 0 0 1 2; 1 2 0 2;3 2 2 2]

  coeff =  randn(size(idx,1))

  C = MapComponent(m, Nx, idx, coeff)

  dxlogC = grad_x_log_pdf(C, X)

  function evaluatef0(x)
    y = copy(x)
    y[end] = 0.0
    return C.I.f.f(y)
  end

  integrand(t,x) = C.I.g(ForwardDiff.gradient(y->C.I.f.f(y), vcat(x[1:end-1],t*x[end]))[end])

  function Ct(x)
      lb = 0.0
      ub = 1.0
      prob = QuadratureProblem(integrand,lb,ub,x)
      out = evaluatef0(x) + x[end]*solve(prob,CubatureJLh(),reltol=1e-6,abstol=1e-6)[1]
      return out
  end

  log_pdfCt(x) = log_pdf(Ct(x)) + log(C.I.g(ForwardDiff.gradient(z->C.I.f.f(z),x)[end]))

  @inbounds for i=1:Ne
    @test norm(ForwardDiff.gradient(log_pdfCt, member(ens,i)) - dxlogC[i,:])<1e-8
  end

end
# Code for optimization with Hessian
# X = randn(Nx, Ne) .* randn(Nx, Ne)
# S = Storage(H.I.f, X; hess = true);
#
# J = 0.0
# dJ = zeros(Nψ)
# d2J = zeros(Nψ, Nψ)
# hess_negative_log_likelihood!(J, dJ, d2J, coeff, S, H, X)
#
# res = Optim.optimize(Optim.only_fgh!(hess_negative_log_likelihood!($S, $H, $X)), $coeff, Optim.NewtonTrustRegion())
