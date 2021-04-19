
@testset "Test storage and update_storage!" begin


    Nx = 2
    Ne = 8

    X    =  [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';

    Blist = [CstProHermite(8); CstPhyHermite(8); CstLinProHermite(8); CstLinPhyHermite(8)]

    for b in Blist

      B = MultiBasis(b, Nx)


      idx = [0 0; 0 1; 1 0; 1 1; 1 2]
      truncidx = idx[1:2:end,:]
      Nψ = 5

      coeff =  [0.6285037650645056;
       -0.4744029092496623;
        1.1405280011620331;
       -0.7217760771894809;
        0.11855056306742319]
      f = ExpandedFunction(B, idx, coeff)
      R = IntegratedFunction(f)

      H = MapComponent(R; α = 1e-2)

      S = Storage(H.I.f, X);

      @test norm(S.ψoff     -  evaluate_offdiagbasis(f, X)) < 1e-8
      @test norm(S.ψoffψd   -  evaluate_offdiagbasis(f, X) .*
                               evaluate_diagbasis(f, X)) < 1e-8
      @test norm(S.ψoffψd0  -  evaluate_offdiagbasis(f, X) .*
                               repeated_evaluate_basis(f, zeros(Ne))) < 1e-8
      @test norm(S.ψoffdψxd -  evaluate_offdiagbasis(f, X) .*
                               repeated_grad_xk_basis(f, X[end,:])) < 1e-8
      @test norm(S.ψnorm - 1/sqrt(Ne)*norm.(eachcol(evaluate_basis(f, X)))) < 1e-8

       #test cache dimension
      @test size(S.cache_dcψxdt) == (Ne, Nψ)
      @test size(S.cache_dψxd) == (Ne,)
      @test size(S.cache_integral) == (Ne + Ne * Nψ,)
      @test size(S.cache_g) == (Ne,)


      # Test add new components via update_storage!
      addedidx = [2 1; 2 2; 2 3; 3 2]

      S = update_storage(S, X, addedidx)

      addednψ = size(addedidx,1)
      fnew = ExpandedFunction(f.B, vcat(f.idx, addedidx), vcat(f.coeff, zeros(addednψ)))

      @test norm(S.ψoff     -  evaluate_offdiagbasis(fnew, X)) < 1e-8
      @test norm(S.ψoffψd   -  evaluate_offdiagbasis(fnew, X) .*
                               evaluate_diagbasis(fnew, X)) < 1e-8
      @test norm(S.ψoffψd0  -  evaluate_offdiagbasis(fnew, X) .*
                               repeated_evaluate_basis(fnew, zeros(Ne))) < 1e-8
      @test norm(S.ψoffdψxd -  evaluate_offdiagbasis(fnew, X) .*
                               repeated_grad_xk_basis(fnew, X[end,:])) < 1e-8
      @test norm(S.ψnorm - 1/sqrt(Ne)*norm.(eachcol(evaluate_basis(fnew, X)))) < 1e-8

      newNψ = addednψ + Nψ
      #test cache dimension
     @test size(S.cache_dcψxdt) == (Ne, newNψ)
     @test size(S.cache_dψxd) == (Ne,)
     @test size(S.cache_integral) == (Ne + Ne * newNψ,)
     @test size(S.cache_g) == (Ne,)
   end
end
