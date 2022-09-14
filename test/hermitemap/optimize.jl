using TransportBasedInference: ncoeff

@testset "Test optimization max_terms = nothing without/with QR" begin
    Nx = 2

    m = 20
    Ne = 500
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)

    L = LinearTransform(X; diag = true)
    transform!(L, X);

    idx = [0 0; 0 1; 0 2; 0 3; 1 1]
    coeff =  [   -0.7708538710735897;
              0.1875006767230035;
              1.419396079869706;
              0.2691388018878241;
             -2.9879004455954723];

    C = HermiteMapComponent(m, Nx, idx, coeff);
    C0 = deepcopy(C)

    C_noqr, error_noqr = TransportBasedInference.optimize(C, X, nothing; withqr = false, verbose = false);
    C_qr, error_qr = TransportBasedInference.optimize(C, X, nothing; withqr = true, verbose = false);


    S = Storage(C0.I.f, X)
    coeff0 = getcoeff(C0)
    precond = zeros(ncoeff(C0), ncoeff(C0))
    precond!(precond, coeff0, S, C0, X)

    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C0, X)), coeff0,
          Optim.LBFGS(; m = 20, P = Preconditioner(precond)))

    @test norm(getcoeff(C_noqr) - res.minimizer) <1e-6
    @test norm(getcoeff(C_qr) - res.minimizer) <1e-6

    error = res.minimum

    @test abs(error_noqr - error) < 1e-8
    @test abs(error_qr - error) < 1e-8

end


@testset "Test optimization max_terms is an integer without/with QR" begin
    Nx = 2

    m = 20
    Ne = 500
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)

    L = LinearTransform(X; diag = true)
    transform!(L, X);

    C = HermiteMapComponent(m, Nx);

    C_noqr, error_noqr = TransportBasedInference.optimize(C, X, 4; withqr = false, verbose = false);
    C_qr, error_qr = TransportBasedInference.optimize(C, X, 4; withqr = true, verbose = false);


    idx0 = getidx(C_noqr)

    C0 = HermiteMapComponent(m, Nx, idx0, zeros(size(idx0,1)));

    S = Storage(C0.I.f, X)
    coeff0 = getcoeff(C0)
    precond = zeros(ncoeff(C0), ncoeff(C0))
    precond!(precond, coeff0, S, C0, X)

    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, C0, X)), coeff0,
          Optim.LBFGS(; m = 20, P = Preconditioner(precond)))

    @test norm(getcoeff(C_noqr) - res.minimizer) <5e-5
    @test norm(getcoeff(C_qr) - res.minimizer) <5e-5

    error = res.minimum

    @test abs(error_noqr[end] - error) < 5e-5
    @test abs(error_qr[end] - error) < 5e-5

end


@testset "Test optimization max_terms is kfold without/with qr" begin

    Nx = 2

    m = 20
    Ne = 200
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)

    L = LinearTransform(X; diag = true)
    transform!(L, X)

    C = HermiteMapComponent(m, Nx)

    n_folds = 5
    folds = kfolds(1:size(X,2), k = n_folds)

    # Run greedy approximation
    max_iter = min(m-1,ceil(Int64, sqrt(size(X,2))))

    valid_error_noqr = zeros(max_iter+1, n_folds)
    valid_error_qr = zeros(max_iter+1, n_folds)

    @inbounds for i=1:n_folds
        idx_train, idx_valid = folds[i]
        @test size(idx_valid,1) == 40
        @test size(idx_train,1) == 160
        @test size(idx_train,1) < size(X,2)
        C_noqr, error_noqr = greedyfit(m, Nx, X[:,idx_train], X[:,idx_valid], max_iter; withqr = false, verbose  = false)
        C_qr, error_qr = greedyfit(m, Nx, X[:,idx_train], X[:,idx_valid], max_iter; withqr = true, verbose  = false)

        # error[2] contains the history of the validation error
        valid_error_noqr[:,i] .= deepcopy(error_noqr[2])
        valid_error_qr[:,i] .= deepcopy(error_qr[2])

    end

    mean_valid_error_noqr = mean(valid_error_noqr, dims  = 2)[:,1]
    mean_valid_error_qr = mean(valid_error_qr, dims  = 2)[:,1]

    @test size(mean_valid_error_noqr,1) == max_iter+1
    @test size(mean_valid_error_noqr,1) == max_iter+1

    value_noqr, opt_nterms_noqr = findmin(mean_valid_error_noqr)
    value_qr, opt_nterms_qr = findmin(mean_valid_error_qr)


    @test value_noqr == mean_valid_error_noqr[opt_nterms_noqr]
    @test value_qr == mean_valid_error_qr[opt_nterms_qr]

    # Run greedy fit up to opt_nterms on all the data
    C_opt_noqr, error_opt_noqr = greedyfit(m, Nx, X, opt_nterms_noqr; withqr = false, verbose  = false)
    C_opt_qr, error_opt_qr = greedyfit(m, Nx, X, opt_nterms_qr; withqr = false, verbose  = false)

    @test size(getcoeff(C_opt_noqr),1) == opt_nterms_noqr
    @test size(getcoeff(C_opt_qr),1) == opt_nterms_qr

    C_new_noqr, error_new_noqr = TransportBasedInference.optimize(C, X, "kfold"; withqr = false)
    C_new_qr, error_new_qr = TransportBasedInference.optimize(C, X, "kfold"; withqr = true)

    @test norm(getcoeff(C_opt_noqr) - getcoeff(C_new_noqr))<1e-5
    @test norm(error_opt_noqr - error_new_noqr)<1e-5

    @test norm(getcoeff(C_opt_qr) - getcoeff(C_new_qr))<1e-5
    @test norm(error_opt_qr - error_new_qr)<1e-5

end


@testset "Test optimization max_terms is split with/without QR" begin

    Nx = 2

    m = 20
    Ne = 200
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)

    L = LinearTransform(X; diag = true)
    transform!(L, X)

    C = HermiteMapComponent(m, Nx)

    nvalid = ceil(Int64, floor(0.2*size(X,2)))
    X_train = X[:,nvalid+1:end]
    X_valid = X[:,1:nvalid]

    @test size(X_train, 2) == 160
    @test size(X_valid, 2) == 40


    # Set maximum patience for optimization
    maxpatience = 20

    # Run greedy approximation
    max_iter = ceil(Int64, sqrt(size(X,2)))

    C_opt_noqr, error_opt_noqr = greedyfit(m, Nx, X_train, X_valid, max_iter;
                                           withqr = false, maxpatience = maxpatience,
                                           verbose  = false)
   C_opt_qr, error_opt_qr = greedyfit(m, Nx, X_train, X_valid, max_iter;
                                          withqr = true, maxpatience = maxpatience,
                                          verbose  = false)

    C_new_noqr, error_new_noqr = TransportBasedInference.optimize(C, X, "split"; withqr = false)

    C_new_qr, error_new_qr = TransportBasedInference.optimize(C, X, "split"; withqr = true)


    @test norm(getcoeff(C_opt_noqr) - getcoeff(C_new_noqr))<1e-8
    @test norm(error_opt_noqr[1] - error_new_noqr[1])<1e-8
    @test norm(error_opt_noqr[2] - error_new_noqr[2])<1e-8

    @test size(getcoeff(C_new_noqr),1) == max_iter

    @test norm(getcoeff(C_opt_qr) - getcoeff(C_new_qr))<1e-8
    @test norm(error_opt_qr[1] - error_new_qr[1])<1e-8
    @test norm(error_opt_qr[2] - error_new_qr[2])<1e-8

    @test size(getcoeff(C_new_qr),1) == max_iter
end
