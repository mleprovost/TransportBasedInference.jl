

@testset "Test optimization max_terms = nothing" begin
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

    Hk = HermiteMapk(m, Nx, idx, coeff);
    Hk0 = deepcopy(Hk)

    Hk_new, error_new = AdaptiveTransportMap.optimize(Hk, X, nothing; verbose = false);


    S = Storage(Hk0.I.f, X)
    coeff0 = getcoeff(Hk0)
    precond = zeros(ncoeff(Hk0), ncoeff(Hk0))
    precond!(precond, coeff0, S, Hk0, X)

    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood!(S, Hk0, X)), coeff0,
          Optim.LBFGS(; m = 20, P = Preconditioner(precond)))

    @test norm(getcoeff(Hk_new) - res.minimizer) <1e-6

    error = res.minimum

    @test abs(error_new - error) < 1e-8
end

@testset "Test optimization max_terms is an integer" begin
    Nx = 2

    m = 20
    Ne = 500
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)

    L = LinearTransform(X; diag = true)
    transform!(L, X);

    Hk = HermiteMapk(m, Nx);

    Hk_new, error_new = AdaptiveTransportMap.optimize(Hk, X, 4; verbose = false);

    idx0 = getidx(Hk_new)

    Hk0 = HermiteMapk(m, Nx, idx0, zeros(size(idx0,1)));

    S = Storage(Hk0.I.f, X)
    coeff0 = getcoeff(Hk0)
    precond = zeros(ncoeff(Hk0), ncoeff(Hk0))
    precond!(precond, coeff0, S, Hk0, X)

    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood!(S, Hk0, X)), coeff0,
          Optim.LBFGS(; m = 20, P = Preconditioner(precond)))

    @test norm(getcoeff(Hk_new) - res.minimizer) <1e-6

    error = res.minimum

    @test abs(error_new[end] - error) < 1e-8
end


@testset "Test optimization max_terms is kfold" begin

    Nx = 2

    m = 20
    Ne = 200
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)

    L = LinearTransform(X; diag = true)
    transform!(L, X)

    Hk = HermiteMapk(m, Nx)

    n_folds = 5
    folds = kfolds(1:size(X,2), k = n_folds)

    # Run greedy approximation
    max_iter = ceil(Int64, sqrt(size(X,2)))

    valid_error = zeros(max_iter, n_folds)
    @inbounds for i=1:n_folds
        idx_train, idx_valid = folds[i]
        @test size(idx_valid,1) == 40
        @test size(idx_train,1) == 160
        @test size(idx_train,1) < size(X,2)
        Hk, error = greedyfit(m, Nx, X[:,idx_train], X[:,idx_valid], max_iter; verbose  = false)

        # error[2] contains the histroy of the validation error
        valid_error[:,i] .= deepcopy(error[2])
    end

    mean_valid_error = mean(valid_error, dims  = 2)[:,1]

    @test size(mean_valid_error,1) == max_iter

    value, opt_nterms = findmin(mean_valid_error)

    @test value == mean_valid_error[opt_nterms]

    # Run greedy fit up to opt_nterms on all the data
    Hk_opt, error_opt = greedyfit(m, Nx, X, opt_nterms; verbose  = false)

    @test size(getcoeff(Hk_opt),1) == opt_nterms

    Hk_new, error_new = AdaptiveTransportMap.optimize(Hk, X, "kfold")

    @test norm(getcoeff(Hk_opt) - getcoeff(Hk_new))<1e-5
    @test norm(error_opt - error_new)<1e-5

end


@testset "Test optimization max_terms is split" begin

    Nx = 2

    m = 20
    Ne = 200
    X = randn(Nx, Ne).^2 + 0.1*randn(Nx, Ne)

    L = LinearTransform(X; diag = true)
    transform!(L, X)

    Hk = HermiteMapk(m, Nx)

    nvalid = ceil(Int64, floor(0.2*size(X,2)))
    X_train = X[:,nvalid+1:end]
    X_valid = X[:,1:nvalid]

    @test size(X_train, 2) == 160
    @test size(X_valid, 2) == 40


    # Set maximum patience for optimization
    maxpatience = 20

    # Run greedy approximation
    max_iter = ceil(Int64, sqrt(size(X,2)))

    Hk_opt, error_opt = greedyfit(m, Nx, X_train, X_valid, max_iter;
                                   maxpatience = maxpatience, verbose  = false)

    Hk_new, error_new = AdaptiveTransportMap.optimize(Hk, X, "split")

    @test norm(getcoeff(Hk_opt) - getcoeff(Hk_new))<1e-8
    @test norm(error_opt[1] - error_new[1])<1e-8
    @test norm(error_opt[2] - error_new[2])<1e-8

    @test size(getcoeff(Hk_new),1) == max_iter
end
