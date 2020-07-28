
@testset "Test function update_coeffs I" begin

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
    # Initialize map with zero coefficients
    Hk_old = HermiteMapk(m, Nx; α = 1e-6);

    setcoeff!(Hk_old, [1.5])
    reduced_margin = getreducedmargin(getidx(Hk_old))
    @show reduced_margin
    idx_new = vcat(getidx(Hk_old), reduced_margin)

    Hk_new = HermiteMapk(m, Nx, idx_new, vcat(getcoeff(Hk_old), zeros(2)));

    coeff_new, coeff_idx_added, idx_added = update_coeffs(Hk_old, Hk_new)

    @test norm(coeff_new - [1.5; 0.0; 0.0])<1e-8
    @test coeff_idx_added == [2; 3]
    @test idx_added == [0 1; 1 0]
end

@testset "Test function update_coeffs II" begin

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
    # Initialize map with zero coefficients
    Hk_old = HermiteMapk(m, Nx, [0 0; 1 0; 2 0; 0 1; 0 2; 1 1], randn(6); α = 1e-6);

    reduced_margin = getreducedmargin(getidx(Hk_old))
    idx_new = vcat(getidx(Hk_old), reduced_margin)
    Hk_new = HermiteMapk(m, Nx, idx_new, vcat(getcoeff(Hk_old), zeros(4)));
    coeff_new, coeff_idx_added, idx_added = update_coeffs(Hk_old, Hk_new)

    @test norm(coeff_new - vcat(getcoeff(Hk_old), zeros(4)))<1e-8
    @test coeff_idx_added == [7; 8; 9; 10]
    @test idx_added == [0 3; 1 2; 2 1; 3 0]
end


@testset "Test function update_coeffs III" begin
    Nx = 3
    m = 5
    # Initialize map with zero coefficients
    Hk_old = HermiteMapk(m, Nx, [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 2 0], randn(5); α = 1e-6);

    reduced_margin = getreducedmargin(getidx(Hk_old))
    idx_new = vcat(getidx(Hk_old), reduced_margin)
    Hk_new = HermiteMapk(m, Nx, idx_new, vcat(getcoeff(Hk_old), zeros(4)));
    coeff_new, coeff_idx_added, idx_added = update_coeffs(Hk_old, Hk_new)

    @test norm(coeff_new - vcat(getcoeff(Hk_old), zeros(4)))<1e-8
    @test coeff_idx_added == [6; 7; 8; 9]
    @test idx_added == reduced_margin
end

@testset "Test update_component I" begin
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
    X = ens.S
    # Initialize map with zero coefficients
    Hk_old = HermiteMapk(m, Nx; α = 1e-6);

    S = Storage(Hk_old.I.f, X)

    dJ_old = zeros(1)
    J_old = 0.0
    J_old = negative_log_likelihood!(J_old, dJ_old, getcoeff(Hk_old), S, Hk_old, X)

    @test abs(J_old - 1.317277788545110)<1e-10

    @test norm(dJ_old - [0.339034875000000])<1e-10


    # setcoeff!(Hk_old, [1.5])
    reduced_margin0 = getreducedmargin(getidx(Hk_old))
    idx_old0 = getidx(Hk_old)
    idx_new0 = vcat(idx_old0, reduced_margin0)

    # Define updated map
    f_new = ExpandedFunction(Hk_old.I.f.f.B, idx_new0, vcat(getcoeff(Hk_old), zeros(size(reduced_margin0,1))))
    Hk_new = HermiteMapk(f_new; α = 1e-6)
    idx_new, reduced_margin = update_component(Hk_old, X, reduced_margin0, S)

    dJ_new = zeros(3)
    J_new = 0.0
    S = update_storage(S, X, reduced_margin0)
    J_new = negative_log_likelihood!(J_new, dJ_new, getcoeff(Hk_new), S, Hk_new, X)


    @test abs(J_new -    1.317277788545110)<1e-8

    @test norm(dJ_new - [0.339034875000000;   0.228966410748600;   0.192950409869679])<1e-8


    @test findmax(abs.(dJ_new[2:3]))[2] == 2-1

    @test idx_new0[3,:] == reduced_margin0[2,:]

    @test reduced_margin == updatereducedmargin(getidx(Hk_old), reduced_margin0, 1)[2]
    @test idx_new == updatereducedmargin(getidx(Hk_old), reduced_margin0, 1)[1]
end

@testset "Test update_component II" begin
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
    X = ens.S

    idx = [0 0; 0 1; 1 0; 0 2; 2 0; 1 1]

    coeff = [ -0.9905841755746164;
          0.6771992097558741;
         -2.243695806805015;
         -0.34591297359447354;
         -1.420159186008486;
         -0.5361337327704369]

    Hk_old = HermiteMapk(m, Nx, idx, coeff; α = 1e-6);

    S = Storage(Hk_old.I.f, X)

    dJ_old = zeros(6)
    J_old = 0.0
    J_old = negative_log_likelihood!(J_old, dJ_old, getcoeff(Hk_old), S, Hk_old, X)

    @test abs(J_old - 3.066735373495125)<1e-5

    dJt_old  = [-1.665181693060798;
                -0.919442416275569;
                -0.900922248322526;
                -0.119522662066802;
                -0.472794468187974;
                -0.496862354324658]

    @test norm(dJ_old - dJt_old)<1e-5



    reduced_margin0 = getreducedmargin(getidx(Hk_old))
    idx_old0 = getidx(Hk_old)
    idx_new0 = vcat(idx_old0, reduced_margin0)

    # Define updated map
    f_new = ExpandedFunction(Hk_old.I.f.f.B, idx_new0, vcat(getcoeff(Hk_old), zeros(size(reduced_margin0,1))))
    Hk_new = HermiteMapk(f_new; α = 1e-6)
    idx_new, reduced_margin = update_component(Hk_old, X, reduced_margin0, S)

    dJ_new = zeros(10)
    J_new = 0.0
    S = update_storage(S, X, reduced_margin0)
    J_new = negative_log_likelihood!(J_new, dJ_new, getcoeff(Hk_new), S, Hk_new, X)

     Jt_new  = 3.066735373495125
    dJt_new  = [-1.665181693060798;
                -0.919442416275569;
                -0.900922248322526;
                -0.119522662066802;
                -0.472794468187974;
                -0.496862354324658;
                 0.338831623096620;
                -0.083502724667314;
                -0.281130533740555;
                 0.271587190562667]

    @test abs(J_new -    Jt_new)<1e-5
    @test norm(dJ_new - dJt_new)<1e-5


    @test findmax(abs.(dJ_new[7:10]))[2] == 7-6

    @test idx_new0[8,:] == reduced_margin0[2,:]

    @test reduced_margin == updatereducedmargin(getidx(Hk_old), reduced_margin0, 1)[2]
    @test idx_new == updatereducedmargin(getidx(Hk_old), reduced_margin0, 1)[1]
end

@testset "Test greedy optimization" begin

Nx = 2
m = 10
Ne = 10^3
X = randn(2, Ne).^2 + 0.1*randn(2, Ne)
L = LinearTransform(X)
transform!(L, X)

X_train = X[:,1:800]
X_valid = X[:,801:1000]


Hk_new, error = greedyfit(m, Nx, X_train, X_valid, 10; verbose = false);

train_error, valid_error = error

Hk_test = deepcopy(Hk_new)
setcoeff!(Hk_test, zero(getcoeff(Hk_new)));

S_test = Storage(Hk_test.I.f, X_train)
coeff_test = getcoeff(Hk_test)

res = Optim.optimize(Optim.only_fg!(negative_log_likelihood!(S_test, Hk_test, X_train)), coeff_test, Optim.BFGS())

@test norm(getcoeff(Hk_new)-Optim.minimizer(res))<1e-7

@test norm(train_error[end] - res.minimum)<1e-7

end
