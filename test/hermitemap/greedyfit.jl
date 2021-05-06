
@testset "Test function update_coeffs I" begin

    Nx = 2
    Ne = 8
    m = 5
    X = zeros(Nx, Ne)

    X .=  [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    # Initialize map with zero coefficients
    C_old = HermiteMapComponent(m, Nx; α = 1e-6);

    setcoeff!(C_old, [1.5])
    reduced_margin = getreducedmargin(getidx(C_old))
    @show reduced_margin
    idx_new = vcat(getidx(C_old), reduced_margin)

    C_new = HermiteMapComponent(m, Nx, idx_new, vcat(getcoeff(C_old), zeros(2)));

    coeff_new, coeff_idx_added, idx_added = update_coeffs(C_old, C_new)

    @test norm(coeff_new - [1.5; 0.0; 0.0])<1e-8
    @test coeff_idx_added == [2; 3]
    @test idx_added == [0 1; 1 0]
end

@testset "Test function update_coeffs II" begin

    Nx = 2
    Ne = 8
    m = 5
    X = zeros(Nx, Ne)

    X    .=  [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    # Initialize map with zero coefficients
    C_old = HermiteMapComponent(m, Nx, [0 0; 1 0; 2 0; 0 1; 0 2; 1 1], randn(6); α = 1e-6);

    reduced_margin = getreducedmargin(getidx(C_old))
    idx_new = vcat(getidx(C_old), reduced_margin)
    C_new = HermiteMapComponent(m, Nx, idx_new, vcat(getcoeff(C_old), zeros(4)));
    coeff_new, coeff_idx_added, idx_added = update_coeffs(C_old, C_new)

    @test norm(coeff_new - vcat(getcoeff(C_old), zeros(4)))<1e-8
    @test coeff_idx_added == [7; 8; 9; 10]
    @test idx_added == [0 3; 1 2; 2 1; 3 0]
end


@testset "Test function update_coeffs III" begin
    Nx = 3
    m = 5
    # Initialize map with zero coefficients
    C_old = HermiteMapComponent(m, Nx, [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 2 0], randn(5); α = 1e-6);

    reduced_margin = getreducedmargin(getidx(C_old))
    idx_new = vcat(getidx(C_old), reduced_margin)
    C_new = HermiteMapComponent(m, Nx, idx_new, vcat(getcoeff(C_old), zeros(4)));
    coeff_new, coeff_idx_added, idx_added = update_coeffs(C_old, C_new)

    @test norm(coeff_new - vcat(getcoeff(C_old), zeros(4)))<1e-8
    @test coeff_idx_added == [6; 7; 8; 9]
    @test idx_added == reduced_margin
end

@testset "Test update_component I" begin
    Nx = 2
    Ne = 8
    m = 5
    X = zeros(Nx, Ne)

    X .=  [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    X = X
    # Initialize map with zero coefficients
    C_old = HermiteMapComponent(m, Nx; α = 1e-6);

    S = Storage(C_old.I.f, X)

    dJ_old = zeros(1)
    J_old = 0.0
    J_old = negative_log_likelihood!(J_old, dJ_old, getcoeff(C_old), S, C_old, X)

    @test abs(J_old - 1.317277788545110)<1e-10

    @test norm(dJ_old - [0.339034875000000])<1e-10


    # setcoeff!(C_old, [1.5])
    reduced_margin0 = getreducedmargin(getidx(C_old))
    idx_old0 = getidx(C_old)
    idx_new0 = vcat(idx_old0, reduced_margin0)

    # Define updated map
    f_new = ExpandedFunction(C_old.I.f.B, idx_new0, vcat(getcoeff(C_old), zeros(size(reduced_margin0,1))))
    C_new = HermiteMapComponent(f_new; α = 1e-6)
    idx_new, reduced_margin = update_component!(C_old, X, reduced_margin0, S)

    dJ_new = zeros(3)
    J_new = 0.0
    S = update_storage(S, X, reduced_margin0)
    J_new = negative_log_likelihood!(J_new, dJ_new, getcoeff(C_new), S, C_new, X)


    @test abs(J_new -    1.317277788545110)<1e-8

    @test norm(dJ_new - [0.339034875000000;   0.228966410748600;   0.192950409869679])<1e-8


    @test findmax(abs.(dJ_new[2:3]))[2] == 2-1

    @test idx_new0[3,:] == reduced_margin0[2,:]

    @test reduced_margin == updatereducedmargin(getidx(C_old), reduced_margin0, 1)[2]
    @test idx_new == updatereducedmargin(getidx(C_old), reduced_margin0, 1)[1]
end

@testset "Test update_component II" begin
    Nx = 2
    Ne = 8
    m = 5
    X = zeros(Nx, Ne)

    X .=  [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    X = X

    idx = [0 0; 0 1; 1 0; 0 2; 2 0; 1 1]

    coeff = [ -0.9905841755746164;
          0.6771992097558741;
         -2.243695806805015;
         -0.34591297359447354;
         -1.420159186008486;
         -0.5361337327704369]

    C_old = HermiteMapComponent(m, Nx, idx, coeff; α = 1e-6);

    S = Storage(C_old.I.f, X)

    dJ_old = zeros(6)
    J_old = 0.0
    J_old = negative_log_likelihood!(J_old, dJ_old, getcoeff(C_old), S, C_old, X)

    @test abs(J_old - 3.066735373495125)<1e-5

    dJt_old  = [-1.665181693060798;
                -0.919442416275569;
                -0.900922248322526;
                -0.119522662066802;
                -0.472794468187974;
                -0.496862354324658]

    @test norm(dJ_old - dJt_old)<1e-5



    reduced_margin0 = getreducedmargin(getidx(C_old))
    idx_old0 = getidx(C_old)
    idx_new0 = vcat(idx_old0, reduced_margin0)

    # Define updated map
    f_new = ExpandedFunction(C_old.I.f.B, idx_new0, vcat(getcoeff(C_old), zeros(size(reduced_margin0,1))))
    C_new = HermiteMapComponent(f_new; α = 1e-6)
    idx_new, reduced_margin = update_component!(C_old, X, reduced_margin0, S)

    dJ_new = zeros(10)
    J_new = 0.0
    S = update_storage(S, X, reduced_margin0)
    J_new = negative_log_likelihood!(J_new, dJ_new, getcoeff(C_new), S, C_new, X)

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

    @test reduced_margin == updatereducedmargin(getidx(C_old), reduced_margin0, 1)[2]
    @test idx_new == updatereducedmargin(getidx(C_old), reduced_margin0, 1)[1]
end

@testset "Test greedy optimization on training set only without QR " begin

Nx = 2
m = 15

X = Matrix([0.267333   1.43021;
          0.364979   0.607224;
         -1.23693    0.249277;
         -2.0526     0.915629;
         -0.182465   0.415874;
          0.412907   1.01672;
          1.41332   -0.918205;
          0.766647  -1.00445]')
L = LinearTransform(X)
# transform!(L, X)
C_new, error = greedyfit(m, Nx, X, 6; withconstant = false, withqr = false, verbose = false)

@test norm(error -  [1.317277788545110;
                     1.240470006245235;
                     1.239755181933741;
                     1.191998917636778;
                     0.989821300578785;
                     0.956587763340890;
                     0.848999846549310])<1e-4


@test norm(getcoeff(C_new) - [ 2.758739330966050;
                               4.992420909046952;
                               0.691952693078807;
                               5.179459837781270;
                              -3.837569097976292;
                               1.660285776681347])<1e-4

@test norm(getidx(C_new) -   [0  1;
                              0  2;
                              0  3;
                              0  4;
                              1  0;
                              2  0])<1e-10
end

@testset "Test greedy optimization on training set only with QR " begin

Nx = 2
m = 15

X = Matrix([0.267333   1.43021;
          0.364979   0.607224;
         -1.23693    0.249277;
         -2.0526     0.915629;
         -0.182465   0.415874;
          0.412907   1.01672;
          1.41332   -0.918205;
          0.766647  -1.00445]')
L = LinearTransform(X)
# transform!(L, X)
C_new, error = greedyfit(m, Nx, X, 6; withconstant = false, withqr = true, verbose = false)

@test norm(error -  [1.317277788545110;
                     1.240470006245235;
                     1.239755181933741;
                     1.191998917636778;
                     0.989821300578785;
                     0.956587763340890;
                     0.848999846549310])<1e-4


@test norm(getcoeff(C_new) - [ 2.758739330966050;
                               4.992420909046952;
                               0.691952693078807;
                               5.179459837781270;
                              -3.837569097976292;
                               1.660285776681347])<1e-4

@test norm(getidx(C_new) -   [0  1;
                              0  2;
                              0  3;
                              0  4;
                              1  0;
                              2  0])<1e-10
end

@testset "Test greedy optimization on train dataset without/with QR" begin

Nx = 2
m = 10
Ne = 10^3
X = randn(2, Ne).^2 + 0.1*randn(2, Ne)
L = LinearTransform(X)
transform!(L, X)

C_noqr, error_noqr = greedyfit(m, Nx, X, 10; withconstant = true, withqr = false, verbose = false);
C_qr, error_qr = greedyfit(m, Nx, X, 10; withconstant = true, withqr = true, verbose = false);

train_error_noqr = error_noqr
train_error_qr   = error_qr


C_test = deepcopy(C_noqr)
setcoeff!(C_test, zero(getcoeff(C_noqr)));

S_test = Storage(C_test.I.f, X)
coeff_test = getcoeff(C_test)

res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S_test, C_test, X)), coeff_test, Optim.BFGS())

@test norm(getcoeff(C_noqr)-Optim.minimizer(res))<5e-5
@test norm(getcoeff(C_qr)-Optim.minimizer(res))<5e-5


@test norm(train_error_noqr[end] - res.minimum)<5e-5
@test norm(train_error_qr[end]   - res.minimum)<5e-5

end

@testset "Test greedy optimization on train/valid dataset without/with QR" begin

Nx = 2
m = 10
Ne = 10^3
X = randn(2, Ne).^2 + 0.1*randn(2, Ne)
L = LinearTransform(X)
transform!(L, X)

X_train = X[:,1:800]
X_valid = X[:,801:1000]


C_noqr, error_noqr = greedyfit(m, Nx, X_train, X_valid, 10; withconstant = true, withqr = false, verbose = false);
C_qr, error_qr = greedyfit(m, Nx, X_train, X_valid, 10; withconstant = true, withqr = true, verbose = false);

train_error_noqr, valid_error_noqr = error_noqr
train_error_qr, valid_error_qr = error_qr

C_test = deepcopy(C_noqr)
setcoeff!(C_test, zero(getcoeff(C_noqr)));

S_test = Storage(C_test.I.f, X_train)
coeff_test = getcoeff(C_test)

res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S_test, C_test, X_train)), coeff_test, Optim.BFGS())

@test norm(getcoeff(C_noqr) - Optim.minimizer(res))<5e-5
@test norm(getcoeff(C_qr)  - Optim.minimizer(res))<5e-5


@test norm(train_error_noqr[end] - res.minimum)<5e-5
@test norm(train_error_qr[end] - res.minimum)<5e-5
@test norm(valid_error_noqr[end] - valid_error_qr[end])<5e-5

end
