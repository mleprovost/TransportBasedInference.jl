
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
    idx_new = vcat(getidx(Hk), reduced_margin)

    Hk_new = HermiteMapk(m, Nx, idx_new, vcat(getcoeff(Hk), zeros(2)));

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

    # setcoeff!(Hk_old, [1.5])
    reduced_margin0 = getreducedmargin(getidx(Hk_old))
    idx_old0 = getidx(Hk)
    idx_new0 = vcat(idx_old, reduced_margin0)

    # Define updated map
    f_new = ExpandedFunction(Hk_old.I.f.f.B, idx_new0, vcat(getcoeff(Hk), zeros(size(reduced_margin0,1))))
    Hk_new = HermiteMapk(f_new; α = 1e-6)
    idx_new, reduced_margin = update_component(Hk_old, X, reduced_margin0, S)

    dJ = zeros(3)
    S = update_storage(S, X, reduced_margin0)
    negative_log_likelihood!(nothing, dJ, getcoeff(Hk_new), S, Hk_new, X)

    dJt =  [ -1.6651836742291564;
             -0.9194410618771688;
             -0.9009267357141437;
             -0.11952335389279932;
             -0.4727973085063471;
             -0.49686342659213534;
              0.3388316230967308;
             -0.0835027246673459;
             -0.2811305337405587;
              0.27158719056266994]

    @test norm(dJ-dJt)<1e-8

    @test findmax(abs.(dJ[2:3]))[2] == 2-1

    @test idx_new[2,:] == reduced_margin0[1,:]

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
    # Initialize map with zero coefficients
    idx = [0 0; 0 1; 1 0; 0 2; 2 0; 1 1]

    coeff = [ -0.9905841755746164;
          0.6771992097558741;
         -2.243695806805015;
         -0.34591297359447354;
         -1.420159186008486;
         -0.5361337327704369]

    Hk_old = HermiteMapk(m, Nx, idx, coeff; α = 1e-6);

    S = Storage(Hk_old.I.f, X)

    reduced_margin0 = getreducedmargin(getidx(Hk_old))
    idx_old0 = getidx(Hk_old)
    idx_new0 = vcat(idx_old0, reduced_margin0)

    # Define updated map
    f_new = ExpandedFunction(Hk_old.I.f.f.B, idx_new0, vcat(getcoeff(Hk_old), zeros(size(reduced_margin0,1))))
    Hk_new = HermiteMapk(f_new; α = 1e-6)

    idx_new, reduced_margin = update_component(Hk_old, X, reduced_margin0, S)

    dJ = zeros(10)
    S = update_storage(S, X, reduced_margin0)
    negative_log_likelihood!(nothing, dJ, getcoeff(Hk_new), S, Hk_new, X)

    dJt =  [ -1.6651836742291564;
             -0.9194410618771688;
             -0.9009267357141437;
             -0.11952335389279932;
             -0.4727973085063471;
             -0.49686342659213534;
              0.3388316230967308;
             -0.0835027246673459;
             -0.2811305337405587;
              0.27158719056266994]

    @test norm(dJ-dJt)<1e-8

    @test idx_new[7,:] == reduced_margin0[1,:]

    @test reduced_margin == updatereducedmargin(getidx(Hk_old), reduced_margin0, 1)[2]
    @test idx_new == updatereducedmargin(getidx(Hk_old), reduced_margin0, 1)[1]
end
