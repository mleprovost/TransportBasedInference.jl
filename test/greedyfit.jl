
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
