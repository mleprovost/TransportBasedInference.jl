
@testset "Forward diff for functions" begin
    x = randn()
    C = constant()
    @test abs(C(x)-1.0)<1e-10

    x = randn()
    F = linear()

    @test abs(F(x)-x)<1e-10
    # @test abs(ForwardDiff.derivative(F,1.0)-F.a₀)<1e-10

    x = randn()
    F= ψ₀(1.0,1.0)
    @test abs(ForwardDiff.derivative(F,x)-ψ₀′(1.0, 1.0)(x))<1e-10
end

@testset "Verify function rbf and its derivative" begin
    f1 = rbf(0.0,1.0)
    f1′ = rbf′(0.0,1.0)

    @test norm(f1(0.0)-1/√(2*π))<1e-10

    # ϕ′(x) = -x ϕ(x)
    x = randn()
    @test norm(f1′(x) -(-x*f1(x)))<1e-10

    @test norm(f1′(x) - ForwardDiff.derivative(f1,x))<1e-10

    f2 = rbf(1.5, 2.0)
    f2′ = rbf′(1.5, 2.0)

    @test norm(f2(1.5)-1/(2.0*√(2*π)))<1e-10

    x = randn()
    y = randn()
    @test norm(f2′(x) - ForwardDiff.derivative(f2, x))<1e-10
    @test norm(f2′(y) - ForwardDiff.derivative(f2, y))<1e-10
end


@testset "Verify ψ₀, ψⱼ, ψpp1" begin
    x = randn()
    ψ0 = ψ₀(0.0, 1.0)

    @test norm(ψ0(0.0) - (-0.5*√(2/π)))<1e-10

    ψⱼ = ψj(0.0,1.0)
    @test norm(ψⱼ(x) - 0.5*(1+erf(x/√2)))<1e-10

    ψpplus1 = ψpp1(0.0, 1.0)
    @test norm(ψpplus1(0.0) - (0.5*√(2/π)))<1e-10
end

@testset "Verify ψ₀′, ψpp1′" begin

    ψ0 = ψ₀(5.0, 5.0)

    ψ0p = ψ₀′(5.0, 5.0)

    @test abs(ForwardDiff.derivative(ψ0, 1.0) - ψ0p(1.0)) <1e-10

    ψpplus1 = ψpp1(1.0, 5.0)

    ψpplus1p = ψpp1′(1.0, 5.0)

    x = randn()
    @test abs(ForwardDiff.derivative(ψpplus1, x) - ψpplus1p(x)) <1e-10
end
