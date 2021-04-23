
@testset "ui and its derivative" begin

    um1 = ui(-1, Float64[], Float64[], Float64[])
    x = randn()
    @test um1(x)==0.0

    @test D(um1)(x)==0.0

    a0 = randn()
    u0 = ui(0, Float64[], Float64[], [a0])
    @test norm(u0(x) - a0*x)<1e-10

    @test norm(D(u0)(x) - ForwardDiff.derivative(u0, x))<1e-10


    @test ui(0, Float64[], Float64[], [1.0])(x)==x

    u1 = ui(1, [-2.0], [1.0] , [3.0,2.0])
    @test abs(u1(x) - (3.0*x + 2.0*rbf(-2.0,1.0)(x)))<1e-10

    u4 = ui(2, zeros(2), ones(2), [1.0,0.0,0.0])
    @test u4.p == 2

    @test u4(2.0) ==2.0

    u4.ai[1] = 0.0
    u4.ai[2] = 1.0

    @test u4.ai == [0.0, 1.0, 0.0]

    @test u4(2.0) == rbf(0.0,1.0)(2.0)

    u4.ai[1] = 2.0
    u4.ai[3] = 2.0
    u4.ξi[2] = 2.0
    u4.σi[2] = 5.0

    @test abs(u4(x)- (Lin(x)*2 + rbf(0.0, 1.0)(x) +
                2.0*rbf(2.0, 5.0)(x))) <1e-10

    @test abs(D(u4)(x) - ForwardDiff.derivative(u4, x))<1e-10

    u5 = ui(4, randn(4), rand(4), randn(5))

    @test abs(D(u5)(x) - ForwardDiff.derivative(u5, x))<1e-10
end


@testset "uk and its derivative (first and second)" begin
    um1 = uk(-1, Float64[], Float64[],Float64[])
    x = randn()
    @test um1(x) == x
    @test D!(um1, x)==1.0
    @test D(um1)(x)==1.0
    @test H!(um1, x)==0.0
    @test H(um1)(x)==0.0

    @test uk(0, Float64[], Float64[],[5.0;2.0])(x)== 5.0 + 2.0*x

    a = randn(2)
    u0 = uk(0, Float64[], Float64[],a)
    @test norm(D!(u0, x) - ForwardDiff.derivative(u0,x))<1e-10

    u4 = uk(2)

    @test u4.p == 2

    @test u4(x) == 0.0
    #

    u4.ak[1] = 3.0

    @test u4.ak == [3.0, 0.0, 0.0, 0.0, 0.0]

    @test u4(x) == 3.0

    u4.ak[1] = 0.0
    u4.ak[2] = 2.0

    @test u4(x) == 2.0ψ₀(0.0,1.0)(x)

    u4.ak .= ones(5)

    @test abs(u4(x) -( 1.0 + ψ₀(0.0,1.0)(x) + 2.0*ψj(0.0,1.0)(x) +
                        ψpp1(0.0,1.0)(x))) <1e-10

    u4.ak[1] = 6.0
    u4.σk .= [1.0,2.0,3.0,4.0]

    @test abs(u4(x) -( 6.0 + ψ₀(0.0,1.0)(x) + ψj(0.0,2.0)(x) + ψj(0.0,3.0)(x) +
                        ψpp1(0.0,4.0)(x))) <1e-10
    @test norm(D!(u4, x) - ForwardDiff.derivative(u4, x))<1e-10

    u5 = uk(5, randn(7), rand(7), randn(8))


    @test norm(D(u5)(x) - ForwardDiff.derivative(u5, x))<1e-10

    @test norm(H!(u5, x) - ForwardDiff.derivative(D(u5), x))<1e-10
    @test norm(H(u5)(x) - ForwardDiff.derivative(D(u5), x))<1e-10

end
