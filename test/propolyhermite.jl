
using Polynomials
using AdaptiveTransportMap: derivative, vander

@testset "Test probabilistic Hermite polynomials"   begin




x = randn()

P0 = ProPolyHermite(0)
@test P0(x) == 1.0
@test derivative(P0,0).coeffs == (1.0,)
@test derivative(P0,0).coeffs == (1.0,)
@test derivative(P0,1).coeffs == ()
@test derivative(P0,2).coeffs  == ()

@test (derivative(P0,1))(x) == 0.0
@test (derivative(P0,2))(x) == 0.0


P0 = ProPolyHermite(0; scaled = true)
@test P0(x) == 1/sqrt(sqrt(2*π))
@test Cpro(0) == sqrt(sqrt(2*π))
@test derivative(P0,0).coeffs == (1/sqrt(sqrt(2*π)),)
@test derivative(P0,0).coeffs == (1/sqrt(sqrt(2*π)),)
@test derivative(P0,1).coeffs == ()
@test derivative(P0,2).coeffs  == ()


P1 = ProPolyHermite(1)
@test P1(x) == x
@test derivative(P1, 0).coeffs == (0.0, 1.0)
@test derivative(P1, 1).coeffs == (1.0,)
@test derivative(P1, 2).coeffs == ()

P1 = ProPolyHermite(1; scaled = true)
@test abs(P1(x) - x/sqrt(sqrt(2*π)*factorial(1)))<1e-10
@test derivative(P1, 1).coeffs == (1/sqrt(sqrt(2*π)*factorial(1)),)
@test derivative(P1, 2).coeffs == ()


# Verify recurrence relation
# Hn+1(x) = x*Hn(x) - Hn′(x)
# Hn′(x) = n*Hn-1(x)

for i=1:11
    Pmm1 = ProPolyHermite(i-1)
    Pm = ProPolyHermite(i)
    Pmp1 = ProPolyHermite(i+1)
    z = 0.5

    @test abs(Pmp1(z) - (z*Pm(z) - (derivative(Pm,1))(z)))<1e-7

    @test abs(derivative(Pm, 0)(z) - Pm(z))<1e-7
    @test abs(derivative(Pm, 1)(z) - i*Pmm1(z))<1e-7

    @test abs(derivative(Pmp1, 2)(z) - (i+1)*(i)*Pmm1(z))<1e-7


    # Verify normalizing constant:
    @test abs(Cpro(i) - sqrt(sqrt(2*π)*factorial(i)))<1e-7
    @test abs(Cpro(Pm)- sqrt(sqrt(2*π)*factorial(i)))<1e-7


end

x = randn()

@test norm([ProPolyHermite(10).P.coeffs...] - [-945.0, 0.0, 4725.0, 0.0, -3150.0, 0.0, 630.0, 0.0, -45.0, 0.0, 1.0])<1e-10


P10 = ProPolyHermite(10)

@test abs(P10(-1.25) -  dot([-945.0, 0.0, 4725.0, 0.0, -3150.0, 0.0, 630.0, 0.0, -45.0, 0.0, 1.0], map(i->(-1.25)^i,0:10)))<1e-10

@test norm([derivative(ProPolyHermite(10),1).coeffs...] - [Polynomials.derivative(ProPolyHermite(10).P).coeffs...])<1e-7
@test norm([derivative(ProPolyHermite(10),2).coeffs...] - [Polynomials.derivative(Polynomials.derivative(ProPolyHermite(10).P)).coeffs...])<1e-7


# Compute inner product for normalized Probabilistic Hermite polynomials

for i=1:6
    Pmm1 = ProPolyHermite(i-1; scaled = true)
    Pm = ProPolyHermite(i; scaled = true)
    Pmp1 = ProPolyHermite(i+1; scaled = true)


    # Check derivatives
    z = 0.25
    @test abs(derivative(Pm, 0)(z) - Pm(z))<1e-7
    @test abs(derivative(Pm, 1)(z) - i*(1/Cpro(i))*FamilyProPolyHermite[i](z))<1e-7

    @test abs(derivative(Pmp1, 2)(z) - (i+1)*(i)*(1/Cpro(i+1))*FamilyProPolyHermite[i](z))<1e-7
end

P12 = ProPolyHermite(12;  scaled=true)
@test norm([derivative(P12,1).coeffs...] - [Polynomials.derivative(P12.P).coeffs...])<1e-10
@test norm([derivative(P12,2).coeffs...] - [Polynomials.derivative(Polynomials.derivative(P12.P)).coeffs...])<1e-10


end


@testset "Vandermonde matrix test for Probabilistic Hermite polynomials" begin
    # k = 0: check simple polynomial evaluation
    N = 20
    m = 10
    k = 0
    x = 0.2*randn(N)

    V = vander(ProPolyHermite(m), k, x)

    @test size(V)==(N, m+1)

    for i=0:m
        @test norm(V[:,i+1] - (FamilyProPolyHermite[i+1]).(x))<1e-8
    end

    # k = 0: check simple polynomial evaluation with scaled = true
    N = 20
    m = 10
    k = 0
    x = 0.2*randn(N)

    V = vander(ProPolyHermite(m), k, x; scaled = true)

    @test size(V)==(N, m+1)

    for i=0:m
        factor = exp(loggamma(i+1) - loggamma(i+1-k))

        @test norm(V[:,i+1] - (FamilyScaledProPolyHermite[i+1]).(x)*sqrt(factor))<1e-8
    end

    # k = 1: check first derivative
    N = 20
    m = 10
    k = 1
    x = 0.2*randn(N)

    V = vander(ProPolyHermite(m), k, x)

    @test size(V)==(N, m+1)

    for i=1:m
        @test norm(V[:,i+1] - (FamilyProPolyHermite[i]).(x)*i)<1e-8
    end

    # # k = 1: check first derivative scaled = true
    # N = 20
    # m = 10
    # k = 1
    # x = 0.2*randn(N)
    #
    # V = vander(m, k, x; scaled = true)
    #
    # @test size(V)==(N, m+1)
    #
    # for i=1:m
    #     factor = 2^k*exp(loggamma(i+1) - loggamma(i+1-k))
    #
    #     @test norm(V[:,i+1] - (FamilyProPolyHermite[i]).(x)*sqrt(factor))<1e-8
    # end


    # k = 2: check second derivative
    N = 20
    m = 10
    k = 2
    x = 0.2*randn(N)

    V = vander(ProPolyHermite(m), k, x)

    @test size(V)==(N, m+1)

    for i=2:m
        @test norm(V[:,i+1] - (FamilyProPolyHermite[i-1]).(x)*i*(i-1))<1e-8
        @test norm(V[:,i+1] - (FamilyProPolyHermite[i-1]).(x)*factorial(i)/factorial(i-k))<1e-8

    end
#
#     # k = 2: check second derivative scaled = true
#     N = 20
#     m = 10
#     k = 2
#     x = 0.2*randn(N)
#
#     V = vander(m, k, x; scaled = true)
#
#     @test size(V)==(N, m+1)
#
#     for i=2:m
#         @test norm(V[:,i+1] - (FamilyProPolyHermite[i-1]).(x)*2*i*2*(i-1)*sqrt(2*exp(loggamma(i+2-1)-loggamma(i+2-2))))<1e-8
#         @test norm(V[:,i+1] - (FamilyProPolyHermite[i-1]).(x)*2^k*factorial(i)/factorial(i-k)*sqrt(2*exp(loggamma(i+2-1)-loggamma(i+2-2))))<1e-8
#
#     end
#
    # k = 3: check third derivative
    N = 20
    m = 10
    k = 3
    x = 0.2*randn(N)

    V = vander(ProPolyHermite(m), k, x)

    @test size(V)==(N, m+1)

    for i=3:m
        @test norm(V[:,i+1] - (FamilyProPolyHermite[i-2]).(x)*i*(i-1)*(i-2))<1e-6
        @test norm(V[:,i+1] - (FamilyProPolyHermite[i-2]).(x)*factorial(i)/factorial(i-k))<1e-6

    end

#
#     # # k = 3: check third derivative scaled = true
#     # N = 20
#     # m = 10
#     # k = 2
#     # x = 0.2*randn(N)
#     #
#     # V = vander(m, k, x)
#     #
#     # @test size(V)==(N, m+1)
#     #
#     # for i=3:m
#     #     @test norm(V[:,i+1] - (FamilyProPolyHermite[i-2]).(x)*2*i*2*(i-1)*sqrt(2*exp(loggamma(i+2-1)-loggamma(i+2-2))))<1e-8
#     #     @test norm(V[:,i+1] - (FamilyProPolyHermite[i-2]).(x)*2^k*factorial(i)/factorial(i-k)*sqrt(2*exp(loggamma(i+2-1)-loggamma(i+2-2))))<1e-8
#     #
#     # end
#
#
#
#
end


@testset "Verify relation between PhyHermite and ProHermite" begin
# Hn(x) = 2^(n/2)Hen(√2*x)
# Hen(x) = 2^(-n/2)Hn(x/√2)
    x = -0.5:0.1:0.5

    for i=1:6
        Hi  = PhyPolyHermite(i)
        Hei = ProPolyHermite(i)
        for xi in x
            @test abs(Hi(xi)-(2^(i/2)*Hei(sqrt(2)*xi)))<1e-8
            @test abs(Hei(xi)-(2^(-i/2)*Hi(xi/sqrt(2))))<1e-8
        end
    end

end
