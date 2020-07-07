
using Polynomials
using AdaptiveTransportMap: derivative, vander

@testset "Test physical Hermite polynomials"   begin




x = randn()

P0 = PhyPolyHermite(0)
@test P0(x) == 1.0
@test derivative(P0,0).coeffs == (1.0,)
@test derivative(P0,0).coeffs == (1.0,)
@test derivative(P0,1).coeffs == ()
@test derivative(P0,2).coeffs  == ()

@test (derivative(P0,1))(x) == 0.0
@test (derivative(P0,2))(x) == 0.0


P0 = PhyPolyHermite(0; scaled = true)
@test P0(x) == 1/sqrt(sqrt(π))
@test Cphy(0) == sqrt(sqrt(π))
@test derivative(P0,0).coeffs == (1/sqrt(sqrt(π)),)
@test derivative(P0,0).coeffs == (1/sqrt(sqrt(π)),)
@test derivative(P0,1).coeffs == ()
@test derivative(P0,2).coeffs  == ()


P1 = PhyPolyHermite(1)
@test P1(x) == 2*x
@test derivative(P1, 0).coeffs == (0.0, 2.0)
@test derivative(P1, 1).coeffs == (2.0,)
@test derivative(P1, 2).coeffs == ()

P1 = PhyPolyHermite(1; scaled = true)
@test abs(P1(x) - 2*x/sqrt(sqrt(π)*2*factorial(1)))<1e-10
@test derivative(P1, 1).coeffs == (2/sqrt(sqrt(π)*2*factorial(1)),)
@test derivative(P1, 2).coeffs == ()


# Verify recurrence relation
# Hn+1(x) = 2*x*Hn(x) - Hn′(x)
# Hn′(x) = 2*n*Hn-1(x)

nodes, weights = gausshermite( 100000 )

for i=1:11
    Pmm1 = PhyPolyHermite(i-1)
    Pm = PhyPolyHermite(i)
    Pmp1 = PhyPolyHermite(i+1)
    z = 0.5

    @test abs(Pmp1(z) - (2*z*Pm(z) - (derivative(Pm,1))(z)))<1e-7

    @test abs(derivative(Pm, 0)(z) - Pm(z))<1e-7
    @test abs(derivative(Pm, 1)(z) - 2*i*Pmm1(z))<1e-7

    @test abs(derivative(Pmp1, 2)(z) - 4*(i+1)*(i)*Pmm1(z))<1e-7


    # Verify normalizing constant:
    @test abs(Cphy(i) - sqrt(sqrt(π)*2^i*factorial(i)))<1e-7
    @test abs(Cphy(Pm)- sqrt(sqrt(π)*2^i*factorial(i)))<1e-7

    if i<4
        # Verify integrals
        @test abs(dot(weights, Pm.P.(nodes).*Pm.P.(nodes))-sqrt(π)*2^i*factorial(i))<1e-7
        @test abs(dot(weights, Pm.P.(nodes).*Pmp1.P.(nodes)))<1e-7
        @test abs(dot(weights, Pm.P.(nodes).*Pmm1.P.(nodes)))<1e-7
        @test abs(dot(weights, Pmp1.P.(nodes).*Pmm1.P.(nodes)))<1e-7
    end
end

x = randn()

@test norm([PhyPolyHermite(10).P.coeffs...] - [-30240.0, 0.0, 302400, 0.0, -403200.0, 0.0, 161280, 0.0, -23040, 0.0, 1024])<1e-7
@test norm([derivative(PhyPolyHermite(10),1).coeffs...] - [Polynomials.derivative(PhyPolyHermite(10).P).coeffs...])<1e-7
@test norm([derivative(PhyPolyHermite(10),2).coeffs...] - [Polynomials.derivative(Polynomials.derivative(PhyPolyHermite(10).P)).coeffs...])<1e-7


P10 = PhyPolyHermite(10)

@test abs(P10(1.5) -  dot([-30240.0, 0.0, 302400, 0.0, -403200.0, 0.0, 161280, 0.0, -23040, 0.0, 1024], map(i->1.5^i,0:10)))<1e-10


# Compute inner product for normalized Physicist Hermite polynomials

for i=1:6
    Pmm1 = PhyPolyHermite(i-1; scaled = true)
    Pm = PhyPolyHermite(i; scaled = true)
    Pmp1 = PhyPolyHermite(i+1; scaled = true)

    @test abs(dot(weights, Pm.P.(nodes).*Pm.P.(nodes))-1.0)<1e-8
    @test abs(dot(weights, Pm.P.(nodes).*Pmp1.P.(nodes)))<1e-8
    @test abs(dot(weights, Pm.P.(nodes).*Pmm1.P.(nodes)))<1e-8
    @test abs(dot(weights, Pmp1.P.(nodes).*Pmm1.P.(nodes)))<1e-8

    # Check derivatives
    z = 0.25
    @test abs(derivative(Pm, 0)(z) - Pm(z))<1e-7
    @test abs(derivative(Pm, 1)(z) - 2*i*(1/Cphy(i))*FamilyPhyPolyHermite[i](z))<1e-7

    @test abs(derivative(Pmp1, 2)(z) - 4*(i+1)*(i)*(1/Cphy(i+1))*FamilyPhyPolyHermite[i](z))<1e-7
end

P12 = PhyPolyHermite(12;  scaled=true)
@test norm([derivative(P12,1).coeffs...] - [Polynomials.derivative(P12.P).coeffs...])<1e-10
@test norm([derivative(P12,2).coeffs...] - [Polynomials.derivative(Polynomials.derivative(P12.P)).coeffs...])<1e-10


end


@testset "Vandermonde matrix test for Physicist Hermite polynomials" begin
    # k = 0: check simple polynomial evaluation
    N = 20
    m = 10
    k = 0
    x = 0.2*randn(N)

    V = vander(PhyPolyHermite(m), k, x)

    @test size(V)==(N, m+1)

    for i=0:m
        @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i+1]).(x))<1e-8
    end

    # k = 0: check simple polynomial evaluation with scaled = true
    N = 20
    m = 10
    k = 0
    x = 0.2*randn(N)

    V = vander(PhyPolyHermite(m), k, x; scaled = true)

    @test size(V)==(N, m+1)

    for i=0:m
        factor = 2^k*exp(loggamma(i+1) - loggamma(i+1-k))

        @test norm(V[:,i+1] - (FamilyScaledPhyPolyHermite[i+1]).(x)*sqrt(factor))<1e-8
    end

    # k = 1: check first derivative
    N = 20
    m = 10
    k = 1
    x = 0.2*randn(N)

    V = vander(PhyPolyHermite(m), k, x)

    @test size(V)==(N, m+1)

    for i=1:m
        @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i]).(x)*2*i)<1e-8
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
    #     @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i]).(x)*sqrt(factor))<1e-8
    # end


    # k = 2: check second derivative
    N = 20
    m = 10
    k = 2
    x = 0.2*randn(N)

    V = vander(PhyPolyHermite(m), k, x)

    @test size(V)==(N, m+1)

    for i=2:m
        @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i-1]).(x)*2*i*2*(i-1))<1e-8
        @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i-1]).(x)*2^k*factorial(i)/factorial(i-k))<1e-8

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
#         @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i-1]).(x)*2*i*2*(i-1)*sqrt(2*exp(loggamma(i+2-1)-loggamma(i+2-2))))<1e-8
#         @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i-1]).(x)*2^k*factorial(i)/factorial(i-k)*sqrt(2*exp(loggamma(i+2-1)-loggamma(i+2-2))))<1e-8
#
#     end
#
    # k = 3: check third derivative
    N = 20
    m = 10
    k = 3
    x = 0.2*randn(N)

    V = vander(PhyPolyHermite(m), k, x)

    @test size(V)==(N, m+1)

    for i=3:m
        @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i-2]).(x)*2*i*2*(i-1)*2*(i-2))<1e-6
        @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i-2]).(x)*2^k*factorial(i)/factorial(i-k))<1e-6

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
#     #     @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i-2]).(x)*2*i*2*(i-1)*sqrt(2*exp(loggamma(i+2-1)-loggamma(i+2-2))))<1e-8
#     #     @test norm(V[:,i+1] - (FamilyPhyPolyHermite[i-2]).(x)*2^k*factorial(i)/factorial(i-k)*sqrt(2*exp(loggamma(i+2-1)-loggamma(i+2-2))))<1e-8
#     #
#     # end
#
#
#
#
end
