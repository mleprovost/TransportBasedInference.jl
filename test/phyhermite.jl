using Polynomials

@testset "Test physical Hermite polynomials"   begin


x = randn()

P0 = PhyPolyHermite(0)
@test P0(x) == 1.0
@test gradient(P0, x) == 0.0
@test hessian(P0, x)  == 0.0

P0 = PhyPolyHermite(0; scaled = true)
@test P0(x) == 1/sqrt(sqrt(π))
@test Cphy(0) == sqrt(sqrt(π))
@test gradient(P0, x) == 0.0
@test hessian(P0, x)  == 0.0


P1 = PhyPolyHermite(1)
@test P1(x) == 2*x
@test gradient(P1, x) == 2.0
@test hessian(P1, x)  == 0.0

P1 = PhyPolyHermite(1; scaled = true)
@test abs(P1(x) - 2*x/sqrt(sqrt(π)*2*factorial(1)))<1e-10
@test abs(gradient(P1, x) - 2.0/sqrt(sqrt(π)*2*factorial(1)))<1e-10
@test hessian(P1, x)  == 0.0


# Verify recurrence relation
# Hn+1(x) = 2*x*Hn(x) - Hn′(x)
# Hn′(x) = 2*n*Hn-1(x)

nodes, weights = gausshermite( 100000 )

for i=1:15
    Pmm1 = PhyPolyHermite(i-1)
    Pm = PhyPolyHermite(i)
    Pmp1 = PhyPolyHermite(i+1)
    z = 1.5

    @test abs(Pmp1(z) - (2*z*Pm(z) - Pm.Pprime(z)))<1e-8
    @test abs(Pmp1(z) - (2*z*Pm(z) - gradient(Pm, z)))<1e-8

    @test abs(gradient(Pm, z) - 2*i*Pmm1(z))<1e-8

    @test abs(hessian(Pmp1, z) - 4*(i+1)*(i)*Pmm1(z))<1e-8


    # Verify normalizing constant:
    @test abs(Cphy(i) - sqrt(sqrt(π)*2^i*factorial(i)))<1e-8
    @test abs(Cphy(Pm)- sqrt(sqrt(π)*2^i*factorial(i)))<1e-8

    if i<6
        # Verify integrals

        @test abs(dot(weights, Pm.P.(nodes).*Pm.P.(nodes))-sqrt(π)*2^i*factorial(i))<1e-8
        @test abs(dot(weights, Pm.P.(nodes).*Pmp1.P.(nodes)))<1e-8
        @test abs(dot(weights, Pm.P.(nodes).*Pmm1.P.(nodes)))<1e-8
        @test abs(dot(weights, Pmp1.P.(nodes).*Pmm1.P.(nodes)))<1e-8
    end
end

x = randn()

@test norm([PhyPolyHermite(10).P.coeffs...] - [-30240.0, 0.0, 302400, 0.0, -403200.0, 0.0, 161280, 0.0, -23040, 0.0, 1024])<1e-10
@test norm([PhyPolyHermite(10).Pprime.coeffs...] - [Polynomials.derivative(PhyPolyHermite(10).P).coeffs...])<1e-10
@test norm([PhyPolyHermite(10).Ppprime.coeffs...] - [Polynomials.derivative(Polynomials.derivative(PhyPolyHermite(10).P)).coeffs...])<1e-10


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
end

P12 = PhyPolyHermite(12;  scaled=true)
@test norm([P12.Pprime.coeffs...] - [Polynomials.derivative(P12.P).coeffs...])<1e-10
@test norm([P12.Ppprime.coeffs...] - [Polynomials.derivative(Polynomials.derivative(P12.P)).coeffs...])<1e-10






end
