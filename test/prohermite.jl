

@testset "Test probabilistic Hermite polynomials"   begin


x = randn()

P0 = ProHermite(0)
@test P0(x) == 1.0
@test gradient(P0, x) == 0.0
@test hessian(P0, x)  == 0.0

P0 = ProHermite(0; scaled = true)
@test P0(x) == 1/sqrt(sqrt(2*π))
@test P0(x) == 1/Cpro(0)
@test gradient(P0, x) == 0.0
@test hessian(P0, x)  == 0.0


P1 = ProHermite(1)
@test P1(x) == x
@test gradient(P1, x) == 1.0
@test hessian(P1, x)  == 0.0

P1 = ProHermite(1; scaled = true)
@test abs(P1(x) - x/sqrt(sqrt(2*π)*factorial(1)))<1e-10
@test abs(gradient(P1, x) - 1.0/sqrt(sqrt(2*π)factorial(1)))<1e-10
@test hessian(P1, x)  == 0.0


# Verify recurrence relation
# Hn+1(x) = x*Hn(x) - Hn′(x)
# Hn′(x) = n*Hn-1(x)

nodes, weights = gausshermite( 100000 )

for i=1:15
    Pmm1 = ProHermite(i-1)
    Pm = ProHermite(i)
    Pmp1 = ProHermite(i+1)
    z = -1.25

    @test abs(Pmp1(z) - (z*Pm(z) - Pm.Pprime(z)))<1e-8
    @test abs(Pmp1(z) - (z*Pm(z) - gradient(Pm, z)))<1e-8

    @test abs(gradient(Pm, z) - i*Pmm1(z))<1e-8

    @test abs(hessian(Pmp1, z) - (i+1)*(i)*Pmm1(z))<1e-8

    @test Pm.P.coeffs[end] == 1.0


    # Verify normalizing constant:
    @test abs(Cpro(i) - sqrt(sqrt(2*π)*factorial(i)))<1e-8
    @test abs(Cpro(Pm)- sqrt(sqrt(2*π)*factorial(i)))<1e-8
end

x = randn()


@test norm([ProHermite(10).P.coeffs...] - [-945.0, 0.0, 4725.0, 0.0, -3150.0, 0.0, 630.0, 0.0, -45.0, 0.0, 1.0])<1e-10


P10 = ProHermite(10)

@test abs(P10(-1.25) -  dot([-945.0, 0.0, 4725.0, 0.0, -3150.0, 0.0, 630.0, 0.0, -45.0, 0.0, 1.0], map(i->(-1.25)^i,0:10)))<1e-10

@test norm([ProHermite(10).Pprime.coeffs...] - [Polynomials.derivative(ProHermite(10).P).coeffs...])<1e-10
@test norm([ProHermite(10).Ppprime.coeffs...] - [Polynomials.derivative(Polynomials.derivative(ProHermite(10).P)).coeffs...])<1e-10



# Verify relation between PhyHermite and ProHermite
# Hn(x) = 2^(n/2)Hen(√2*x)
# Hen(x) = 2^(-n/2)Hn(x/√2)

x = -0.5:0.1:0.5

for i=1:6
    @show i
    Hi  = PhyHermite(i)
    Hei = ProHermite(i)
    for xi in x
        @test abs(Hi(xi)-(2^(i/2)*Hei(sqrt(2)*xi)))<1e-8
        @test abs(Hei(xi)-(2^(-i/2)*Hi(xi/sqrt(2))))<1e-8
    end

end

end
