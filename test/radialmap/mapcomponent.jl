
@testset "RadialMapComponent (Nx,p) = (1,0)" begin

    C = RadialMapComponent(1,0)

    @test C.Nx == 1
    @test C.p == 0
    @test size(C.ξ, 1)==1
    @test size(C.ξ[1], 1)==0

    @test size(C.σ, 1)==1
    @test size(C.σ[1], 1)==0

    @test size(C.coeff,1)==1
    @test size(C.coeff[1],1)==2

    C.coeff[1]=[2.0;1.0]
    @test C(1.0)==3.0
end

@testset "RadialMapComponent (Nx,p) = (1,1)" begin

    C = RadialMapComponent(1,1)

    @test C.Nx == 1
    @test C.p == 1

    @test size(C.ξ,1) ==1
    @test size(C.ξ[1],1) ==1+2

    @test size(C.σ,1) ==1
    @test size(C.σ[1],1) ==1+2

    @test size(C.ξ,1) ==1
    @test size(C.ξ[1],1) ==1+2

    @test size(C.coeff,1) ==1
    @test size(C.coeff[1],1) ==1+3

    C.coeff[1] .= [1.0; 2.0; 3.0; 1.0]

    x = randn()
    @test abs(C(x) - (1.0 + 2.0*ψ₀(0.0,1.0)(x) +
                     3.0*ψj(0.0,1.0)(x) + 1.0*ψpp1(0.0,1.0)(x)))< 1e-10

    @test abs(C([x]) - (1.0 + 2.0*ψ₀(0.0,1.0)(x) +
                      3.0*ψj(0.0,1.0)(x) + 1.0*ψpp1(0.0,1.0)(x)))< 1e-10

end


@testset "RadialMapComponent (Nx,p) = (Nx,0) with Nx>1" begin

    C = RadialMapComponent(3, 0)
    x1 = randn()
    x2 = randn()
    x3 = randn()

    @test C.Nx == 3
    @test C.p == 0

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.coeff,1)==3

    @test size(C.ξ[1],1)==0
    @test size(C.σ[1],1)==0
    @test size(C.coeff[1],1)==1

    @test size(C.ξ[2],1)==0
    @test size(C.σ[2],1)==0
    @test size(C.coeff[2],1)==1

    @test size(C.ξ[3],1)==0
    @test size(C.σ[3],1)==0
    @test size(C.coeff[3],1)==2

    @test C(1.0) == 0.0


    C.coeff[1][1] = 1.5

    @test isapprox(C([x1;0.0;0.0]), 1.5*x1, atol = 1e-10)

    C.coeff[2][1] = 2.0
    @test isapprox(C([x1;x2;0.0]), 1.5*x1 + 2.0*x2, atol = 1e-10)

    C.coeff[3][1] = 3.0
    @test isapprox(C([x1; x2; x3]), 1.5*x1 + 2.0*x2 + 3.0, atol = 1e-10)
    @test isapprox(C(x1), 1.5*x1+2.0*x1+3.0, atol = 1e-10)

    C.coeff[3][2] = 4.0
    @test isapprox(C([x1; x2; x3]), 1.5*x1 + 2.0*x2 + 3.0 + 4.0*x3, atol = 1e-10)
    @test isapprox(C(x1), 1.5*x1+2.0*x1+3.0 + 4.0*x1, atol = 1e-10)

end


@testset "RadialMapComponent (Nx,p) = (Nx,p) with Nx>1 and p>1" begin

    C = RadialMapComponent(3, 2)
    x1 = randn()
    x2 = randn()
    x3 = randn()


    @test C.Nx == 3
    @test C.p == 2

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.coeff,1)==3

    @test size(C.ξ[1],1)==2
    @test size(C.σ[1],1)==2
    @test size(C.coeff[1],1)==2+1

    @test size(C.ξ[2],1)==2
    @test size(C.σ[2],1)==2
    @test size(C.coeff[2],1)==2+1

    @test size(C.ξ[3],1)==2+2
    @test size(C.σ[3],1)==2+2
    @test size(C.coeff[3],1)==2+3

    @test C.ξ[1]==zeros(2)
    @test C.ξ[2]==zeros(2)
    @test C.ξ[3]==zeros(2+2)

    @test C.σ[1]==ones(2)
    @test C.σ[2]==ones(2)
    @test C.σ[3]==ones(2+2)

    @test C.coeff[1]==zeros(2+1)
    @test C.coeff[2]==zeros(2+1)
    @test C.coeff[3]==zeros(2+3)

    C.coeff[1] .= ones(3)

    @test isapprox(C(x1), ui(2,C.ξ[1], C.σ[1], C.coeff[1])(x1), atol = 1e-10)

    @test isapprox(C([x1; 0.0; 0.0]), ui(2,C.ξ[1], C.σ[1], C.coeff[1])(x1), atol = 1e-10)

    C.coeff[2] .= 2*ones(3)

    @test isapprox(C([x1; x2; 0.0]), ui(2,C.ξ[1], C.σ[1], C.coeff[1])(x1) + ui(2,C.ξ[2], C.σ[2], C.coeff[2])(x2), atol = 1e-10)

    C.coeff[3] .= 3*ones(5)

    @test isapprox(C([x1; x2; x3]), ui(2,C.ξ[1], C.σ[1], C.coeff[1])(x1) +
                                 ui(2,C.ξ[2], C.σ[2], C.coeff[2])(x2) +
                                 uk(2,C.ξ[3], C.σ[3], C.coeff[3])(x3), atol = 1e-10)


    @test isapprox(C(x1),   ui(2,C.ξ[1], C.σ[1], C.coeff[1])(x1) +
                            ui(2,C.ξ[2], C.σ[2], C.coeff[2])(x1) +
                            uk(2,C.ξ[3], C.σ[3], C.coeff[3])(x1), atol = 1e-10)
end


@testset "Component function for RadialMapComponent type" begin

    C = RadialMapComponent(1, 2)

    @test C.Nx == 1
    @test C.p == 2


    @test  TransportBasedInference.component(C, 1).ξk == zeros(2+2)
    @test  TransportBasedInference.component(C, 1).σk == ones(2+2)
    @test  TransportBasedInference.component(C, 1).coeffk == zeros(2+3)

    C = RadialMapComponent(5, 2)

    @test C.Nx == 5
    @test C.p == 2


    @test  TransportBasedInference.component(C, 5).ξk == zeros(2+2)
    @test  TransportBasedInference.component(C, 5).σk == ones(2+2)
    @test  TransportBasedInference.component(C, 5).coeffk == zeros(2+3)

    @test  TransportBasedInference.component(C, 2).ξi == zeros(2)
    @test  TransportBasedInference.component(C, 2).σi == ones(2)
    @test  TransportBasedInference.component(C, 2).coeffi == zeros(2+1)
end


@testset "Verify off_diagonal function" begin

    # Nx=1  & p=0
    C = RadialMapComponent(1,0)
    C.coeff[1] .= randn(2)

    @test off_diagonal(C, randn()) == 0.0

    # Nx=1 & p= 3
    C = RadialMapComponent(1, 3)
    C.coeff[1] .= rand(6)

    @test off_diagonal(C, randn()) == 0.0


    # Nx=3 & p = 0
    C = RadialMapComponent(3, 0)
    a1 =randn()
    a2 = randn()
    a3 = rand(2)
    C.coeff[1] .= a1
    C.coeff[2] .= a2
    C.coeff[3] .= a3

    z = randn(3)
    @test norm(off_diagonal(C, z) - (C(z) - TransportBasedInference.component(C,3)(z[3])))<1e-10

    # Nx=3 & p = 3
    C = RadialMapComponent(3, 3)
    for i=1:2
    C.ξ[i] .= randn(3)
    C.σ[i] .= rand(3)
    end

    C.ξ[3] .= randn(5)
    C.σ[3] .= rand(5)

    a1 =randn(4)
    a2 = randn(4)
    a3 = rand(6)
    C.coeff[1] .= a1
    C.coeff[2] .= a2
    C.coeff[3] .= a3

    z = randn(3)
    @test norm(off_diagonal(C, z) - (C(z) - TransportBasedInference.component(C,3)(z[3])))<1e-10
end



@testset "extract and modify coefficients of RadialMapComponent" begin
    # Nx=1 and p=0
    C = RadialMapComponent(1, 0)

    @test C.Nx == 1
    @test C.p == 0

    modifycoeff!(C, [1.0;2.0])
    @test extractcoeff(C) == [1.0; 2.0]

    C = RadialMapComponent(1, 3)
    modifycoeff!(C, [1.0;2.0; 3.0; 4.0; 5.0; 6.0])
    @test extractcoeff(C) == [1.0;2.0; 3.0; 4.0; 5.0; 6.0]


    # Nx =3 and p=2
    A = collect(1.0:1.0:11.0)
    C = RadialMapComponent(3, 2)

    modifycoeff!(C, A)
    @test  extractcoeff(C) == A

    C = RadialMapComponent(3, 2)
    C.coeff[1] = [1.0; 2.0; 3.0]
    C.coeff[2] = [4.0; 5.0; 6.0]
    C.coeff[3] = [7.0; 8.0; 9.0; 10.0; 11.0]

    @test extractcoeff(C) ==A

    # Nx = 3 and p = 0

    C = RadialMapComponent(3, 0)

    @test C.Nx == 3
    @test C.p == 0

    C.coeff[1] .= [1.0]
    C.coeff[2] .= [2.0]
    C.coeff[3] .= [3.0; 4.0]

    @test extractcoeff(C)==collect(1.0:4.0)

    modifycoeff!(C, collect(5.0:8.0))

    @test extractcoeff(C)==collect(5.0:8.0)
end
