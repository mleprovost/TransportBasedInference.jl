using Test

using LinearAlgebra, Statistics
using SpecialFunctions, ForwardDiff
using TransportMap

@testset "Uk (k,p) = (1,0)" begin

Vk = Uk(1,0)

@test Vk.k == 1
@test Vk.p == 0
@test size(Vk.ξ, 1)==1
@test size(Vk.ξ[1], 1)==0

@test size(Vk.σ, 1)==1
@test size(Vk.σ[1], 1)==0

@test size(Vk.a,1)==1
@test size(Vk.a[1],1)==2

Vk.a[1]=[2.0;1.0]
@test Vk(1.0)==3.0
end

@testset "Uk (k,p) = (1,1)" begin

Vk = Uk(1,1)

@test Vk.k == 1
@test Vk.p == 1

@test size(Vk.ξ,1) ==1
@test size(Vk.ξ[1],1) ==1+2

@test size(Vk.σ,1) ==1
@test size(Vk.σ[1],1) ==1+2

@test size(Vk.ξ,1) ==1
@test size(Vk.ξ[1],1) ==1+2

@test size(Vk.a,1) ==1
@test size(Vk.a[1],1) ==1+3

Vk.a[1] .= [1.0; 2.0; 3.0; 1.0]

@test abs(Vk(1.0) - (1.0 + 2.0*ψ₀(0.0,1.0)(1.0) +
                     3.0*ψj(0.0,1.0)(1.0) + 1.0*ψpp1(0.0,1.0)(1.0)))< 1e-10

 @test abs(Vk([1.0]) - (1.0 + 2.0*ψ₀(0.0,1.0)(1.0) +
                      3.0*ψj(0.0,1.0)(1.0) + 1.0*ψpp1(0.0,1.0)(1.0)))< 1e-10

end


@testset "Uk (k,p) = (k,0) with k>1" begin

Vk = Uk(3, 0)

@test Vk.k == 3
@test Vk.p == 0

@test size(Vk.ξ,1)==3
@test size(Vk.σ,1)==3
@test size(Vk.a,1)==3

@test size(Vk.ξ[1],1)==0
@test size(Vk.σ[1],1)==0
@test size(Vk.a[1],1)==1

@test size(Vk.ξ[2],1)==0
@test size(Vk.σ[2],1)==0
@test size(Vk.a[2],1)==1

@test size(Vk.ξ[3],1)==0
@test size(Vk.σ[3],1)==0
@test size(Vk.a[3],1)==2

@test Vk(1.0) == 0.0


Vk.a[1][1] = 1.0

@test Vk([1.0;0.0;0.0]) == 1.0

Vk.a[2][1] = 2.0
@test Vk([1.0;4.0;0.0]) == 9.0

Vk.a[3][1] = 3.0
@test Vk([1.0;4.0;5.0]) == 1.0*1.0 + 2.0*4.0 + 3.0
@test Vk(1.0) == 1.0*1.0+2.0*1.0+3.0

Vk.a[3][2] = 2.0
@test Vk([1.0;4.0;5.0]) == 1.0*1.0 + 2.0*4.0 + 3.0 + 2.0*5.0
@test Vk(1.0) == 1.0*1.0+2.0*1.0+3.0 + 2.0*1.0

end


@testset "Uk (k,p) = (k,p) with k>1 and p>1" begin

Vk = Uk(3, 2)

@test Vk.k == 3
@test Vk.p == 2

@test size(Vk.ξ,1)==3
@test size(Vk.σ,1)==3
@test size(Vk.a,1)==3

@test size(Vk.ξ[1],1)==2
@test size(Vk.σ[1],1)==2
@test size(Vk.a[1],1)==2+1

@test size(Vk.ξ[2],1)==2
@test size(Vk.σ[2],1)==2
@test size(Vk.a[2],1)==2+1

@test size(Vk.ξ[3],1)==2+2
@test size(Vk.σ[3],1)==2+2
@test size(Vk.a[3],1)==2+3

@test Vk.ξ[1]==zeros(2)
@test Vk.ξ[2]==zeros(2)
@test Vk.ξ[3]==zeros(2+2)

@test Vk.σ[1]==ones(2)
@test Vk.σ[2]==ones(2)
@test Vk.σ[3]==ones(2+2)

@test Vk.a[1]==zeros(2+1)
@test Vk.a[2]==zeros(2+1)
@test Vk.a[3]==zeros(2+3)

Vk.a[1] .= ones(3)

@test Vk(1.0)== ui(2,Vk.ξ[1], Vk.σ[1], Vk.a[1])(1.0)

@test Vk([1.0; 0.0; 0.0])== ui(2,Vk.ξ[1], Vk.σ[1], Vk.a[1])(1.0)

Vk.a[2] .= 2*ones(3)

@test Vk([1.0; 2.0; 0.0])== ui(2,Vk.ξ[1], Vk.σ[1], Vk.a[1])(1.0) + ui(2,Vk.ξ[2], Vk.σ[2], Vk.a[2])(2.0)

Vk.a[3] .= 3*ones(5)

@test Vk([1.0; 2.0; -1.0])== ui(2,Vk.ξ[1], Vk.σ[1], Vk.a[1])( 1.0) +
                             ui(2,Vk.ξ[2], Vk.σ[2], Vk.a[2])( 2.0) +
                             uk(2,Vk.ξ[3], Vk.σ[3], Vk.a[3])(-1.0)


@test Vk(5.0) == ui(2,Vk.ξ[1], Vk.σ[1], Vk.a[1])(5.0) +
                             ui(2,Vk.ξ[2], Vk.σ[2], Vk.a[2])(5.0) +
                             uk(2,Vk.ξ[3], Vk.σ[3], Vk.a[3])(5.0)
end


@testset "Component function for Uk type" begin

Vk = Uk(1, 2)

@test Vk.k == 1
@test Vk.p == 2


@test  TransportMap.component(Vk, 1).ξk == zeros(2+2)
@test  TransportMap.component(Vk, 1).σk == ones(2+2)
@test  TransportMap.component(Vk, 1).ak == zeros(2+3)

Vk = Uk(5, 2)

@test Vk.k == 5
@test Vk.p == 2


@test  TransportMap.component(Vk, 5).ξk == zeros(2+2)
@test  TransportMap.component(Vk, 5).σk == ones(2+2)
@test  TransportMap.component(Vk, 5).ak == zeros(2+3)

@test  TransportMap.component(Vk, 2).ξi == zeros(2)
@test  TransportMap.component(Vk, 2).σi == ones(2)
@test  TransportMap.component(Vk, 2).ai == zeros(2+1)
end


@testset "Verify off_diagonal function" begin

# k=1  & p=0
Vk = Uk(1,0)
Vk.a[1] .= randn(2)

@test off_diagonal(Vk, randn()) == 0.0

# k=1 & p= 3
Vk = Uk(1, 3)
Vk.a[1] .= rand(6)

@test off_diagonal(Vk, randn()) == 0.0


# k=3 & p = 0
Vk = Uk(3, 0)
a1 =randn()
a2 = randn()
a3 = rand(2)
Vk.a[1] .= a1
Vk.a[2] .= a2
Vk.a[3] .= a3

z = randn(3)
@test norm(off_diagonal(Vk, z) - (Vk(z) - TransportMap.component(Vk,3)(z[3])))<1e-10

# k=3 & p = 3
Vk = Uk(3, 3)
for i=1:2
Vk.ξ[i] .= randn(3)
Vk.σ[i] .= rand(3)
end

Vk.ξ[3] .= randn(5)
Vk.σ[3] .= rand(5)

a1 =randn(4)
a2 = randn(4)
a3 = rand(6)
Vk.a[1] .= a1
Vk.a[2] .= a2
Vk.a[3] .= a3

z = randn(3)
@test norm(off_diagonal(Vk, z) - (Vk(z) - TransportMap.component(Vk,3)(z[3])))<1e-10
end



@testset "extract and modify coefficients of Uk" begin
    # k=1 and p=0
    Vk = Uk(1, 0)

    @test Vk.k == 1
    @test Vk.p == 0

    modify_a([1.0;2.0], Vk)
    @test extract_a(Vk) == [1.0; 2.0]

    Vk = Uk(1, 3)
    modify_a([1.0;2.0; 3.0; 4.0; 5.0; 6.0], Vk)
    @test extract_a(Vk) == [1.0;2.0; 3.0; 4.0; 5.0; 6.0]


    # k =3 and p=2
    A = collect(1.0:1.0:11.0)
    Vk = Uk(3, 2)

    modify_a(A, Vk)
    @test  extract_a(Vk) == A

    Vk = Uk(3, 2)
    Vk.a[1] = [1.0; 2.0; 3.0]
    Vk.a[2] = [4.0; 5.0; 6.0]
    Vk.a[3] = [7.0; 8.0; 9.0; 10.0; 11.0]

    @test extract_a(Vk) ==A

    # k = 3 and p = 0

    Vk = Uk(3, 0)

    @test Vk.k == 3
    @test Vk.p == 0

    Vk.a[1] .= [1.0]
    Vk.a[2] .= [2.0]
    Vk.a[3] .= [3.0; 4.0]

    @test extract_a(Vk)==collect(1.0:4.0)

    modify_a(collect(5.0:8.0), Vk)

    @test extract_a(Vk)==collect(5.0:8.0)
end
