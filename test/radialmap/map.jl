using Test

using LinearAlgebra, Statistics
using SpecialFunctions, ForwardDiff
using TransportMap

@testset "Knothe-Rosenblatt map" begin
γ = 1.0
λ = 0.2
δ = 1e-4
κ = 5.0

U = RadialMap(4,2, γ=γ, λ=λ, δ=δ, κ=κ)
@test size(U) == (4,2)

@test U.γ==1.0
@test U.λ==0.2
@test U.δ==1e-4
@test U.κ==5.0

for i=1:4
    @test size(U.U[i]) ==(i,2)
end


U = RadialMap(2,2)
U.U[1].a[1] .= ones(2+3)
U.U[2].a[1] .= 5*ones(2+1)
U.U[2].a[2] .= ones(2+3)


U11 = RadialMapComponent(1,2, [zeros(4)], [ones(4)], [ones(2+3)])
U22 = RadialMapComponent(2,2, [zeros(2),zeros(4)], [ones(2), ones(4)], [5*ones(2+1), ones(2+3)])

@test U([1.0; 1.0]) == [U11([1.0]); U22([1.0; 1.0])]

@test U([2.0; 1.0]) == [U11([2.0]); U22([2.0; 1.0])]

@test U([1.0; 2.0]) == [U11([1.0]); U22([1.0; 2.0])]


U = RadialMapComponent(2,2)
U.a[2] .= ones(5)

@test ForwardDiff.gradient(x->U(x), [1.0;1.0])[2] == (ψ₀′(0.0, 1.0)(1.0)+ 2*rbf(0.0, 1.0)(1.0)+ψpp1′(0.0, 1.0)(1.0))


end


@testset "Sparse Knothe-Rosenblatt map I" begin
    γ = 1.0
    λ = 0.2
    δ = 1e-4
    κ = 5.0
    # p = [[2];[-1; 1]; []

    U = SparseRadialMap(4,fill(2,4), γ=γ, λ=λ, δ=δ, κ=κ)
    @test size(U) == (4,[fill(2,i) for i=1:4])

    @test U.γ==1.0
    @test U.λ==0.2
    @test U.δ==1e-4
    @test U.κ==5.0

    for i=1:4
        @test size(U.U[i]) ==(i,fill(2,i))
    end
end

@testset "Sparse Knothe-Rosenblatt map I" begin
    p = [Int64[-1], [0; -1], [2; 0; 1], [-1 ;2;-1; 0]]
    U = SparseRadialMap(4,p)

    for i=1:4
        @test U.p[i]==p[i]
    end

    z =randn(4)

    @test U(z)[1]==z[1]

    U.U[2].a[1] =randn(1)

    U.U[3].ξ[1] = randn(2)
    U.U[3].σ[1] = rand(2)
    U.U[3].a[1] = randn(3)

    U(z)
    U.U[3].a[2] = randn(1)

    U.U[3].ξ[3] = randn(3)
    U.U[3].σ[3] = rand(3)
    U.U[3].a[3] = randn(4)

    U(z)

    U.U[4].ξ[2] = randn(2)
    U.U[4].σ[2] = rand(2)
    U.U[4].a[2] = randn(3)

    U.U[4].a[4] = randn(2)

    @test norm(U(z)[2] - ( ui(p[2][1],Float64[],Float64[],U.U[2].a[1])(z[1]) + z[2]))<1e-12

    @test norm(U(z)[3] -( ui(p[3][1],U.U[3].ξ[1],U.U[3].σ[1],U.U[3].a[1])(z[1])+
                     ui(p[3][2],U.U[3].ξ[2],U.U[3].σ[2],U.U[3].a[2])(z[2])+
                     uk(p[3][3],U.U[3].ξ[3],U.U[3].σ[3],U.U[3].a[3])(z[3])))<1e-12



    @test norm(U(z)[4] -( ui(p[4][2],U.U[4].ξ[2],U.U[4].σ[2],U.U[4].a[2])(z[2])+
                    uk(p[4][4],U.U[4].ξ[4],U.U[4].σ[4],U.U[4].a[4])(z[4])))<1e-12

end
