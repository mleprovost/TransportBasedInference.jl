using Test

using LinearAlgebra, Statistics
using SpecialFunctions, ForwardDiff
using TransportMap, IterativeSolvers



@testset "Test optimization with iterative solver" begin
    # Test 1
    Ne = 20
    C = SparseRadialMapComponent(2, [0,0])
    ens = EnsembleState(randn(2, Ne).*randn(2, Ne))

    xricardo = optimize_ricardo(C, ens, 0.0, 0.0)
    xiter = iterative(C, ens, 0.0, 0.0)

    @test norm(xricardo-xiter)<1e-3

    # Test  2
    Ne = 200
    C = SparseRadialMapComponent(3, [2,2,0])
    ens = EnsembleState(randn(3, Ne).*randn(3, Ne))

    λ = 0.1


    xricardo = optimize_ricardo(C, ens, λ, 0.0)
    xiter = iterative(C, ens, 0.1, 0.0)

    @test norm(xricardo-xiter)<1e-3

    # Test 3
    Ne = 200
    C = SparseRadialMapComponent(6, [2, -1,2,-1, 2,0])
    ens = EnsembleState(randn(6, Ne).*randn(6, Ne))

    λ = 0.1


    xricardo = optimize_ricardo(C, ens, λ, 0.0)
    xiter = iterative(C, ens, 0.1, 0.0)
    if norm(xricardo-xiter)>1e-3
        xiter = iterative(C, ens, 0.1, 0.0)
    end
    @test norm(xricardo-xiter)<1e-3

end
