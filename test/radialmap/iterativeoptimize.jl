
@testset "Test optimization with iterative solver" begin
    # Test 1
    Ne = 20
    C = SparseRadialMapComponent(2, [0,0])
    X = randn(2, Ne).*randn(2, Ne)

    xricardo = optimize(C, X, 0.0, 0.0)
    xiter = iterative(C, X, 0.0, 0.0)

    @test norm(xricardo-xiter)<1e-3

    # Test  2
    Ne = 200
    C = SparseRadialMapComponent(3, [2,2,0])
    X = randn(3, Ne).*randn(3, Ne)

    λ = 0.1


    xricardo = optimize(C, X, λ, 0.0)
    xiter = iterative(C, X, 0.1, 0.0)

    @test norm(xricardo-xiter)<1e-3

    # Test 3
    Ne = 200
    C = SparseRadialMapComponent(6, [2, -1,2,-1, 2,0])
    X = randn(6, Ne).*randn(6, Ne)

    λ = 0.1


    xricardo = optimize(C, X, λ, 0.0)
    xiter = iterative(C, X, 0.1, 0.0)
    if norm(xricardo-xiter)>1e-3
        xiter = iterative(C, X, 0.1, 0.0)
    end
    @test norm(xricardo-xiter)<1e-3

end
