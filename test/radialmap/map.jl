
@testset "radial map" begin
    γ = 1.0
    λ = 0.2
    δ = 1e-4
    κ = 5.0

    M = RadialMap(4,2, γ=γ, λ=λ, δ=δ, κ=κ)
    @test size(M) == (4,2)

    @test M.γ==1.0
    @test M.λ==0.2
    @test M.δ==1e-4
    @test M.κ==5.0

    for i=1:4
        @test size(M.C[i]) ==(i,2)
    end


    M = RadialMap(2,2)
    M.C[1].a[1] .= ones(2+3)
    M.C[2].a[1] .= 5*ones(2+1)
    M.C[2].a[2] .= ones(2+3)

    x = randn()
    y = randn(2)


    C11 = RadialMapComponent(1,2, [zeros(4)], [ones(4)], [ones(2+3)])
    C22 = RadialMapComponent(2,2, [zeros(2),zeros(4)], [ones(2), ones(4)], [5*ones(2+1), ones(2+3)])

    @test isapprox(M([x; x]), [C11([x]); C22([x; x])], atol = 1e-10)
    @test isapprox(M(y), [C11([y[1]]); C22(y)], atol = 1e-10)


    C = RadialMapComponent(2,2)
    A = randn(8)
    modify_a!(A, C)

    Cprime = ForwardDiff.gradient(x->C(x), y)

    @test isapprox(Cprime[1], C.a[1][1] + C.a[1][2]*rbf′(0.0, 1.0)(y[1]) + C.a[1][3]*rbf′(0.0, 1.0)(y[1]), atol = 1e-10)
    @test isapprox(Cprime[2], 0.0*C.a[2][1] + C.a[2][2]*ψ₀′(0.0, 1.0)(y[2]) + C.a[2][3]*rbf(0.0, 1.0)(y[2]) + C.a[2][4]*rbf(0.0, 1.0)(y[2]) + C.a[2][5]*ψpp1′(0.0, 1.0)(y[2]), atol = 1e-10)
end


@testset "Sparse radial map I" begin
    γ = 1.0
    λ = 0.2
    δ = 1e-4
    κ = 5.0
    # p = [[2];[-1; 1]; []

    M = SparseRadialMap(4,fill(2,4), γ=γ, λ=λ, δ=δ, κ=κ)
    @test size(M) == (4,[fill(2,i) for i=1:4])

    @test M.γ==1.0
    @test M.λ==0.2
    @test M.δ==1e-4
    @test M.κ==5.0

    for i=1:4
        @test size(M.C[i]) ==(i,fill(2,i))
    end
end

@testset "Sparse radial map II" begin
    p = [Int64[-1], [0; -1], [2; 0; 1], [-1 ;2;-1; 0]]
    M = SparseRadialMap(4,p)

    for i=1:4
        @test M.p[i]==p[i]
    end

    z =randn(4)

    @test M(z)[1]==z[1]

    M.C[2].a[1] =randn(1)

    M.C[3].ξ[1] = randn(2)
    M.C[3].σ[1] = rand(2)
    M.C[3].a[1] = randn(3)

    M(z)
    M.C[3].a[2] = randn(1)

    M.C[3].ξ[3] = randn(3)
    M.C[3].σ[3] = rand(3)
    M.C[3].a[3] = randn(4)

    M(z)

    M.C[4].ξ[2] = randn(2)
    M.C[4].σ[2] = rand(2)
    M.C[4].a[2] = randn(3)

    M.C[4].a[4] = randn(2)

    @test norm(M(z)[2] - ( ui(p[2][1],Float64[],Float64[],M.C[2].a[1])(z[1]) + z[2]))<1e-12

    @test norm(M(z)[3] -( ui(p[3][1],M.C[3].ξ[1],M.C[3].σ[1],M.C[3].a[1])(z[1])+
                     ui(p[3][2],M.C[3].ξ[2],M.C[3].σ[2],M.C[3].a[2])(z[2])+
                     uk(p[3][3],M.C[3].ξ[3],M.C[3].σ[3],M.C[3].a[3])(z[3])))<1e-12



    @test norm(M(z)[4] -( ui(p[4][2],M.C[4].ξ[2],M.C[4].σ[2],M.C[4].a[2])(z[2])+
                    uk(p[4][4],M.C[4].ξ[4],M.C[4].σ[4],M.C[4].a[4])(z[4])))<1e-12

end
