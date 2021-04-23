
@testset "weights ui" begin
    x = randn()

    um1 = ui(-1)
    woff = Float64[]
    weights(um1, 2.0, woff)
    @test woff ==Float64[]

    u4 = ui(0, zeros(0), ones(0), [0.0])

    woff = zeros(1)

    weights(u4, 2.0, woff)
    @test woff[1]==2.0


    u4 = ui(2, zeros(2), ones(2), [0.0; 0.0; 0.0])

    woff = zeros(3)

    weights(u4, x, woff)
    @test isapprox(woff[1], x, atol = 1e-10)
    @test isapprox(woff[2], rbf(0.0, 1.0)(x), atol = 1e-10)
    @test isapprox(woff[3], rbf(0.0, 1.0)(x), atol = 1e-10)


    u4.ξi[1] = -5.0
    u4.σi[1] = 5.0

    u4.ξi[2] = 2.0
    u4.σi[2] = 2.0


    weights(u4, x, woff)
    @test isapprox(woff[1], x, atol = 1e-10)
    @test isapprox(woff[2], rbf(-5.0, 5.0)(x), atol = 1e-10)
    @test isapprox(woff[3], rbf(2.0, 2.0)(x), atol = 1e-10)
end


@testset "weights uk" begin
    x = randn()
    um1 = uk(-1)
    wdiag = Float64[]
    w∂k = Float64[]
    weights(um1, 2.0, wdiag, w∂k)
    @test wdiag ==Float64[]
    @test w∂k ==Float64[]

    u1 = uk(0)
    wdiag = zeros(2)
    w∂k = zeros(1)
    weights(u1, 2.0, wdiag, w∂k)
    @test wdiag == [1.0; 2.0]
    @test w∂k[1] === 1.0

    u4 = uk(0)
    wdiag = zeros(2)
    w∂k = zeros(1)

    weights(u4, 2.0, wdiag, w∂k)

    @test wdiag == [1.0; 2.0]
    @test w∂k[1] === 1.0

    u4 = uk(3)
    wdiag = zeros(3+3)
    w∂k = zeros(3+2)

    weights(u4, x, wdiag, w∂k)
    @test size(wdiag,1)== 3+3
    @test size(w∂k,1)== 3+2
    @test norm(wdiag - [1.0;
                        ψ₀(0.0, 1.0)(x);
                        ψj(0.0, 1.0)(x);
                        ψj(0.0, 1.0)(x);
                        ψj(0.0, 1.0)(x);
                        ψpp1(0.0, 1.0)(x)])<1e-10

    @test norm(w∂k -[ψ₀′(0.0, 1.0)(x);
                     rbf(0.0, 1.0)(x);
                     rbf(0.0, 1.0)(x);
                     rbf(0.0, 1.0)(x);
                   ψpp1′(0.0, 1.0)(x)])<1e-10


    u4 = uk(3)
    wdiag = zeros(3+3)
    w∂k = zeros(3+2)

    u4.ξk .= [-1.0; 2.0; -3.0; 4.0; -5.0]
    u4.σk .= [1.0; 2.0; 3.0; 4.0; 5.0]

    weights(u4, x, wdiag, w∂k)
    @test size(wdiag,1)== 3+3
    @test size(w∂k,1)== 3+2

    @test norm(wdiag - [1.0;
                        ψ₀(-1.0, 1.0)(x);
                         ψj(2.0, 2.0)(x);
                        ψj(-3.0, 3.0)(x);
                         ψj(4.0, 4.0)(x);
                      ψpp1(-5.0, 5.0)(x)])<1e-10
end




@testset "Weights of a RadialMapComponent map" begin
    C = RadialMap(1, 0)
    x = randn()
    y = randn(3)

    woff, wdiag, w∂k = create_weights(C)
    # wdiag = zeros(2)
    # w∂k = zeros(1)
    # woff = zeros(0)

    # w, w∂k = weights(TransportMap.component(C.U[1],1), [2.0], w, w∂k)
    weights(AdaptiveTransportMap.component(C.U[1],1), x, wdiag, w∂k)

    @test size(wdiag,1) == 2
    @test size(w∂k, 1) == 1
    @test size(woff, 1) == 0

    @test wdiag == [1.0; x]
    @test w∂k   == [1.0]

    #
    C = RadialMap(1, 1)

    woff, wdiag, w∂k = create_weights(C)

    weights(C, [x], woff,  wdiag, w∂k)
    #
    @test size(woff,1)==0
    @test size(wdiag,1) == 1+3
    @test size(w∂k, 1) == 1+2
    #
    @test woff == Float64[]
    @test isapprox(wdiag, [1.0; ψ₀(0.0, 1.0)(x); ψj(0.0, 1.0)(x); ψpp1(0.0, 1.0)(x)], atol = 1e-10)
    @test isapprox(w∂k, [ψ₀′(0.0, 1.0)(x); rbf(0.0, 1.0)(x); ψpp1′(0.0, 1.0)(x)], atol = 1e-10)
    #
    #
    C = RadialMap(2, 0)

    woff, wdiag, w∂k = create_weights(C)

    #
    weights(C, y[1:2], woff, wdiag, w∂k)
    #
    @test size(woff, 1) == 1
    @test size(wdiag,1) == 2+2
    @test size(w∂k, 1) == 1+1
    #
    #
    @test isapprox(wdiag, [1.0; y[1]; 1.0; y[2]], atol = 1e-10)
    @test isapprox(w∂k, [1.0; 1.0], atol = 1e-10)
    @test isapprox(woff, [y[1]], atol = 1e-10)
    #
    #
    C = RadialMap(3, 2)

    woff, wdiag, w∂k = create_weights(C)

    weights(C, y, woff, wdiag, w∂k)
    #
    @test size(woff,1) == 2*(2+1)
    @test size(wdiag,1) == 3*(2+3)
    @test size(w∂k, 1) == 3*(2+2)
    #
    #
    @test isapprox(woff, [y[1]; rbf(0.0, 1.0)(y[1]); rbf(0.0, 1.0)(y[1]);
                          y[2]; rbf(0.0, 1.0)(y[2]); rbf(0.0, 1.0)(y[2])], atol = 1e-10)

    @test isapprox(wdiag, [1.0; ψ₀(0.0, 1.0)(y[1]); ψj(0.0, 1.0)(y[1]);
                           ψj(0.0, 1.0)(y[1]); ψpp1(0.0, 1.0)(y[1]);
                           1.0; ψ₀(0.0, 1.0)(y[2]); ψj(0.0, 1.0)(y[2]); ψj(0.0, 1.0)(y[2]);ψpp1(0.0, 1.0)(y[2]);
                           1.0; ψ₀(0.0, 1.0)(y[3]); ψj(0.0, 1.0)(y[3]); ψj(0.0, 1.0)(y[3]); ψpp1(0.0, 1.0)(y[3])],
                           atol = 1e-10)

    @test isapprox(w∂k, [ψ₀′(0.0, 1.0)(y[1]); rbf(0.0, 1.0)(y[1]) ; rbf(0.0, 1.0)(y[1]); ψpp1′(0.0, 1.0)(y[1]);
                        ψ₀′(0.0, 1.0)(y[2]); rbf(0.0, 1.0)(y[2]) ; rbf(0.0, 1.0)(y[2]); ψpp1′(0.0, 1.0)(y[2]);
                        ψ₀′(0.0, 1.0)(y[3]); rbf(0.0, 1.0)(y[3]) ; rbf(0.0, 1.0)(y[3]); ψpp1′(0.0, 1.0)(y[3])],
                        atol = 1e-10)
    Nx = 5
    Ne = 50
    p = 2

    X = randn(Nx, Ne) .* randn(Nx, Ne)
    C = RadialMap(Nx, p)

    W = create_weights(C, X)

    woff  = zero(W.woff)
    wdiag = zero(W.wdiag)
    w∂k   = zero(W.w∂k)
    weights(C, X, woff, wdiag, w∂k)

    weights(C, X, W)

    @test isapprox(woff, W.woff, atol = 1e-10)
    @test isapprox(wdiag, W.wdiag, atol = 1e-10)
    @test isapprox(w∂k, W.w∂k, atol = 1e-10)

    woval, wval, w∂kval = create_weights(C)
    weights(C, zeros(Nx), woval, wval, w∂kval)
    weights(C, zeros(Nx,Ne), woff, wdiag, w∂k)


    @test woff[:,1] == woval
    @test wdiag[:,1] == wval
    @test w∂k[:,1] == w∂kval
end


@testset "Weights SparseRadialMapComponentMap I" begin
    x = randn()

    ## k = 1 and p = -1
    C = SparseRadialMapComponent(1, [-1])

    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==0
    @test nd ==0
    @test n∂ ==0

    wo, wd, w∂ = weights(C, [x])
    @test wo == Float64[]
    @test wd == Float64[]
    @test w∂ == Float64[]

    ## Nx = 1 and p = 0
    C = SparseRadialMapComponent(1, [0])

    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==0
    @test nd ==1
    @test n∂ ==1

    wo, wd, w∂ = weights(C, [x])
    @test wo == Float64[]
    @test isapprox(wd, [x], atol = 1e-10)
    @test w∂ == [1.0]


    z = randn(1,5)
    wo, wd, w∂ = weights(C, z)
    @test wo == Float64[]
    @test isapprox(wd, z, atol = 1e-10)
    @test w∂ == ones(1,5)

    ## Nx = 1 and p = 2
    C = SparseRadialMapComponent(1, [2])

    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==0
    @test nd ==2+2
    @test n∂ ==2+2

    wo, wd, w∂ = weights(C, [x])
    @test wo == Float64[]
    @test isapprox(wd, [ψ₀(0.0, 1.0)(x); ψj(0.0, 1.0)(x);ψj(0.0, 1.0)(x);ψpp1(0.0, 1.0)(x)], atol = 1e-10)
    @test isapprox(w∂, [ψ₀′(0.0, 1.0)(x); rbf(0.0, 1.0)(x);rbf(0.0, 1.0)(x);ψpp1′(0.0, 1.0)(x)], atol = 1e-10)

    z = randn(1,5)

    wo, wd, w∂ = weights(C, z)
    @test wo == Float64[]

    for i=1:5
        zi  = z[1,i]
        @test wd[:,i] == [ψ₀(0.0, 1.0)(zi); ψj(0.0, 1.0)(zi);ψj(0.0, 1.0)(zi);ψpp1(0.0, 1.0)(zi)]
        @test w∂[:,i] == [ψ₀′(0.0, 1.0)(zi); rbf(0.0, 1.0)(zi);rbf(0.0, 1.0)(zi);ψpp1′(0.0, 1.0)(zi)]
    end

end

@testset "Weights SparseRadialMapComponent II" begin
    ## k=2 and p = [-1 -1]
    y = randn(2)
    C = SparseRadialMapComponent(2, [-1; -1])

    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==0
    @test nd ==0
    @test n∂ ==0


    wo, wd, w∂ = weights(C, [1.0; 2.0])
    @test wo == Float64[]
    @test wd == Float64[]
    @test w∂ == Float64[]

    wo, wd, w∂ = weights(C, randn(2,5))
    @test wo == Float64[]
    @test wd == Float64[]
    @test w∂ == Float64[]

    ## k = 2 and p = [-1 0]
    C = SparseRadialMapComponent(2, [-1; 0])
    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==0
    @test nd ==1
    @test n∂ ==1

    wo, wd, w∂ = weights(C, y)
    @test wo == Float64[]
    @test wd == [y[2]]
    @test w∂ == [1.0]

    z = randn(2, 5)
    wo, wd, w∂ = weights(C, deepcopy(z))
    @test wo == Float64[]
    @test isapprox(wd[1,:], z[2,:], atol = 1e-10)
    @test isapprox(w∂[1,:], ones(5), atol = 1e-10)

    ## k = 2 and p = [-1 1]
    C = SparseRadialMapComponent(2, [-1; 1])
    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==0
    @test nd ==3
    @test n∂ ==3

    wo, wd, w∂ = weights(C, y)
    @test wo == Float64[]
    @test isapprox(wd, [ψ₀(0.0, 1.0)(y[2]); ψj(0.0, 1.0)(y[2]);ψpp1(0.0, 1.0)(y[2])], atol = 1e-10)
    @test isapprox(w∂, [ψ₀′(0.0, 1.0)(y[2]); rbf(0.0, 1.0)(y[2]);ψpp1′(0.0, 1.0)(y[2])], atol = 1e-10)

    z = randn(2,5)

    wo, wd, w∂ = weights(C, z)
    @test wo == Float64[]

    for i=1:5
        zi  = z[:,i]
        @test wd[:,i] == [ψ₀(0.0, 1.0)(zi[2]); ψj(0.0, 1.0)(zi[2]);ψpp1(0.0, 1.0)(zi[2])]
        @test w∂[:,i] == [ψ₀′(0.0, 1.0)(zi[2]); rbf(0.0, 1.0)(zi[2]);ψpp1′(0.0, 1.0)(zi[2])]
    end

    ## k = 2 and p = [0 1]

    C = SparseRadialMapComponent(2, [0; 1])
    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==1
    @test nd ==3
    @test n∂ ==3

    wo, wd, w∂ = weights(C, y)
    @test isapprox(wo, [y[1]], atol = 1e-10)
    @test isapprox(wd, [ψ₀(0.0, 1.0)(y[2]); ψj(0.0, 1.0)(y[2]);ψpp1(0.0, 1.0)(y[2])], atol = 1e-10)
    @test isapprox(w∂, [ψ₀′(0.0, 1.0)(y[2]); rbf(0.0, 1.0)(y[2]);ψpp1′(0.0, 1.0)(y[2])], atol = 1e-10)

    z = randn(2,5)

    wo, wd, w∂ = weights(C, deepcopy(z))


    for i=1:5
        zi  = z[:,i]
        @test norm(wo[:,i] - [zi[1]])<1e-10

        @test norm(wd[:,i]  - [ψ₀(0.0, 1.0)(zi[2]); ψj(0.0, 1.0)(zi[2]);ψpp1(0.0, 1.0)(zi[2])])<1e-10
        @test norm(w∂[:,i]  - [ψ₀′(0.0, 1.0)(zi[2]); rbf(0.0, 1.0)(zi[2]);ψpp1′(0.0, 1.0)(zi[2])])<1e-10
    end

    # k = 2 and p = [2 1]

    C = SparseRadialMapComponent(2, [2; 1])
    C.ξ[1] .= randn(2)
    C.ξ[2] .= randn(3)
    C.σ[1] .= rand(2)
    C.σ[2] .= rand(3)

    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==3
    @test nd ==3
    @test n∂ ==3

    wo, wd, w∂ = weights(C, y)
    @test isapprox(wo, [y[1]; rbf(C.ξ[1][1], C.σ[1][1])(y[1]); rbf(C.ξ[1][2], C.σ[1][2])(y[1])], atol = 1e-10)
    @test isapprox(wd, [ψ₀(C.ξ[2][1], C.σ[2][1])(y[2]); ψj(C.ξ[2][2], C.σ[2][2])(y[2]);
                 ψpp1(C.ξ[2][3], C.σ[2][3])(y[2])], atol = 1e-10)
    @test isapprox(w∂, [ψ₀′(C.ξ[2][1], C.σ[2][1])(y[2]); rbf(C.ξ[2][2], C.σ[2][2])(y[2]);
                       ψpp1′(C.ξ[2][3], C.σ[2][3])(y[2])], atol = 1e-10)


    z = randn(2,5)

    wo, wd, w∂ = weights(C, deepcopy(z))


    for i=1:5
        zi  = z[:,i]
        @test norm(wo[:,i]  - [zi[1]; rbf(C.ξ[1][1], C.σ[1][1])(zi[1]); rbf(C.ξ[1][2], C.σ[1][2])(zi[1])])<1e-14

        @test norm(wd[:,i]  - [ψ₀(C.ξ[2][1], C.σ[2][1])(zi[2]); ψj(C.ξ[2][2], C.σ[2][2])(zi[2]);ψpp1(C.ξ[2][3], C.σ[2][3])(zi[2])])<1e-14
        @test norm(w∂[:,i]  - [ψ₀′(C.ξ[2][1], C.σ[2][1])(zi[2]); rbf(C.ξ[2][2], C.σ[2][2])(zi[2]);ψpp1′(C.ξ[2][3], C.σ[2][3])(zi[2])])<1e-14
    end


    ## k = 4 p = [-1 2 0 2]

    C = SparseRadialMapComponent(4, [-1; 2; 0; 2])

    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==3+1
    @test nd ==4
    @test n∂ ==4

    y = randn(4)

    wo, wd, w∂ = weights(C, deepcopy(y))

    @test norm(wo - [y[2]; rbf(0.0,1.0)(y[2]); rbf(0.0,1.0)(y[2]); y[3]])<1e-14
    @test norm(wd - [ψ₀(0.0, 1.0)(y[4]); ψj(0.0, 1.0)(y[4]); ψj(0.0, 1.0)(y[4]);ψpp1(0.0, 1.0)(y[4])])<1e-14
    @test norm(w∂ - [ψ₀′(0.0, 1.0)(y[4]); rbf(0.0, 1.0)(y[4]); rbf(0.0, 1.0)(y[4]);ψpp1′(0.0, 1.0)(y[4])])<1e-14

    z = randn(4, 5)

    wo, wd, w∂ = weights(C, deepcopy(z))
    for i=1:5
        zi  = z[:,i]
        @test norm(wo[:,i]  - [zi[2]; rbf(0.0,1.0)(zi[2]); rbf(0.0,1.0)(zi[2]); zi[3]])<1e-14

        @test norm(wd[:,i]  - [ψ₀(0.0, 1.0)(zi[4]); ψj(0.0, 1.0)(zi[4]); ψj(0.0, 1.0)(zi[4]);ψpp1(0.0, 1.0)(zi[4])])<1e-14
        @test norm(w∂[:,i]  - [ψ₀′(0.0, 1.0)(zi[4]); rbf(0.0, 1.0)(zi[4]); rbf(0.0, 1.0)(zi[4]);ψpp1′(0.0, 1.0)(zi[4])])<1e-14
    end

    ## k = 5 and p = [2 -1 0 -1 0]

    C = SparseRadialMapComponent(5, [2; -1; 0; -1; 0])

    no, nd, n∂ = ncoeff(C.Nx, C.p)

    @test no ==3+1
    @test nd ==1
    @test n∂ ==1

    y = randn(5)
    wo, wd, w∂ = weights(C, y)

    @test norm(wo - [y[1]; rbf(0.0,1.0)(y[1]); rbf(0.0,1.0)(y[1]); y[3]])<1e-14
    @test norm(wd - [y[5]])<1e-14
    @test norm(w∂ - [1.0])<1e-14

    z = randn(5, 10)

    wo, wd, w∂ = weights(C, z)
    for i=1:10
        zi  = z[:,i]
        @test norm(wo[:,i]  - [zi[1]; rbf(0.0,1.0)(zi[1]); rbf(0.0,1.0)(zi[1]); zi[3]])<1e-14

        @test norm(wd[:,i]  - [zi[5]])<1e-14
        @test norm(w∂[:,i]  - [1.0])<1e-14
    end

end
