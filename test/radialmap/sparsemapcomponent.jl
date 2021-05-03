
@testset "SparseRadialMapComponent (Nx,p) = (1,-1)" begin
    x = randn()

    C = SparseRadialMapComponent(1,-1)

    @test C.Nx == 1
    @test C.p == [-1]
    @test C.activedim == Int64[]
    @test size(C.ξ, 1)==1
    @test size(C.ξ[1], 1)==0

    @test size(C.σ, 1)==1
    @test size(C.σ[1], 1)==0

    @test size(C.a,1)==1
    @test size(C.a[1],1)==0

    @test isapprox(C(x), x, atol = 1e-10)
end

@testset "SparseRadialMapComponent (Nx,p) = (1,0)" begin

    C = SparseRadialMapComponent(1,0)
    x = randn()

    @test C.Nx == 1
    @test C.p == [0]
    @test C.activedim == Int64[1]
    @test size(C.ξ, 1)==1
    @test size(C.ξ[1], 1)==0

    @test size(C.σ, 1)==1
    @test size(C.σ[1], 1)==0

    @test size(C.a,1)==1
    @test size(C.a[1],1)==2

    C.a[1]=[2.0;3.0]
    @test isapprox(C(x), 2.0 + 3.0*x, atol = 1e-10)
end

@testset "SparseRadialMapComponent (Nx,p) = (1,1)" begin

    C = SparseRadialMapComponent(1,1)
    x = randn()

    @test C.Nx == 1
    @test C.p == [1]
    @test C.activedim == Int64[1]

    @test size(C.ξ,1) ==1
    @test size(C.ξ[1],1) ==1+2

    @test size(C.σ,1) ==1
    @test size(C.σ[1],1) ==1+2

    @test size(C.ξ,1) ==1
    @test size(C.ξ[1],1) ==1+2

    @test size(C.a,1) ==1
    @test size(C.a[1],1) ==1+3

    C.a[1] .= [1.0; 2.0; 3.0; -1.0]

    @test isapprox(C(x), 1.0 + 2.0*ψ₀(0.0,1.0)(x) +
                     3.0*ψj(0.0,1.0)(x) - 1.0*ψpp1(0.0,1.0)(x), atol = 1e-10)

    @test isapprox(C([x]), 1.0 + 2.0*ψ₀(0.0,1.0)(x) +
                      3.0*ψj(0.0,1.0)(x) - 1.0*ψpp1(0.0,1.0)(x), atol = 1e-10)
end


@testset "SparseRadialMapComponent (Nx,p) = (Nx,-1) with Nx>1" begin

    C = SparseRadialMapComponent(3, -1)
    x = randn()
    y = randn(3)

    @test C.Nx == 3
    @test C.p == [-1; -1 ; -1]
    @test C.activedim == Int64[]

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.a,1)==3

    @test size(C.ξ[1],1)==0
    @test size(C.σ[1],1)==0
    @test size(C.a[1],1)==0

    @test size(C.ξ[2],1)==0
    @test size(C.σ[2],1)==0
    @test size(C.a[2],1)==0

    @test size(C.ξ[3],1)==0
    @test size(C.σ[3],1)==0
    @test size(C.a[3],1)==0

    @test isapprox(C(x), x, atol = 1e-10)
    @test isapprox(C(y), y[3], atol = 1e-10)
end


@testset "SparseRadialMapComponent (Nx,p) = (Nx,0) with Nx>1" begin

    C = SparseRadialMapComponent(3, 0)
    x = randn()
    y = randn(3)

    @test C.Nx == 3
    @test C.p == [0 ; 0 ; 0]
    @test C.activedim == collect(1:3)

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.a,1)==3

    @test size(C.ξ[1],1)==0
    @test size(C.σ[1],1)==0
    @test size(C.a[1],1)==1

    @test size(C.ξ[2],1)==0
    @test size(C.σ[2],1)==0
    @test size(C.a[2],1)==1

    @test size(C.ξ[3],1)==0
    @test size(C.σ[3],1)==0
    @test size(C.a[3],1)==2

    @test isapprox(C(x), 0.0, atol = 1e-10)


    C.a[1][1] = 2.0

    @test isapprox(C([x;0.0;0.0]), 2.0*x, atol = 1e-10)

    C.a[2][1] = 3.0
    @test isapprox(C([y[1];y[2];0.0]), 2*y[1] + 3*y[2], atol = 1e-10)

    C.a[3][1] = 4.0
    @test isapprox(C(y), 2.0*y[1] + 3.0*y[2] + 4.0, atol = 1e-10)
    @test isapprox(C(x), 2.0*x+3.0*x+ 4.0, atol = 1e-10)

    C.a[3][2] = -1.0
    @test isapprox(C(y), 2.0*y[1] + 3.0*y[2]+ 4.0 - 1.0*y[3], atol = 1e-10)
    @test isapprox(C(x), 2.0*x + 3.0*x + 4.0 + -1.0*x, atol = 1e-10)

end


@testset "RadialMapComponent (Nx,p) = (Nx,p) with Nx>1 and p>1" begin

    C = SparseRadialMapComponent(3, 2)
    x = randn()
    y = randn(3)

    @test C.Nx == 3
    @test C.p == [2;2;2]
    @test C.activedim == collect(1:3)

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.a,1)==3

    @test size(C.ξ[1],1)==2
    @test size(C.σ[1],1)==2
    @test size(C.a[1],1)==2+1

    @test size(C.ξ[2],1)==2
    @test size(C.σ[2],1)==2
    @test size(C.a[2],1)==2+1

    @test size(C.ξ[3],1)==2+2
    @test size(C.σ[3],1)==2+2
    @test size(C.a[3],1)==2+3

    @test C.ξ[1]==zeros(2)
    @test C.ξ[2]==zeros(2)
    @test C.ξ[3]==zeros(2+2)

    @test C.σ[1]==ones(2)
    @test C.σ[2]==ones(2)
    @test C.σ[3]==ones(2+2)

    @test C.a[1]==zeros(2+1)
    @test C.a[2]==zeros(2+1)
    @test C.a[3]==zeros(2+3)

    C.a[1] .= ones(3)

    @test isapprox(C(x), ui(2,C.ξ[1], C.σ[1], C.a[1])(x), atol = 1e-10)

    @test isapprox(C([x; 0.0; 0.0]), ui(2,C.ξ[1], C.σ[1], C.a[1])(x), atol = 1e-10)

    C.a[2] .= 2*ones(3)

    @test isapprox(C([y[1]; y[2];  0.0]), ui(2,C.ξ[1], C.σ[1], C.a[1])(y[1]) + ui(2,C.ξ[2], C.σ[2], C.a[2])(y[2]), atol = 1e-10)

    C.a[3] .= 3*ones(5)

    @test isapprox(C(y), ui(2,C.ξ[1], C.σ[1], C.a[1])(y[1]) +
                                 ui(2,C.ξ[2], C.σ[2], C.a[2])(y[2]) +
                                 uk(2,C.ξ[3], C.σ[3], C.a[3])(y[3]), atol = 1e-10)


    @test isapprox(C(x), ui(2,C.ξ[1], C.σ[1], C.a[1])(x) +
                                 ui(2,C.ξ[2], C.σ[2], C.a[2])(x) +
                                 uk(2,C.ξ[3], C.σ[3], C.a[3])(x), atol = 1e-10)
end



@testset "RadialMapComponent (Nx,p) = (Nx,p) with Nx>1 and p =[-1  0 -1]" begin

    C = SparseRadialMapComponent(3, [-1; 0; -1])
    x = randn()
    y = randn(3)

    @test C.Nx == 3
    @test C.p == [-1; 0; -1]
    @test C.activedim == Int64[2]

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.a,1)==3

    @test size(C.ξ[1],1)==0
    @test size(C.σ[1],1)==0
    @test size(C.a[1],1)==0

    @test size(C.ξ[2],1)==0
    @test size(C.σ[2],1)==0
    @test size(C.a[2],1)==1

    @test size(C.ξ[3],1)==0
    @test size(C.σ[3],1)==0
    @test size(C.a[3],1)==0

    @test C.ξ[1]==zeros(0)
    @test C.ξ[2]==zeros(0)
    @test C.ξ[3]==zeros(0)

    @test C.σ[1]==ones(0)
    @test C.σ[2]==ones(0)
    @test C.σ[3]==ones(0)

    @test C.a[1]==zeros(0)
    @test C.a[2]==zeros(1)
    @test C.a[3]==zeros(0)

    C.a[2] .= [randn()]

    @test isapprox(C(x), ui(0,C.ξ[2], C.σ[2], C.a[2])(x)+uk(-1,C.ξ[3], C.σ[3], C.a[3])(x), atol = 1e-10)
    @test isapprox(C(y), ui(-1,C.ξ[1], C.σ[1], C.a[1])(y[1])+
                         ui(0,C.ξ[2], C.σ[2], C.a[2])(y[2])+
                         uk(-1,C.ξ[3], C.σ[3], C.a[3])(y[3]), atol = 1e-10)

end



@testset "RadialMapComponent (Nx,p) = (Nx,p) with Nx>1 and p =[1  0 -1]" begin

    C = SparseRadialMapComponent(3, [1; 0; -1])
    x = randn()
    y = randn(3)

    @test C.Nx == 3
    @test C.p == [1; 0; -1]
    @test C.activedim == [1; 2]

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.a,1)==3

    @test size(C.ξ[1],1)==1
    @test size(C.σ[1],1)==1
    @test size(C.a[1],1)==2

    @test size(C.ξ[2],1)==0
    @test size(C.σ[2],1)==0
    @test size(C.a[2],1)==1

    @test size(C.ξ[3],1)==0
    @test size(C.σ[3],1)==0
    @test size(C.a[3],1)==0

    @test C.ξ[1]==zeros(1)
    @test C.ξ[2]==zeros(0)
    @test C.ξ[3]==zeros(0)

    @test C.σ[1]==ones(1)
    @test C.σ[2]==ones(0)
    @test C.σ[3]==ones(0)

    @test C.a[1]==zeros(2)
    @test C.a[2]==zeros(1)
    @test C.a[3]==zeros(0)

    C.a[1] .= randn(2)
    @test isapprox(C(x), ui(1,C.ξ[1], C.σ[1], C.a[1])(x)+
                         ui(0,C.ξ[2], C.σ[2], C.a[2])(x)+
                         uk(-1,C.ξ[3], C.σ[3], C.a[3])(x), atol = 1e-10)
    C.a[2] .= [randn()]
    @test isapprox(C(x), ui(1,C.ξ[1], C.σ[1], C.a[1])(x) +
                         ui(0,C.ξ[2], C.σ[2], C.a[2])(x) + x, atol = 1e-10)

    @test isapprox(C(y), ui(1,C.ξ[1], C.σ[1], C.a[1])(y[1]) +
                         ui(0,C.ξ[2], C.σ[2], C.a[2])(y[2]) + y[3], atol = 1e-10)

end

@testset "RadialMapComponent (Nx,p) = (Nx,p) with Nx>1 and p =[-1  0 0]" begin

    C = SparseRadialMapComponent(3, [-1; 0; 0])
    x = randn()
    y = randn(3)

    @test C.Nx == 3
    @test C.p == [-1; 0; 0]
    @test C.activedim == [2; 3]

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.a,1)==3

    @test size(C.ξ[1],1)==0
    @test size(C.σ[1],1)==0
    @test size(C.a[1],1)==0

    @test size(C.ξ[2],1)==0
    @test size(C.σ[2],1)==0
    @test size(C.a[2],1)==1

    @test size(C.ξ[3],1)==0
    @test size(C.σ[3],1)==0
    @test size(C.a[3],1)==2

    @test C.ξ[1]==zeros(0)
    @test C.ξ[2]==zeros(0)
    @test C.ξ[3]==zeros(0)

    @test C.σ[1]==ones(0)
    @test C.σ[2]==ones(0)
    @test C.σ[3]==ones(0)

    @test C.a[1]==zeros(0)
    @test C.a[2]==zeros(1)
    @test C.a[3]==zeros(2)

    C.a[2] .= [randn()]
    @test isapprox(C(x), ui(0,C.ξ[2], C.σ[2], C.a[2])(x), atol = 1e-10)
    @test isapprox(C(y), ui(0,C.ξ[2], C.σ[2], C.a[2])(y[2]), atol = 1e-10)

    C.a[3] .= randn(2)
    @test isapprox(C(x), ui(0,C.ξ[2], C.σ[2], C.a[2])(x) +
                         uk(0,C.ξ[3], C.σ[3], C.a[3])(x), atol = 1e-10)

    @test isapprox(C(y), ui(0,C.ξ[2], C.σ[2], C.a[2])(y[2]) +
                         uk(0,C.ξ[3], C.σ[3], C.a[3])(y[3]), atol = 1e-10)

end

@testset "RadialMapComponent (Nx,p) = (Nx,p) with Nx>1 and p =[2  -1  1]" begin

    C = SparseRadialMapComponent(3, [2; -1; 1])
    x = randn()
    y = randn(3)

    @test C.Nx == 3
    @test C.p == [2; -1; 1]
    @test C.activedim == [1; 3]

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.a,1)==3

    @test size(C.ξ[1],1)==2
    @test size(C.σ[1],1)==2
    @test size(C.a[1],1)==3

    @test size(C.ξ[2],1)==0
    @test size(C.σ[2],1)==0
    @test size(C.a[2],1)==0

    @test size(C.ξ[3],1)==3
    @test size(C.σ[3],1)==3
    @test size(C.a[3],1)==4

    @test C.ξ[1]==zeros(2)
    @test C.ξ[2]==zeros(0)
    @test C.ξ[3]==zeros(3)

    @test C.σ[1]==ones(2)
    @test C.σ[2]==ones(0)
    @test C.σ[3]==ones(3)

    @test C.a[1]==zeros(3)
    @test C.a[2]==zeros(0)
    @test C.a[3]==zeros(4)

    @test isapprox(C(x), 0.0, atol =  1e-10)
    @test isapprox(C(y), 0.0, atol = 1e-10)

    C.a[1] .= randn(3)
    @test isapprox(C(x), ui(2,C.ξ[1], C.σ[1], C.a[1])(x), atol  = 1e-10)
    @test isapprox(C(y), ui(2,C.ξ[1], C.σ[1], C.a[1])(y[1]), atol = 1e-10)

    C.a[3] .= randn(4)
    @test isapprox(C(x), ui(2,C.ξ[1], C.σ[1], C.a[1])(x)+uk(1,C.ξ[3], C.σ[3], C.a[3])(x), atol = 1e-10)

    @test isapprox(C(y), ui(2,C.ξ[1], C.σ[1], C.a[1])(y[1]) + uk(1,C.ξ[3], C.σ[3], C.a[3])(y[3]), atol = 1e-10)
end

@testset "RadialMapComponent (Nx,p) = (Nx,p) with Nx>1 and p =[2  -1  2]" begin

    C = SparseRadialMapComponent(3, [2; -1; 2])
    x = randn()
    y = randn(3)

    @test C.Nx == 3
    @test C.p == [2; -1; 2]
    @test C.activedim == [1; 3]

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.a,1)==3

    @test size(C.ξ[1],1)==2
    @test size(C.σ[1],1)==2
    @test size(C.a[1],1)==3

    @test size(C.ξ[2],1)==0
    @test size(C.σ[2],1)==0
    @test size(C.a[2],1)==0

    @test size(C.ξ[3],1)==4
    @test size(C.σ[3],1)==4
    @test size(C.a[3],1)==5

    @test C.ξ[1]==zeros(2)
    @test C.ξ[2]==zeros(0)
    @test C.ξ[3]==zeros(4)

    @test C.σ[1]==ones(2)
    @test C.σ[2]==ones(0)
    @test C.σ[3]==ones(4)

    @test C.a[1]==zeros(3)
    @test C.a[2]==zeros(0)
    @test C.a[3]==zeros(5)

    @test isapprox(C(x), 0.0, atol = 1e-10)
    @test isapprox(C(y), 0.0, atol = 1e-10)

    C.a[1] .= randn(3)
    @test isapprox(C(x), ui(2,C.ξ[1], C.σ[1], C.a[1])(x), atol = 1e-10)
    @test isapprox(C(y), ui(2,C.ξ[1], C.σ[1], C.a[1])(y[1]), atol = 1e-10)

    C.a[3] .= randn(5)
    @test isapprox(C(x), ui(2,C.ξ[1], C.σ[1], C.a[1])(x) + uk(2,C.ξ[3], C.σ[3], C.a[3])(x), atol = 1e-10)
    @test isapprox(C(y), ui(2,C.ξ[1], C.σ[1], C.a[1])(y[1]) + uk(2,C.ξ[3], C.σ[3], C.a[3])(y[3]), atol = 1e-10)

end



@testset "RadialMapComponent (Nx,p) = (Nx,p) with Nx>1 and p =[2  2  -1]" begin

    C = SparseRadialMapComponent(3, [2; 2; -1])
    x = randn()
    y = randn(3)

    @test C.Nx == 3
    @test C.p == [2; 2; -1]
    @test C.activedim == [1; 2]

    @test size(C.ξ,1)==3
    @test size(C.σ,1)==3
    @test size(C.a,1)==3

    @test size(C.ξ[1],1)==2
    @test size(C.σ[1],1)==2
    @test size(C.a[1],1)==3

    @test size(C.ξ[2],1)==2
    @test size(C.σ[2],1)==2
    @test size(C.a[2],1)==3

    @test size(C.ξ[3],1)==0
    @test size(C.σ[3],1)==0
    @test size(C.a[3],1)==0

    @test C.ξ[1]==zeros(2)
    @test C.ξ[2]==zeros(2)
    @test C.ξ[3]==zeros(0)

    @test C.σ[1]==ones(2)
    @test C.σ[2]==ones(2)
    @test C.σ[3]==ones(0)

    @test C.a[1]==zeros(3)
    @test C.a[2]==zeros(3)
    @test C.a[3]==zeros(0)

    @test isapprox(C(x), x, atol = 1e-10)
    @test isapprox(C(y), y[3], atol = 1e-10)

    C.a[1] .= randn(3)
    @test isapprox(C(x), ui(2,C.ξ[1], C.σ[1], C.a[1])(x) + x, atol = 1e-10)
    @test isapprox(C(y), ui(2,C.ξ[1], C.σ[1], C.a[1])(y[1])+ y[3], atol = 1e-10)

    C.a[2] .= randn(3)
    @test isapprox(C(x), ui(2,C.ξ[1], C.σ[1], C.a[1])(x) +
                         ui(2,C.ξ[2], C.σ[2], C.a[2])(x) +
                         uk(-1,C.ξ[3], C.σ[3], C.a[3])(x), atol = 1e-10)

    @test isapprox(C(y), ui(2,C.ξ[1], C.σ[1], C.a[1])(y[1])+ ui(2,C.ξ[2], C.σ[2], C.a[2])(y[2]) +
                         y[3], atol = 1e-10)

end

@testset "set_id function" begin
    C = SparseRadialMapComponent(1, 1)
    set_id(C)
    @test C.Nx==1
    @test C.ξ[1]== Float64[]
    @test C.σ[1]== Float64[]
    @test C.a[1]== Float64[]
    @test C.p == [-1]

    C = SparseRadialMapComponent(4, [1;2;-1;-1])
    set_id(C)
    @test C.Nx ==4
    for i=1:4
        @test C.ξ[i]== Float64[]
        @test C.σ[i]== Float64[]
        @test C.a[i]== Float64[]
        @test C.p[i]== -1
    end
end

@testset "Component function for SparseRadialMapComponent type" begin

    C = SparseRadialMapComponent(1, 2)

    @test C.Nx == 1
    @test C.p == [2]


    @test  AdaptiveTransportMap.component(C, 1).ξk == zeros(2+2)
    @test  AdaptiveTransportMap.component(C, 1).σk == ones(2+2)
    @test  AdaptiveTransportMap.component(C, 1).ak == zeros(2+3)

    C = SparseRadialMapComponent(5, 2)

    @test C.Nx == 5
    @test C.p == [2; 2; 2; 2; 2]


    @test  AdaptiveTransportMap.component(C, 5).ξk == zeros(2+2)
    @test  AdaptiveTransportMap.component(C, 5).σk == ones(2+2)
    @test  AdaptiveTransportMap.component(C, 5).ak == zeros(2+3)

    @test  AdaptiveTransportMap.component(C, 2).ξi == zeros(2)
    @test  AdaptiveTransportMap.component(C, 2).σi == ones(2)
    @test  AdaptiveTransportMap.component(C, 2).ai == zeros(2+1)

    C = SparseRadialMapComponent(5, [1; 0; -1; 2; 1])

    solξ = Int64[1; 0 ; 0; 2; 3]
    solσ = Int64[1; 0 ; 0; 2; 3]
    sola = Int64[2; 1 ; 0; 3; 4]

    for i=1:4
        @test AdaptiveTransportMap.component(C, i).p == C.p[i]
        @test typeof(AdaptiveTransportMap.component(C, i))<:ui
        @test  AdaptiveTransportMap.component(C, i).ξi == zeros(solξ[i])
        @test  AdaptiveTransportMap.component(C, i).σi == ones(solσ[i])
        @test  AdaptiveTransportMap.component(C, i).ai == zeros(sola[i])
    end

    for i=5:5
        @test AdaptiveTransportMap.component(C, i).p == C.p[i]
        @test typeof(AdaptiveTransportMap.component(C, i))<:uk
        @test  AdaptiveTransportMap.component(C, i).ξk == zeros(solξ[i])
        @test  AdaptiveTransportMap.component(C, i).σk == ones(solσ[i])
        @test  AdaptiveTransportMap.component(C, i).ak == zeros(sola[i])
    end

end

@testset "Verify off_diagonal function" begin

    # Nx=1  & p=0
    C = SparseRadialMapComponent(1,0)
    C.a[1] .= randn(2)

    @test off_diagonal(C, randn()) == 0.0

    # Nx=1 & p= 3
    C = SparseRadialMapComponent(1, 3)
    C.a[1] .= rand(6)

    @test off_diagonal(C, randn()) == 0.0


    # Nx=3 & p = 0
    C = SparseRadialMapComponent(3, 0)
    a1 =randn()
    a2 = randn()
    a3 = rand(2)
    C.a[1] .= a1
    C.a[2] .= a2
    C.a[3] .= a3

    z = randn(3)
    @test norm(off_diagonal(C, z) - (C(z) - AdaptiveTransportMap.component(C,3)(z[3])))<1e-10

    # Nx=3 & p = 3
    C = SparseRadialMapComponent(3, 3)
    for i=1:2
    C.ξ[i] .= randn(3)
    C.σ[i] .= rand(3)
    end

    C.ξ[3] .= randn(5)
    C.σ[3] .= rand(5)

    a1 =randn(4)
    a2 = randn(4)
    a3 = rand(6)
    C.a[1] .= a1
    C.a[2] .= a2
    C.a[3] .= a3

    z = randn(3)
    @test norm(off_diagonal(C, z) - (C(z) - AdaptiveTransportMap.component(C,3)(z[3])))<1e-10


    # Nx=1  & p=-1
    C = SparseRadialMapComponent(1,-1)

    @test off_diagonal(C, 2.0)==0.0


    # Nx=3  & p=[-1 -1 -1]
    C = SparseRadialMapComponent(3,-1)

    @test off_diagonal(C, 2.0) == 0.0

    # Nx=3  & p=[0 -1 -1]
    C = SparseRadialMapComponent(3, [0; -1; -1])
    a1 = randn(1)
    C.a[1] .= a1

    z = randn(3)
    @test norm(off_diagonal(C, z) - (C(z) - AdaptiveTransportMap.component(C,3)(z[3])))<1e-10


    # Nx=3  & p=[-1  2 -1]
    C = SparseRadialMapComponent(3, [-1; 2; -1])
    ξ2 = randn(2)
    σ2 = rand(2)
    a2 = randn(3)

    C.ξ[2] .= ξ2
    C.σ[2] .= σ2
    C.a[2] .= a2

    z = randn(3)
    @test norm(off_diagonal(C, z) - (C(z) - AdaptiveTransportMap.component(C,3)(z[3])))<1e-10
end

@testset "extract and modify coefficients of RadialMapComponent" begin
    # Nx=1 and p=0
    C = SparseRadialMapComponent(1, 0)

    @test C.Nx == 1
    @test C.p == [0]

    modify_a([1.0;2.0], C)
    @test extract_a(C) == [1.0; 2.0]

    C = RadialMapComponent(1, 3)
    modify_a([1.0;2.0; 3.0; 4.0; 5.0; 6.0], C)
    @test extract_a(C) == [1.0;2.0; 3.0; 4.0; 5.0; 6.0]


    # Nx =3 and p=2
    A = collect(1.0:1.0:11.0)
    C = SparseRadialMapComponent(3, 2)

    modify_a(A, C)
    @test  extract_a(C) == A

    C = RadialMapComponent(3, 2)
    C.a[1] = [1.0; 2.0; 3.0]
    C.a[2] = [4.0; 5.0; 6.0]
    C.a[3] = [7.0; 8.0; 9.0; 10.0; 11.0]

    @test extract_a(C) ==A

    # Nx = 3 and p = 0

    C = SparseRadialMapComponent(3, 0)

    @test C.Nx == 3
    @test C.p == [0; 0; 0]

    C.a[1] .= [1.0]
    C.a[2] .= [2.0]
    C.a[3] .= [3.0; 4.0]

    @test extract_a(C)==collect(1.0:4.0)

    modify_a(collect(5.0:8.0), C)

    @test extract_a(C)==collect(5.0:8.0)

    # Nx = 5 and p = [-1 2 -1 0 -1]
    C = SparseRadialMapComponent(5, [-1; 2; -1; 0; -1])
    A = randn(4)
    modify_a(A, C)
    @test extract_a(C)==A

    # Nx = 11 and p = [2 -1 0 -1 0 2 -1 0 -1 2  1]
    C = SparseRadialMapComponent(11, [2; -1; 0; -1; 0; 2; -1; 0; -1; 2; 2])
    A = randn(17)
    modify_a(A, C)
    @test C.a[1]  == A[1:3]
    @test C.a[2]  == Float64[]
    @test C.a[3]  == [A[4]]
    @test C.a[4]  == Float64[]
    @test C.a[5]  == [A[5]]
    @test C.a[6]  == A[6:8]
    @test C.a[7]  == Float64[]
    @test C.a[8]  == [A[9]]
    @test C.a[9]  == Float64[]
    @test C.a[10] == A[10:12]
    @test C.a[11] == A[13:17]

    @test extract_a(C)==A

    # There is an extra verification that we have made exactly as much assignment
    # that there are component of C.a
end
