
@testset "Test defined families list" begin

    B = CstPhyHermite(10; scaled =false)
    @test size(B,1)==12

    @test B[1] ==B.f[1] == FamilyProPolyHermite[1]

    for i=2:size(B,1)
        @test B[i] == FamilyPhyHermite[i-1]
    end


    B = CstPhyHermite(10; scaled =true)
    @test size(B,1)==12

    @test B[1] ==B.f[1] == FamilyProPolyHermite[1]

    for i=2:size(B,1)
        @test B[i] == FamilyScaledPhyHermite[i-1]
    end


    B = CstProHermite(10; scaled =false)
    @test size(B,1)==12

    @test B[1] ==B.f[1] == FamilyProPolyHermite[1]

    for i=2:size(B,1)
        @test B[i] == FamilyProHermite[i-1]
    end


    B = CstProHermite(10; scaled =true)
    @test size(B,1)==12

    @test B[1] ==B.f[1] == FamilyProPolyHermite[1]

    for i=2:size(B,1)
        @test B[i] == FamilyScaledProHermite[i-1]
    end

    B = CstLinPhyHermite(10; scaled =false)
    @test size(B,1)==13

    @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
    @test B[2] ==B.f[2] == FamilyProPolyHermite[2]


    for i=3:size(B,1)
        @test B[i] == FamilyPhyHermite[i-2]
    end

    B = CstLinPhyHermite(10; scaled = true)
    @test size(B,1)==13

    @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
    @test B[2] ==B.f[2] == FamilyProPolyHermite[2]


    for i=3:size(B,1)
        @test B[i] == FamilyScaledPhyHermite[i-2]
    end

    B = CstLinProHermite(10; scaled =false)
    @test size(B,1)==13

    @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
    @test B[2] ==B.f[2] == FamilyProPolyHermite[2]


    for i=3:size(B,1)
        @test B[i] == FamilyProHermite[i-2]
    end

    B = CstLinProHermite(10; scaled = true)
    @test size(B,1)==13

    @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
    @test B[2] ==B.f[2] == FamilyProPolyHermite[2]


    for i=3:size(B,1)
        @test B[i] == FamilyScaledProHermite[i-2]
    end
end


@testset "Test evaluation of the basis" begin
    # @testset "Test evaluation of the basis" begin
        Ne = 10
        x = 0.5*randn(Ne)

        B = CstPhyHermite(5; scaled =false)
        m = size(B,1)

        Bx = B(x[1])
        @test Bx[1] == 1.0
        for i=0:5
            @test abs(Bx[i+2] -  PhyHermite(i)(x[1]))< 1e-8
        end


        B = CstPhyHermite(5; scaled = true)
        m = size(B,1)

        Bx = B(x[1])
        @test Bx[1] == 1.0
        for i=0:5
            @test abs(Bx[i+2] -  PhyHermite(i; scaled = true)(x[1]))< 1e-8
        end

        B = CstProHermite(5; scaled =false)
        m = size(B,1)

        Bx = B(x[1])
        @test Bx[1] == 1.0
        for i=0:5
            @test abs(Bx[i+2] -  ProHermite(i)(x[1]))< 1e-8
        end


        B = CstProHermite(5; scaled = true)
        m = size(B,1)

        Bx = B(x[1])
        @test Bx[1] == 1.0
        for i=0:5
            @test abs(Bx[i+2] -  ProHermite(i; scaled = true)(x[1]))< 1e-8
        end

        B = CstLinPhyHermite(5; scaled =false)
        m = size(B,1)

        Bx = B(x[1])
        @test Bx[1] == 1.0
        @test Bx[2] == x[1]

        for i=0:5
            @test abs(Bx[i+3] -  PhyHermite(i)(x[1]))< 1e-8
        end

        B = CstLinPhyHermite(5; scaled =true)
        m = size(B,1)

        Bx = B(x[1])
        @test Bx[1] == 1.0
        @test Bx[2] == x[1]

        for i=0:5
            @test abs(Bx[i+3] -  PhyHermite(i; scaled = true)(x[1]))< 1e-8
        end

        B = CstLinProHermite(5; scaled =false)
        m = size(B,1)

        Bx = B(x[1])
        @test Bx[1] == 1.0
        @test Bx[2] == x[1]

        for i=0:5
            @test abs(Bx[i+3] -  ProHermite(i)(x[1]))< 1e-8
        end

        B = CstLinProHermite(5; scaled =true)
        m = size(B,1)

        Bx = B(x[1])
        @test Bx[1] == 1.0
        @test Bx[2] == x[1]

        for i=0:5
            @test abs(Bx[i+3] -  ProHermite(i; scaled = true)(x[1]))< 1e-8
        end
end


@testset "Verify vander for basis: evaluation first and second derivatives" begin
    Ne = 20
    x = randn(Ne)
    #Test all the basis
    Btab = [CstPhyHermite(5; scaled = false);
            CstPhyHermite(5; scaled = true);
            CstProHermite(5; scaled = false);
            CstPhyHermite(5; scaled = true);
            CstLinPhyHermite(5; scaled = false);
            CstLinPhyHermite(5; scaled = true);
            CstLinProHermite(5; scaled = false);
            CstLinProHermite(5; scaled = true)]
            # Test evaluation, derivative and second derivative
    ktab = [0; 1; 2]

    for B in Btab
        for k in ktab
            # Test evaluation
            dV = vander(B, k, x)
            dVtrue = zeros(Ne, size(B,1))

            for (i, fi) in enumerate(B.f)
                if typeof(fi) <: Union{PhyPolyHermite, ProPolyHermite}
                    Pik = derivative(fi, k)
                    dVtrue[:,i] .= Pik.(x)
                elseif typeof(fi) <: Union{PhyHermite, ProHermite}
                    dVtrue[:,i] .= derivative(fi, k , x)
                end
            end
            @test norm(dV - dVtrue)<1e-6
        end
    end
end
