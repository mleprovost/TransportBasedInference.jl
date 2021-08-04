
@testset "Verify vander for ProHermiteBasis: evaluation, first and second derivatives" begin
    Ne = 200
    x = randn(Ne)
    #Test all the basis
    B = CstProHermiteBasis(6)
    ktab = [0; 1; 2]
    for k in ktab
        # Test evaluation
        dV = vander(B, k, x)
        dVtrue = zeros(Ne, size(B))

        for i=1:B.m
            dVtrue[:,i] .= derivative(FamilyScaledProHermite[i], k , x)
        end
        @test norm(dV - dVtrue)<1e-6
    end
end

@testset "Verify vander for PhyHermiteBasis: evaluation, first and second derivatives" begin
    Ne = 200
    x = randn(Ne)
    #Test all the basis
    B = CstProHermiteBasis(6)
    ktab = [0; 1; 2]
    for k in ktab
        # Test evaluation
        dV = vander(B, k, x)
        dVtrue = zeros(Ne, size(B))

        for i=1:B.m
            dVtrue[:,i] .= derivative(FamilyScaledPhyHermite[i], k , x)
        end
        @test norm(dV - dVtrue)<1e-6
    end
end

@testset "Verify vander for CstProHermiteBasis: evaluation, first and second derivatives" begin
    Ne = 200
    x = randn(Ne)
    #Test all the basis
    B = CstProHermiteBasis(6)
    ktab = [0; 1; 2]
    for k in ktab
        # Test evaluation
        dV = vander(B, k, x)
        dVtrue = zeros(Ne, size(B))

        for i=1:B.m
            if i==1
                Pik = derivative(FamilyProPolyHermite[1], k)
                dVtrue[:,i] .= Pik.(x)
            else
                dVtrue[:,i] .= derivative(FamilyScaledProHermite[i-1], k , x)
            end
        end
        @test norm(dV - dVtrue)<1e-6
    end
end

@testset "Verify vander for CstPhyHermiteBasis: evaluation, first and second derivatives" begin
    Ne = 200
    x = randn(Ne)
    #Test all the basis
    B = CstPhyHermiteBasis(6)
    ktab = [0; 1; 2]

    for k in ktab
        # Test evaluation
        dV = vander(B, k, x)
        dVtrue = zeros(Ne, size(B))

        for i=1:B.m
            if i==1
                Pik = derivative(FamilyProPolyHermite[1], k)
                dVtrue[:,i] .= Pik.(x)
            else
                dVtrue[:,i] .= derivative(FamilyScaledPhyHermite[i-1], k , x)
            end
        end
        @test norm(dV - dVtrue)<1e-6
    end
end

@testset "Verify vander for CstLinProHermiteBasis: evaluation, first and second derivatives" begin
    Ne = 200
    x = randn(Ne)
    #Test all the basis
    B = CstLinProHermiteBasis(6)
    ktab = [0; 1; 2]

    for k in ktab
        # Test evaluation
        dV = vander(B, k, x)
        dVtrue = zeros(Ne, size(B))

        for i=1:B.m
            if i==1
                Pik = derivative(FamilyProPolyHermite[1], k)
                dVtrue[:,i] .= Pik.(x)
            elseif i==2
                Pik = derivative(FamilyProPolyHermite[2], k)
                dVtrue[:,i] .= Pik.(x)
            else
                dVtrue[:,i] .= derivative(FamilyScaledProHermite[i-2], k , x)
            end
        end
        @test norm(dV - dVtrue)<1e-6
    end
end

@testset "Verify vander for CstLinPhyHermiteBasis: evaluation, first and second derivatives" begin
    Ne = 200
    x = randn(Ne)
    #Test all the basis
    B = CstLinPhyHermiteBasis(6)
    ktab = [0; 1; 2]

    for k in ktab
        # Test evaluation
        dV = vander(B, k, x)
        dVtrue = zeros(Ne, size(B))

        for i=1:B.m
            if i==1
                Pik = derivative(FamilyProPolyHermite[1], k)
                dVtrue[:,i] .= Pik.(x)
            elseif i==2
                Pik = derivative(FamilyProPolyHermite[2], k)
                dVtrue[:,i] .= Pik.(x)
            else
                dVtrue[:,i] .= derivative(FamilyScaledPhyHermite[i-2], k , x)
            end
        end
        @test norm(dV - dVtrue)<1e-6
    end
end
#
#
# @testset "Verify vander for basis: evaluation, first and second derivatives" begin
#     Ne = 200
#     x = randn(Ne)
#     #Test all the basis
#     B = Basis(5)#CstProHermiteBasis(5; scaled = true)
#     # Btab = [CstProHermiteBasis(5; scaled = false);
#     #         CstProHermiteBasis(5; scaled = true)];
#             # CstPhyHermiteBasis(5; scaled = false);
#             # CstPhyHermiteBasis(5; scaled = true);
#             # CstLinPhyHermiteBasis(5; scaled = false);
#             # CstLinPhyHermiteBasis(5; scaled = true);
#             # CstLinProHermiteBasis(5; scaled = false);
#             # CstLinProHermiteBasis(5; scaled = true)]
#             # Test evaluation, derivative and second derivative
#     ktab = [0; 1; 2]
#
#     for k in ktab
#         # Test evaluation
#         dV = vander(B, k, x)
#         dVtrue = zeros(Ne, size(B))
#
#         for i=1:B.m
#             if i==1
#                 Pik = derivative(FamilyProPolyHermite[1], k)
#                 dVtrue[:,i] .= Pik.(x)
#             else
#                 dVtrue[:,i] .= derivative(FamilyScaledProHermite[i-1], k , x)
#             end
#         end
#         @test norm(dV - dVtrue)<1e-6
#     end
# end

# @testset "Test defined families list" begin
#
#     B = CstPhyHermiteBasis(10; scaled =false)
#     @test size(B,1)==12
#
#     @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
#
#     for i=2:size(B,1)
#         @test B[i] == FamilyPhyHermite[i-1]
#     end
#
#
#     B = CstPhyHermiteBasis(10; scaled =true)
#     @test size(B,1)==12
#
#     @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
#
#     for i=2:size(B,1)
#         @test B[i] == FamilyScaledPhyHermite[i-1]
#     end
#
#
#     B = CstProHermiteBasis(10; scaled =false)
#     @test size(B,1)==12
#
#     @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
#
#     for i=2:size(B,1)
#         @test B[i] == FamilyProHermite[i-1]
#     end
#
#
#     B = CstProHermiteBasis(10; scaled =true)
#     @test size(B,1)==12
#
#     @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
#
#     for i=2:size(B,1)
#         @test B[i] == FamilyScaledProHermite[i-1]
#     end
#
#     B = CstLinPhyHermiteBasis(10; scaled =false)
#     @test size(B,1)==13
#
#     @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
#     @test B[2] ==B.f[2] == FamilyProPolyHermite[2]
#
#
#     for i=3:size(B,1)
#         @test B[i] == FamilyPhyHermite[i-2]
#     end
#
#     B = CstLinPhyHermiteBasis(10; scaled = true)
#     @test size(B,1)==13
#
#     @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
#     @test B[2] ==B.f[2] == FamilyProPolyHermite[2]
#
#
#     for i=3:size(B,1)
#         @test B[i] == FamilyScaledPhyHermite[i-2]
#     end
#
#     B = CstLinProHermiteBasis(10; scaled =false)
#     @test size(B,1)==13
#
#     @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
#     @test B[2] ==B.f[2] == FamilyProPolyHermite[2]
#
#
#     for i=3:size(B,1)
#         @test B[i] == FamilyProHermite[i-2]
#     end
#
#     B = CstLinProHermiteBasis(10; scaled = true)
#     @test size(B,1)==13
#
#     @test B[1] ==B.f[1] == FamilyProPolyHermite[1]
#     @test B[2] ==B.f[2] == FamilyProPolyHermite[2]
#
#
#     for i=3:size(B,1)
#         @test B[i] == FamilyScaledProHermite[i-2]
#     end
# end


# @testset "Test evaluation of the basis" begin
#     # @testset "Test evaluation of the basis" begin
#         Ne = 10
#         x = 0.5*randn(Ne)
#
#         B = CstPhyHermiteBasis(5; scaled =false)
#         m = size(B,1)
#
#         Bx = B(x[1])
#         @test Bx[1] == 1.0
#         for i=0:5
#             @test abs(Bx[i+2] -  PhyHermite(i)(x[1]))< 1e-8
#         end
#
#
#         B = CstPhyHermiteBasis(5; scaled = true)
#         m = size(B,1)
#
#         Bx = B(x[1])
#         @test Bx[1] == 1.0
#         for i=0:5
#             @test abs(Bx[i+2] -  PhyHermite(i; scaled = true)(x[1]))< 1e-8
#         end
#
#         B = CstProHermiteBasis(5; scaled =false)
#         m = size(B,1)
#
#         Bx = B(x[1])
#         @test Bx[1] == 1.0
#         for i=0:5
#             @test abs(Bx[i+2] -  ProHermite(i)(x[1]))< 1e-8
#         end
#
#
#         B = CstProHermiteBasis(5; scaled = true)
#         m = size(B,1)
#
#         Bx = B(x[1])
#         @test Bx[1] == 1.0
#         for i=0:5
#             @test abs(Bx[i+2] -  ProHermite(i; scaled = true)(x[1]))< 1e-8
#         end
#
#         B = CstLinPhyHermiteBasis(5; scaled =false)
#         m = size(B,1)
#
#         Bx = B(x[1])
#         @test Bx[1] == 1.0
#         @test Bx[2] == x[1]
#
#         for i=0:5
#             @test abs(Bx[i+3] -  PhyHermite(i)(x[1]))< 1e-8
#         end
#
#         B = CstLinPhyHermiteBasis(5; scaled =true)
#         m = size(B,1)
#
#         Bx = B(x[1])
#         @test Bx[1] == 1.0
#         @test Bx[2] == x[1]
#
#         for i=0:5
#             @test abs(Bx[i+3] -  PhyHermite(i; scaled = true)(x[1]))< 1e-8
#         end
#
#         B = CstLinProHermiteBasis(5; scaled =false)
#         m = size(B,1)
#
#         Bx = B(x[1])
#         @test Bx[1] == 1.0
#         @test Bx[2] == x[1]
#
#         for i=0:5
#             @test abs(Bx[i+3] -  ProHermite(i)(x[1]))< 1e-8
#         end
#
#         B = CstLinProHermiteBasis(5; scaled =true)
#         m = size(B,1)
#
#         Bx = B(x[1])
#         @test Bx[1] == 1.0
#         @test Bx[2] == x[1]
#
#         for i=0:5
#             @test abs(Bx[i+3] -  ProHermite(i; scaled = true)(x[1]))< 1e-8
#         end
# end
