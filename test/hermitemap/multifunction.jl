
@testset "Test MultiFunction in multiple dimensions" begin
    Nxlist= collect(1:5)
    Blist = [CstProHermite(10); CstPhyHermite(10); CstLinProHermite(10); CstLinPhyHermite(10)]
    Nψ = 10
    Ne = 50

    for Nx in Nxlist
        X = randn(Nx, Ne)
        idx = rand(0:10,Nψ, Nx)
        for b in Blist
            for i=1:Nψ
            B = MultiBasis(b, Nx)
            F = MultiFunction(B, idx[i,:])
                for j=1:Ne
                    @test norm(F(X[:,j]) - prod(k-> B.B[idx[i,k]+1](X[k,j]), 1:Nx))<1e-6
                end
            end
        end
    end
end
