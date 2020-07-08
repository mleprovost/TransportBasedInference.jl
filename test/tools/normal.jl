
@testset "Verify hesslogpdf for a Normal distribution" begin

N = Normal(1.5, 2.3)

Nx = 10
x = randn(Nx)

    for i =1:Nx
        @test abs(hesslogpdf(N, x[i]) - (-1/N.σ^2))<1e-10
    end
end
