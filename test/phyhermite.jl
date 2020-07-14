

@testset "Verify evaluation of Physicist hermite functions" begin

    # @testset "Verify evaluation of Physicist hermite functions" begin
    x = randn(100)
    m = 5
    dV = vander(PhyHermite(m; scaled = false), 0, x)

    for i=0:5
        @test norm(dV[:,i+1] - FamilyPhyPolyHermite[i+1].(x) .* exp.(-x.^2/2))<1e-8
    end

    dV = vander(PhyHermite(m; scaled = true), 0, x)


    for i=0:5
        @test dV[:,i+1] == FamilyScaledPhyPolyHermite[i+1].(x).*exp.(-x.^2/2)
    end
end

@testset "Verify 1st derivative of Physicist hermite functions" begin
    # Unscaled
    x = randn(100)
    m = 5
    dV = vander(PhyHermite(m; scaled = false), 1, x)

    for i=0:5
        F = PhyHermite(i; scaled = false)
        @test norm(dV[:,i+1] - map(xi->ForwardDiff.derivative(F, xi), x) )<1e-8
        @test norm(dV[:,i+1] - (FamilyPhyPolyHermite[i+1].(x) .* (-x) .* exp.(-x.^2/2) +
            derivative(FamilyPhyPolyHermite[i+1],1).(x) .* exp.(-x.^2/2)))<1e-8
    end

    # Scaled
    x = randn(100)
    m = 5
    dV = vander(PhyHermite(m; scaled = true), 1, x)

    for i=0:5
        F = PhyHermite(i; scaled = true)
        @test norm(dV[:,i+1] - map(xi->ForwardDiff.derivative(F, xi), x) )<1e-8

        @test norm(dV[:,i+1] - 1/Cphy(i)*(FamilyPhyPolyHermite[i+1].(x) .* (-x) .* exp.(-x.^2/2) +
            derivative(FamilyPhyPolyHermite[i+1],1).(x) .* exp.(-x.^2/2)))<1e-8
    end


end


@testset "Verify second derivative of Physicist hermite functions" begin
    # Unscaled
    x = randn(100)
    m = 5
    dV = vander(PhyHermite(m; scaled = false), 2, x)

    for i=0:5
        Hn = FamilyPhyPolyHermite[i+1].(x)
        Hnp = derivative(FamilyPhyPolyHermite[i+1],1).(x)
        Hnpp = derivative(FamilyPhyPolyHermite[i+1],2).(x)
        E = exp.(-x.^2/2)
        F = PhyHermite(i; scaled = false)
        @test norm(dV[:,i+1] - map(xi->ForwardDiff.derivative(y->ForwardDiff.derivative(z->F(z), y), xi), x))<1e-8

        @test norm(dV[:,i+1] - (Hnpp .*E -2 .*x .* Hnp.*E + Hn .* (x.^2 .- 1.0) .* E))<1e-8
    end

    # Scaled

    x = randn(100)
    m = 5
    dV = vander(PhyHermite(m; scaled = true), 2, x)

    for i=0:5
        Hn = FamilyPhyPolyHermite[i+1].(x)
        Hnp = derivative(FamilyPhyPolyHermite[i+1],1).(x)
        Hnpp = derivative(FamilyPhyPolyHermite[i+1],2).(x)
        E = exp.(-x.^2/2)

        F = PhyHermite(i; scaled = true)
        @test norm(dV[:,i+1] - map(xi->ForwardDiff.derivative(y->ForwardDiff.derivative(z->F(z), y), xi), x))<1e-8


        @test norm(dV[:,i+1] - (1/Cphy(i))*(Hnpp .*E -2 .*x .* Hnp.*E + Hn .* (x.^2 .- 1.0) .* E))<1e-8
    end
end



@testset "Verify integration of Physicist hermite functions" begin
    Nx = 10
    x = randn(10)

    # Unscaled
    dV = vander(PhyHermite(8; scaled = false), -1, x)

    for i=0:8
        for j=1:Nx
            @test abs(dV[j,i+1] - quadgk(y-> PhyHermite(i)(y), 0.0, x[j], rtol=1e-12)[1])<1e-8
        end
    end

    # Scaled
    dV = vander(PhyHermite(8; scaled = true), -1, x)

    for i=0:8
        for j=1:Nx
            @test abs(dV[j,i+1] - quadgk(y-> PhyHermite(i; scaled=true)(y), 0.0, x[j], rtol=1e-12)[1])<1e-8
        end
    end

end
