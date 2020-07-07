

@testset "Verify evaluation of Physicist hermite functions" begin

    # @testset "Verify evaluation of Physicist hermite functions" begin
    x = randn(100)
    m = 5
    dV = vander(PhyHermite(m), 0, x; scaled = false)

    for i=0:5
        @test norm(dV[:,i+1] - FamilyPhyPolyHermite[i+1].(x) .* exp.(-x.^2/2))<1e-8
    end

    dV = vander(PhyHermite(m), 0, x; scaled = true)


    for i=0:5
        @test dV[:,i+1] == FamilyScaledPhyPolyHermite[i+1].(x).*exp.(-x.^2/2)
    end
end

@testset "Verify 1st derivative of Physicist hermite functions" begin

    x = randn(100)
    m = 5
    dV = vander(PhyHermite(m), 1, x; scaled = false)

    for i=0:5
        @test norm(dV[:,i+1] - (FamilyPhyPolyHermite[i+1].(x) .* (-x) .* exp.(-x.^2/2) +
            derivative(FamilyPhyPolyHermite[i+1],1).(x) .* exp.(-x.^2/2)))<1e-8
    end
end


@testset "Verify second derivative of Physicist hermite functions" begin

    x = randn(100)
    m = 5
    dV = vander(PhyHermite(m), 2, x; scaled = false)

    for i=0:5
        Hn = FamilyPhyPolyHermite[i+1].(x)
        Hnp = derivative(FamilyPhyPolyHermite[i+1],1).(x)
        Hnpp = derivative(FamilyPhyPolyHermite[i+1],2).(x)
        E = exp.(-x.^2/2)

        @test norm(dV[:,i+1] - (Hnpp .*E -2 .*x .* Hnp.*E + Hn .* (x.^2 .- 1.0) .* E))<1e-8
    end
end


# Need to write test for integration
