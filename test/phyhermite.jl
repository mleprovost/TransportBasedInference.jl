

@testset "Verify evaluation of Physicist hermite functions" begin


x = randn(10)
m = 5
dV = vander(PhyHermite(m), 0, x; scaled = false)

for i=0:5
    @test dV[:,i+1] == PhyHermite[i+1].(x).*exp.(-x.^2/2)
end

dV = vander(PhyHermite(m), 0, x; scaled = true)


for i=0:5
    @test dV[:,i+1] == PhyHermite[i+1].(x).*exp.(-x.^2/2)
end


end

@testset "Verify derivative of Physicist hermite functions" begin





end


@testset "Verify second derivative of Physicist hermite functions" begin





end
