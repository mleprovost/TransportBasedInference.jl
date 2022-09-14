

@testset "product of integer" begin

    @test fact2(0)==1
    @test fact2(1)==1

    @test fact2(2)==2


    @test fact2(8)==2*4*6*8
    @test fact2(11)==3*5*7*9*11
end
