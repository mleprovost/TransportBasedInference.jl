

@testset "Test getreducedmargin d = 1" begin
    # d = 1

    idx = reshape([0],(1,1))

    reducedmargin = getreducedmargin(idx)

    @test reducedmargin == reshape([1],(1,1))

    idx = reshape([0 ,1 , 2 ,3],(4,1))

    reducedmargin = getreducedmargin(idx)

    @test reducedmargin == reshape([4],(1,1))

    # Reorder variable
    for i=1:1000
        p = randperm(size(idx,1))

        idxshuffle = idx[p,:]
        reducedmargin = getreducedmargin(idx)
        @test reducedmargin == reshape([4],(1,1))
    end
end


@testset "Test getreducedmargin d = 2" begin
    # d = 2

    idx = [0 0]
    reducedmargin = getreducedmargin(idx)
    @test reducedmargin == [0 1; 1 0]

    idx = [0 0; 0 1; 0 2; 0 3; 1 0; 1 1; 2 0; 3 0]
    reducedmargin = getreducedmargin(idx)
    @test reducedmargin == [0 4; 1 2; 2 1; 4 0]

    # Reorder variable
    for i=1:1000
        p = randperm(size(idx,1))

        idxshuffle = idx[p,:]
        reducedmargin = getreducedmargin(idx)
        @test reducedmargin == [0 4; 1 2; 2 1; 4 0]
    end
end


@testset "Test getreducedmargin d = 3" begin
    # d = 3

    idx = [0 0 0]

    reducedmargin = getreducedmargin(idx)

    @test reducedmargin == [0 0 1; 0 1 0; 1 0 0]



    idx = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 1 0; 1 0 1]

    reducedmargin = getreducedmargin(idx)

    @test reducedmargin ==
     [0     0     2;
      0     2     0;
      1     0     0;
      1     1     1]


    # Reorder variable
    for i=1:1000
        p = randperm(size(idx,1))

        idxshuffle = idx[p,:]
        reducedmargin = getreducedmargin(idx)
        @test reducedmargin ==
         [0     0     2;
          0     2     0;
          1     0     0;
          1     1     1]
    end


    # Difficult test
    idx = [0 0 0; 0 0 1; 0 1 0; 1 0 0; 0 0 2; 0 0 3; 1 1 0; 1 2 0; 1 1 0]

    reducedmargin = getreducedmargin(idx)

    @test reducedmargin == [ 0 0 4; 0 1 1; 0 2 0; 1 0 1; 2 0 0]

end


@testset "Test getreducedmargin d = 4" begin
    # d = 4

    idx = [0 0 0 0]

    reducedmargin = getreducedmargin(idx)

    @test reducedmargin == [0 0 0 1; 0 0 1 0;0 1 0 0; 1 0 0 0]
end


@testset "Test update reduced margin d = 2" begin

    lowerset = [0 0; 0 1; 0 2; 0 3; 1 0; 1 1; 2 0; 3 0]
    reduced_margin = getreducedmargin(lowerset)
    reduced_margin0 = deepcopy(reduced_margin)
    lowerset0 = deepcopy(lowerset)

    for idx = 1:size(reduced_margin,1)
        # @show idx
        new_lowerset, new_reduced_margin = updatereducedmargin(lowerset, reduced_margin, idx)
        # @show reduced_margin[idx,:]
        # @show new_reduced_margin
        # @show new_lowerset
        @test lowerset == lowerset0
        @test reduced_margin == reduced_margin0

        @test new_lowerset[1:end-1,:] == lowerset0

        @test new_lowerset[end,:] == reduced_margin[idx,:]

        @test all([any(x in eachslice(getreducedmargin(new_lowerset);dims = 1)) for x in eachslice(new_reduced_margin; dims = 1)])
    end
end


@testset "Test update reduced margin d = 3" begin

    lowerset = [0 0 0; 0 0 1; 0 1 0; 0 2 0; 1 0 0; 0 0 2; 0 0 3; 1 1 0; 1 2 0; 1 1 0]
    reduced_margin = getreducedmargin(lowerset)
    reduced_margin0 = deepcopy(reduced_margin)
    lowerset0 = deepcopy(lowerset)

    for idx = 1:size(reduced_margin,1)
        # @show idx
        new_lowerset, new_reduced_margin = updatereducedmargin(lowerset, reduced_margin, idx)
        # @show reduced_margin[idx,:]
        # @show new_reduced_margin
        # @show new_lowerset
        @test lowerset == lowerset0
        @test reduced_margin == reduced_margin0

        @test new_lowerset[1:end-1,:] == lowerset0

        @test new_lowerset[end,:] == reduced_margin[idx,:]

        @test all([any(x in eachslice(getreducedmargin(new_lowerset);dims = 1)) for x in eachslice(new_reduced_margin; dims = 1)])
    end
end
