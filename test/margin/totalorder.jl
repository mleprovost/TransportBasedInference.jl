

@testset "Test total multi order indices" begin

# Nx = 1
midxs = totalorder([0])
@test midxs == reshape([0],(1,1))
midxs = totalorder([1])
@test midxs == reshape([0; 1],(2,1))
midxs = totalorder([3])
@test midxs == reshape([0; 1; 2; 3],(4,1))

#Nx = 2

midxs = totalorder([0; 0])
@test midxs == reshape([0; 0],(1,2))
midxs = totalorder([1; 0])
@test midxs == [0 0; 1 0]
midxs = totalorder([3;4])
@test midxs ==
     [0     0
     0     1
     0     2
     0     3
     0     4
     1     0
     1     1
     1     2
     1     3
     2     0
     2     1
     2     2
     3     0
     3     1]

# Nx = 3

midxs = totalorder([0; 0; 0])
@test midxs == reshape([0; 0; 0],(1,3))
midxs = totalorder([0; 1; 0])
@test midxs == [0 0 0; 0 1 0]

midxs = totalorder([2;4;3])

@test midxs ==
[0     0     0
0     0     1
0     0     2
0     0     3
0     1     0
0     1     1
0     1     2
0     1     3
0     2     0
0     2     1
0     2     2
0     3     0
0     3     1
0     4     0
1     0     0
1     0     1
1     0     2
1     0     3
1     1     0
1     1     1
1     1     2
1     2     0
1     2     1
1     3     0
2     0     0
2     0     1
2     0     2
2     1     0
2     1     1
2     2     0]
end
