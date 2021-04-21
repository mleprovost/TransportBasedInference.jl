using Test

using LinearAlgebra, Statistics
using SpecialFunctions, ForwardDiff
using TransportMap


@testset "weights ui" begin

    um1 = ui(-1)
    woff = Float64[]
    weights(um1, 2.0, woff)
    @test woff ==Float64[]

    u4 = ui(0, zeros(0), ones(0), [0.0])

    woff = zeros(1)

    weights(u4, 2.0, woff)
    @test woff[1]==2.0


    u4 = ui(2, zeros(2), ones(2), [0.0; 0.0; 0.0])

    woff = zeros(3)

    weights(u4, 5.0, woff)
    @test woff[1] == 5.0
    @test woff[2] == rbf(0.0, 1.0)(5.0)
    @test woff[3] == rbf(0.0, 1.0)(5.0)


    u4.ξi[1] = -5.0
    u4.σi[1] = 5.0

    u4.ξi[2] = 2.0
    u4.σi[2] = 2.0


    weights(u4, -2.0, woff)
    @test woff[1] == -2.0
    @test woff[2] == rbf(-5.0, 5.0)(-2.0)
    @test woff[3] == rbf(2.0, 2.0)(-2.0)
end


@testset "weights uk" begin

    um1 = uk(-1)
    wdiag = Float64[]
    w∂k = Float64[]
    weights(um1, 2.0, wdiag, w∂k)
    @test wdiag ==Float64[]
    @test w∂k ==Float64[]

    u1 = uk(0)
    wdiag = zeros(2)
    w∂k = zeros(1)
    weights(u1, 2.0, wdiag, w∂k)
    @test wdiag == [1.0; 2.0]
    @test w∂k[1] === 1.0

    u4 = uk(0)
    wdiag = zeros(2)
    w∂k = zeros(1)

    weights(u4, 2.0, wdiag, w∂k)

    @test wdiag == [1.0; 2.0]
    @test w∂k[1] === 1.0

    u4 = uk(3)
    wdiag = zeros(3+3)
    w∂k = zeros(3+2)

    weights(u4, 2.0, wdiag, w∂k)
    @test size(wdiag,1)== 3+3
    @test size(w∂k,1)== 3+2
    @test norm(wdiag - [1.0; ψ₀(0.0, 1.0)(2.0); ψj(0.0, 1.0)(2.0); ψj(0.0, 1.0)(2.0); ψj(0.0, 1.0)(2.0); ψpp1(0.0, 1.0)(2.0)])<1e-10
    @test norm(w∂k -[ψ₀′(0.0, 1.0)(2.0); rbf(0.0, 1.0)(2.0); rbf(0.0, 1.0)(2.0); rbf(0.0, 1.0)(2.0); ψpp1′(0.0, 1.0)(2.0)])<1e-10


    u4 = uk(3)
    wdiag = zeros(3+3)
    w∂k = zeros(3+2)

    u4.ξk .= [-1.0; 2.0; -3.0; 4.0; -5.0]
    u4.σk .= [1.0; 2.0; 3.0; 4.0; 5.0]

    weights(u4, 3.0, wdiag, w∂k)
    @test size(wdiag,1)== 3+3
    @test size(w∂k,1)== 3+2

    @test norm(wdiag - [1.0; ψ₀(-1.0, 1.0)(3.0); ψj(2.0, 2.0)(3.0); ψj(-3.0, 3.0)(3.0); ψj(4.0, 4.0)(3.0); ψpp1(-5.0, 5.0)(3.0)])<1e-10
end




@testset "Weights of a Uk map" begin
Vk = KRmap(1, 0)

woff, wdiag, w∂k = create_weights(Vk)
# wdiag = zeros(2)
# w∂k = zeros(1)
# woff = zeros(0)

# w, w∂k = weights(TransportMap.component(Vk.U[1],1), [2.0], w, w∂k)
weights(TransportMap.component(Vk.U[1],1), 2.0, wdiag, w∂k)

@test size(wdiag,1) == 2
@test size(w∂k, 1) == 1
@test size(woff, 1) == 0

@test wdiag == [1.0; 2.0]
@test w∂k   == [1.0]

#
Vk = KRmap(1, 1)

woff, wdiag, w∂k = create_weights(Vk)

weights(Vk, [2.0], woff,  wdiag, w∂k)
#
@test size(woff,1)==0
@test size(wdiag,1) == 1+3
@test size(w∂k, 1) == 1+2
#
@test woff == Float64[]
@test wdiag == [1.0; ψ₀(0.0, 1.0)(2.0); ψj(0.0, 1.0)(2.0); ψpp1(0.0, 1.0)(2.0)]
@test w∂k   == [ψ₀′(0.0, 1.0)(2.0); rbf(0.0, 1.0)(2.0); ψpp1′(0.0, 1.0)(2.0)]
#
#
Vk = KRmap(2, 0)

woff, wdiag, w∂k = create_weights(Vk)

#
weights(Vk, [-2.0; 2.0], woff, wdiag, w∂k)
#
@test size(woff, 1) == 1
@test size(wdiag,1) == 2+2
@test size(w∂k, 1) == 1+1
#
#
@test wdiag == [1.0; -2.0; 1.0; 2.0]
@test w∂k  == [1.0;1.0]
@test woff == [-2.0]
#
#
Vk = KRmap(3, 2)

woff, wdiag, w∂k = create_weights(Vk)

weights(Vk, [-2.0; 2.0; 5.0], woff, wdiag, w∂k)
#
@test size(woff,1) == 2*(2+1)
@test size(wdiag,1) == 3*(2+3)
@test size(w∂k, 1) == 3*(2+2)
#
#
@test woff == [-2.0; rbf(0.0, 1.0)(-2.0); rbf(0.0, 1.0)(-2.0); 2.0; rbf(0.0, 1.0)(2.0); rbf(0.0, 1.0)(2.0)]

@test wdiag == [1.0; ψ₀(0.0, 1.0)(-2.0); ψj(0.0, 1.0)(-2.0); ψj(0.0, 1.0)(-2.0); ψpp1(0.0, 1.0)(-2.0);1.0;
ψ₀(0.0, 1.0)(2.0); ψj(0.0, 1.0)(2.0); ψj(0.0, 1.0)(2.0);ψpp1(0.0, 1.0)(2.0);1.0; ψ₀(0.0, 1.0)(5.0);
ψj(0.0, 1.0)(5.0); ψj(0.0, 1.0)(5.0); ψpp1(0.0, 1.0)(5.0)]

@test w∂k   == [ψ₀′(0.0, 1.0)(-2.0); rbf(0.0, 1.0)(-2.0) ; rbf(0.0, 1.0)(-2.0); ψpp1′(0.0, 1.0)(-2.0);
ψ₀′(0.0, 1.0)(2.0); rbf(0.0, 1.0)(2.0) ; rbf(0.0, 1.0)(2.0); ψpp1′(0.0, 1.0)(2.0);
ψ₀′(0.0, 1.0)(5.0); rbf(0.0, 1.0)(5.0) ; rbf(0.0, 1.0)(5.0); ψpp1′(0.0, 1.0)(5.0)]
#
#
Nx = 5
Ne = 50
p=2

ens = EnsembleState(Nx, Ne)
Vk = KRmap(Nx, p)

W = create_weights(Vk, ens)

woff  = deepcopy(W.woff)
wdiag = deepcopy(W.wdiag)
w∂k   = deepcopy(W.w∂k)
weights(Vk, ens, woff, wdiag, w∂k)

weights(Vk, ens, W)

@test woff  == W.woff
@test wdiag == W.wdiag
@test w∂k   == W.w∂k

woval, wval, w∂kval = create_weights(Vk)
weights(Vk, zeros(5), woval, wval, w∂kval)

@test woff[:,1] == woval
@test wdiag[:,1] == wval
@test w∂k[:,1] == w∂kval
end


@testset "Weights SparseUkMap I" begin
## k = 1 and p = -1
Vk = SparseUk(1, [-1])

no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==0
@test nd ==0
@test n∂ ==0

wo, wd, w∂ = weights(Vk, [1.0])
@test wo == Float64[]
@test wd == Float64[]
@test w∂ == Float64[]

## k = 1 and p = 0
Vk = SparseUk(1, [0])

no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==0
@test nd ==1
@test n∂ ==1

wo, wd, w∂ = weights(Vk, [2.0])
@test wo == Float64[]
@test wd == [2.0]
@test w∂ == [1.0]


z = randn(1,5)
wo, wd, w∂ = weights(Vk, EnsembleState(z))
@test wo == Float64[]
@test wd == z
@test w∂ == ones(1,5)

## k = 1 and p = 2
Vk = SparseUk(1, [2])

no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==0
@test nd ==2+2
@test n∂ ==2+2

wo, wd, w∂ = weights(Vk, [2.0])
@test wo == Float64[]
@test wd == [ψ₀(0.0, 1.0)(2.0); ψj(0.0, 1.0)(2.0);ψj(0.0, 1.0)(2.0);ψpp1(0.0, 1.0)(2.0)]
@test w∂ == [ψ₀′(0.0, 1.0)(2.0); rbf(0.0, 1.0)(2.0);rbf(0.0, 1.0)(2.0);ψpp1′(0.0, 1.0)(2.0)]

z = randn(1,5)

wo, wd, w∂ = weights(Vk, EnsembleState(z))
@test wo == Float64[]

for i=1:5
    zi  = z[1,i]
    @test wd[:,i] == [ψ₀(0.0, 1.0)(zi); ψj(0.0, 1.0)(zi);ψj(0.0, 1.0)(zi);ψpp1(0.0, 1.0)(zi)]
    @test w∂[:,i] == [ψ₀′(0.0, 1.0)(zi); rbf(0.0, 1.0)(zi);rbf(0.0, 1.0)(zi);ψpp1′(0.0, 1.0)(zi)]
end

end

@testset "Weights SparseUk II" begin
## k=2 and p = [-1 -1]
Vk = SparseUk(2, [-1; -1])

no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==0
@test nd ==0
@test n∂ ==0


wo, wd, w∂ = weights(Vk, [1.0; 2.0])
@test wo == Float64[]
@test wd == Float64[]
@test w∂ == Float64[]

wo, wd, w∂ = weights(Vk, EnsembleState(randn(2,5)))
@test wo == Float64[]
@test wd == Float64[]
@test w∂ == Float64[]

## k = 2 and p = [-1 0]
Vk = SparseUk(2, [-1; 0])
no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==0
@test nd ==1
@test n∂ ==1

wo, wd, w∂ = weights(Vk, [1.0; 2.0])
@test wo == Float64[]
@test wd == [2.0]
@test w∂ == [1.0]

z = deepcopy(randn(2, 5))
wo, wd, w∂ = weights(Vk, EnsembleState(z))
@test wo == Float64[]
@test wd[1,:] == z[2,:]
@test w∂[1,:] == ones(5)

## k = 2 and p = [-1 1]
Vk = SparseUk(2, [-1; 1])
no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==0
@test nd ==3
@test n∂ ==3

wo, wd, w∂ = weights(Vk, [3.0; -2.0])
@test wo == Float64[]
@test wd == [ψ₀(0.0, 1.0)(-2.0); ψj(0.0, 1.0)(-2.0);ψpp1(0.0, 1.0)(-2.0)]
@test w∂ == [ψ₀′(0.0, 1.0)(-2.0); rbf(0.0, 1.0)(-2.0);ψpp1′(0.0, 1.0)(-2.0)]

z = randn(2,5)

wo, wd, w∂ = weights(Vk, EnsembleState(z))
@test wo == Float64[]

for i=1:5
    zi  = z[:,i]
    @test wd[:,i] == [ψ₀(0.0, 1.0)(zi[2]); ψj(0.0, 1.0)(zi[2]);ψpp1(0.0, 1.0)(zi[2])]
    @test w∂[:,i] == [ψ₀′(0.0, 1.0)(zi[2]); rbf(0.0, 1.0)(zi[2]);ψpp1′(0.0, 1.0)(zi[2])]
end

## k = 2 and p = [0 1]

Vk = SparseUk(2, [0; 1])
no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==1
@test nd ==3
@test n∂ ==3

wo, wd, w∂ = weights(Vk, [3.0; -2.0])
@test wo == [3.0]
@test wd == [ψ₀(0.0, 1.0)(-2.0); ψj(0.0, 1.0)(-2.0);ψpp1(0.0, 1.0)(-2.0)]
@test w∂ == [ψ₀′(0.0, 1.0)(-2.0); rbf(0.0, 1.0)(-2.0);ψpp1′(0.0, 1.0)(-2.0)]

z = randn(2,5)

wo, wd, w∂ = weights(Vk, EnsembleState(z))


for i=1:5
    zi  = z[:,i]
    @test norm(wo[:,i] - [zi[1]])<1e-14

    @test norm(wd[:,i]  - [ψ₀(0.0, 1.0)(zi[2]); ψj(0.0, 1.0)(zi[2]);ψpp1(0.0, 1.0)(zi[2])])<1e-14
    @test norm(w∂[:,i]  - [ψ₀′(0.0, 1.0)(zi[2]); rbf(0.0, 1.0)(zi[2]);ψpp1′(0.0, 1.0)(zi[2])])<1e-14
end

## k = 2 and p = [2 1]

Vk = SparseUk(2, [2; 1])
Vk.ξ[1] .= randn(2)
Vk.ξ[2] .= randn(3)
Vk.σ[1] .= rand(2)
Vk.σ[2] .= rand(3)

no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==3
@test nd ==3
@test n∂ ==3

wo, wd, w∂ = weights(Vk, [3.0; -2.0])
@test wo == [3.0; rbf(Vk.ξ[1][1], Vk.σ[1][1])(3.0); rbf(Vk.ξ[1][2], Vk.σ[1][2])(3.0)]
@test wd == [ψ₀(Vk.ξ[2][1], Vk.σ[2][1])(-2.0); ψj(Vk.ξ[2][2], Vk.σ[2][2])(-2.0);ψpp1(Vk.ξ[2][3], Vk.σ[2][3])(-2.0)]
@test w∂ == [ψ₀′(Vk.ξ[2][1], Vk.σ[2][1])(-2.0); rbf(Vk.ξ[2][2], Vk.σ[2][2])(-2.0);ψpp1′(Vk.ξ[2][3], Vk.σ[2][3])(-2.0)]


z = randn(2,5)

wo, wd, w∂ = weights(Vk, EnsembleState(z))


for i=1:5
    zi  = z[:,i]
    @test norm(wo[:,i]  - [zi[1]; rbf(Vk.ξ[1][1], Vk.σ[1][1])(zi[1]); rbf(Vk.ξ[1][2], Vk.σ[1][2])(zi[1])])<1e-14

    @test norm(wd[:,i]  - [ψ₀(Vk.ξ[2][1], Vk.σ[2][1])(zi[2]); ψj(Vk.ξ[2][2], Vk.σ[2][2])(zi[2]);ψpp1(Vk.ξ[2][3], Vk.σ[2][3])(zi[2])])<1e-14
    @test norm(w∂[:,i]  - [ψ₀′(Vk.ξ[2][1], Vk.σ[2][1])(zi[2]); rbf(Vk.ξ[2][2], Vk.σ[2][2])(zi[2]);ψpp1′(Vk.ξ[2][3], Vk.σ[2][3])(zi[2])])<1e-14
end


## k = 4 p = [-1 2 0 2]

Vk = SparseUk(4, [-1; 2; 0; 2])

no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==3+1
@test nd ==4
@test n∂ ==4

wo, wd, w∂ = weights(Vk, [-1.0; 4.0; 3.0; -2.0])

@test norm(wo - [4.0; rbf(0.0,1.0)(4.0); rbf(0.0,1.0)(4.0); 3.0])<1e-14
@test norm(wd - [ψ₀(0.0, 1.0)(-2.0); ψj(0.0, 1.0)(-2.0); ψj(0.0, 1.0)(-2.0);ψpp1(0.0, 1.0)(-2.0)])<1e-14
@test norm(w∂ - [ψ₀′(0.0, 1.0)(-2.0); rbf(0.0, 1.0)(-2.0); rbf(0.0, 1.0)(-2.0);ψpp1′(0.0, 1.0)(-2.0)])<1e-14

z = randn(4, 5)

wo, wd, w∂ = weights(Vk, EnsembleState(z))
for i=1:5
    zi  = z[:,i]
    @test norm(wo[:,i]  - [zi[2]; rbf(0.0,1.0)(zi[2]); rbf(0.0,1.0)(zi[2]); zi[3]])<1e-14

    @test norm(wd[:,i]  - [ψ₀(0.0, 1.0)(zi[4]); ψj(0.0, 1.0)(zi[4]); ψj(0.0, 1.0)(zi[4]);ψpp1(0.0, 1.0)(zi[4])])<1e-14
    @test norm(w∂[:,i]  - [ψ₀′(0.0, 1.0)(zi[4]); rbf(0.0, 1.0)(zi[4]); rbf(0.0, 1.0)(zi[4]);ψpp1′(0.0, 1.0)(zi[4])])<1e-14
end

## k = 5 and p = [2 -1 0 -1 0]

Vk = SparseUk(5, [2; -1; 0; -1; 0])

no, nd, n∂ = ncoeff(Vk.k, Vk.p)

@test no ==3+1
@test nd ==1
@test n∂ ==1

wo, wd, w∂ = weights(Vk, [-1.0; 4.0; 3.0; 5.0; -2.0])

@test norm(wo - [-1.0; rbf(0.0,1.0)(-1.0); rbf(0.0,1.0)(-1.0); 3.0])<1e-14
@test norm(wd - [-2.0])<1e-14
@test norm(w∂ - [1.0])<1e-14

z = randn(5, 5)

wo, wd, w∂ = weights(Vk, EnsembleState(z))
for i=1:5
    zi  = z[:,i]
    @test norm(wo[:,i]  - [zi[1]; rbf(0.0,1.0)(zi[1]); rbf(0.0,1.0)(zi[1]); zi[3]])<1e-14

    @test norm(wd[:,i]  - [zi[5]])<1e-14
    @test norm(w∂[:,i]  - [1.0])<1e-14
end

end
