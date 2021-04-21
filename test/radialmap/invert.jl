using Test

using LinearAlgebra, Statistics
using SpecialFunctions, ForwardDiff
using TransportMap

@testset "Bracket root" begin
    u = uk(3)
    u.ξk .= sort!(randn(5))
    u.σk .= σscale(u.ξk,2.0)
    u.ak .= rand(6);
    κ = 5.0
    xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])

    zlim = bracket(u, xlim[1], xlim[2])

    @test u(zlim[1])*u(zlim[2])<0.0

end
@testset "invert_uk p=-1" begin
    u = uk(-1)
    zopt = invert_uk(u,2.0,500.0)
    @test zopt == 2.0
    @test u(zopt) == 2.0
end

@testset "invert_uk p=0" begin
    u = uk(0)
    u.ak .= [1.0;2.0]
    zopt = invert_uk(u,2.0,500.0)
    @test zopt ==(2.0-1.0)/2.0
    @test u(zopt) == 2.0
end

@testset "invert_uk p>0" begin
    u = uk(5)
    u.ξk .= sort!(randn(7))
    u.σk .= σscale(u.ξk,2.0)
    u.ak .= rand(8);

    zopt = invert_uk(u,1.0,500.0)
    @test norm(u(zopt) -1.0)<1e-9

    # u = uk(5, randn(7), rand(7), [-0.5; rand(7)])
    #
    # zopt = invert_uk(u,1.0;z0 = 0.1)
    # @test norm(u(zopt) -1.0)<1e-10
end

@testset "Invert S" begin
    k = 100
    Ne = 200
    p = 2
    γ = 2.0
    λ = 0.1
    δ = 1e-5
    κ = 10.0
    ens = EnsembleState(k, Ne)
    ens⁺ = EnsembleState(k, Ne)
    ens.S .= randn(k,Ne) .* randn(k,Ne)
    S = KRmap(k, p, γ=γ, λ=λ, δ=δ, κ=κ)
    run_optimization(S, ens);



    Sval = S(ens)
    # Truth is a weak pertrubation
    ystar = ens.S[1:50,1] + 0.01*cos.(randn(50))
    for i=1:Ne
    zplus = view(ens⁺.S,:,i)
    invert_S(S, view(Sval,:,i), ystar, zplus)
    end

    @test norm(S(ens⁺)[51:k,:] - Sval[51:k,:])<1e-8
end


@testset "Invert S with Multi-threading" begin
    k = 100
    Ne = 200
    p = 2
    γ = 2.0
    λ = 0.1
    δ = 1e-5
    κ = 10.0
    ens = EnsembleState(k, Ne)
    ens⁺ = EnsembleState(k, Ne)
    ens.S .= randn(k,Ne) .* randn(k,Ne)
    Sserial = KRmap(k, p, γ=γ, λ=λ, δ=δ, κ=κ)
    Sthread = KRmap(k, p, γ=γ, λ=λ, δ=δ, κ=κ)
    run_optimization(Sserial, ens; P = serial);
    run_optimization(Sthread, ens; P = thread);

    Sval = Sserial(ens)
    @test norm(Sserial(ens)-Sthread(ens))<1e-12
    # Truth is a weak pertrubation
    ystar = ens.S[1:50,1] + 0.01*cos.(randn(50))
    Threads.@threads for i=1:Ne
    zplus = view(ens⁺.S,:,i)
    invert_S(Sserial, view(Sval,:,i), ystar, zplus)
    invert_S(Sthread, view(Sval,:,i), ystar, zplus)

    end

    @test norm(Sserial(ens⁺)[51:k,:] - Sval[51:k,:])<1e-8
    @test norm(Sthread(ens⁺)[51:k,:] - Sval[51:k,:])<1e-8

end

#
# @testset "Benchmark Algorithm" begin
# function compare_algo()
# @btime begin
# for i=1:1000
# u = uk(3)
# u.ξk .= sort!(randn(5))
# u.σk .= σscale(u.ξk,2.0)
# u.ak .= rand(6);
# κ = 5.0
# xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# zlim = bracket(u, xlim[1], xlim[2])
# end
# end
#
# @btime begin
# for i=1:1000
# u = uk(3)
# u.ξk .= sort!(randn(5))
# u.σk .= σscale(u.ξk,2.0)
# u.ak .= rand(6);
# κ = 5.0
# xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# zlim = bracket(u, xlim[1], xlim[2])
# Roots.bisection(u, zlim[1], zlim[2])
# end
# end
#
# @btime begin
# for i=1:1000
# u = uk(3)
# u.ξk .= sort!(randn(5))
# u.σk .= σscale(u.ξk,2.0)
# u.ak .= rand(6);
# κ = 5.0
# xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# zlim = bracket(u, xlim[1], xlim[2])
# find_zero(u, zlim, Bisection())
# end
# end
#
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.ak .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero(u, zlim, Roots.A42())
# # end
# # end
#
# @btime begin
# for i=1:1000
# u = uk(3)
# u.ξk .= sort!(randn(5))
# u.σk .= σscale(u.ξk,2.0)
# u.ak .= rand(6);
# κ = 5.0
# xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# zlim = bracket(u, xlim[1], xlim[2])
# find_zero(u, zlim, Roots.Brent())
# end
# end
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.ak .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # Roots.secant_method(u, zlim)
# # end
# # end
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.ak .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero(u, zlim, Roots.Secant())
# # end
# # end
#
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.ak .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero((z->u(z), D(u),H(u)), mean(zlim), Roots.Newton())
# # end
# # end
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.ak .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero((z->u(z), D(u),H(u)), mean(zlim), Roots.Newton())
# # end
# # end
# #
# # @btime begin
# # for i=1:1000
# # u = uk(3)
# # u.ξk .= sort!(randn(5))
# # u.σk .= σscale(u.ξk,2.0)
# # u.ak .= rand(6);
# # κ = 5.0
# # xlim = (u.ξk[1]-κ*u.σk[1], u.ξk[end]+κ*u.σk[end])
# # zlim = bracket(u, xlim[1], xlim[2])
# # find_zero((z->u(z), D(u),H(u)), mean(zlim), Roots.Halley())
# # end
# # end
#
# end
# compare_algo()
#
# end
