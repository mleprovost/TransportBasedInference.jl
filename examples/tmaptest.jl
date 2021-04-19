#let's see if we can get an example of the non-adaptive map filter to work
using Revise
using AdaptiveTransportMap
using LinearAlgebra
using Statistics
using Distributions

#setup Lorenz Model

#state dimensions
Nx = 3
Ny = 3

#time steps for dynamic and observation models
Δtdyn = 0.05
Δtobs = 0.1

#time span of interest
t0 = 0.0
tf = 100.0
Tf = ceil(Int64, (tf - t0)/Δtobs) #number of steps to propagate filter


#distribution of initial condition
π0 = MvNormal(zeros(Nx), Matrix(1.0*I, Nx, Nx))

# Define the observation operator and the dynamics
# (will create a simpler dynamic model later)
h(x,t) = x
F = StateSpace(lorenz63!, h)

# Define additive noise
σx = 1e-1
σy = 4.0

ϵx = AdditiveInflation(Nx, zeros(Nx), σx) #added to the dynamics
ϵy = AdditiveInflation(Ny, zeros(Ny), σy) #observation noise

#put together the model
model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx, ϵy, π0, 0, 0, 0, F)

#set the initial condition of the TRUE state and generate true data
x0 = rand(model.π0) #TODO may need to put this x0 into prob
data = generate_lorenz63(model, x0, Tf)

#let's see if we can instantiate a TMap
p = 3 #degree
γ = 2.0
λ = 0.1
δ = 1e-8
κ = 10.0

mapper = TMap(Nx, Ny, p, γ, λ, δ, κ, x -> x, ϵy, Δtdyn, Δtobs, false, false)

##
#aaah it worked! After much weeping and gnashing of teeth
# need to make an ensemble now
Ne = 500

#generate initial conditions
ens = EnsembleStateMeas(Nx, Ny, Ne)
ens.state.S .= rand(model.π0, Ne)

## see if this filtering thing works

nassim = ceil(Int64, (tf - t0)/Δtobs)
Xassim = seqassim(F, data, nassim, model.ϵx, mapper, ens, model.Ny, model.Nx, t0)


# we now have TMap <: SeqFilter, which will be a better starting point for
# mfmapfilter than AdaptiveTransportMap. Figure out how to initialize...
# need to find documentation on what all of these parameters mean
