#let's see if we can get an example of the non-adaptive map filter to work
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
x0 = rand(model.π0)
data = generate_lorenz63(model, x0, Tf)

XY = [data.xt; data.yt]

#have to set up some extra stuff for the map filter
m = 10 #number of map components
S = HermiteMap(m, XY, diag=true, α=1e-4)

#now, define the mf map filter! We'll see how this goes...
#mfmf = MfMapFilter()

#stochenkf call will not work
#mfmf = MfMapFilter(model.ϵy, model.Δtdyn, model.Δtobs)
