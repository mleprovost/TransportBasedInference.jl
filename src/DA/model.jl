export Model, SyntheticData

"""
        Model

A structure to perform the twin-experiment

## Fields
- `Nx` : Dimension of the state variable
- `Ny` : Dimension of the observation variable
- `Δtdyn` : Time-step for the dynamical model
- `Δtobs` : Time step between two observations of the state
- `ϵx` : Process noise
- `ϵy` : Observation noise
- `π0` : Multivariate distribution for the initial condition
- `Tburn` : Number of steps to burn from the end of the spin up to compute the metrics
- `Tstep` : Number of steps for which the filter to be tested is applied
- `Tspinup` : Number of steps of spin-up phase,
   i.e. number of steps to generate the initial ensemble for the filtering algorithms
- `F` : State-Space Model
"""
struct Model

    "Dimension of the state variable"
    Nx::Int64

	"Dimension of the observation variable"
    Ny::Int64

	"Time-step for the dynamical model"
    Δtdyn::Float64

	"Time step between two observations of the state"
    Δtobs::Float64

	"Process noise"
    ϵx::InflationType

	"Observation noise"
    ϵy::AdditiveInflation

    "Multivariate distribution for the initial condition"
    π0::ContinuousMultivariateDistribution

    "Number of steps to burn from the end of the spin up to compute the metrics"
    Tburn::Int64
    Tstep::Int64
    Tspinup::Int64

    " State-Space Model"
    F::StateSpace
end

"""
    SyntheticData

A structure to store the synthetic data in a twin-experiment


## Fields
- `tt` : time history
- `x0` : the initial condition
- `xt` : history of the state
- `yt` : history of the observations
"""
struct SyntheticData
	tt::Array{Float64,1}
	Δt::Float64
	x0::Array{Float64,1}
	xt::Array{Float64,2}
	yt::Array{Float64,2}
end
