export EnsembleState, EnsembleStateMeas, norm, sort, sort!, member, deviation!, deviation

import Base: size, +, -, sort, sort!

import Statistics: mean, cov

import LinearAlgebra: norm


# """
#     EnsembleState
#
# An structure  for Ensemble.
#
# Fields:
# - 'S' : Array of the different members
# """
struct EnsembleState{Nx, Ne}
    " Array of the different ensemble members"
    S::Array{Float64,2}
end

function EnsembleState(Nx::Int64, Ne::Int64)
    return EnsembleState{Nx, Ne}(zeros(Nx,Ne))
end

size(ens::EnsembleState{Nx, Ne}) where {Nx, Ne} = (Nx, Ne)

function EnsembleState(States::Array{Float64,2})
    Nx, Ne = size(States)
    return EnsembleState{Nx, Ne}(States)
end


# """
#     Define addition of two EnsembleState
# """
function (+)(A::EnsembleState{Nx, Ne}, B::EnsembleState{Nx, Ne}) where {Nx, Ne}
    C = deepcopy(A)
    C.S .+= B.S
    return C
end

# """
#     Define addition of an Array and an EnsembleState
# """
function (+)(A::EnsembleState{Nx, Ne}, B::Array{Float64,1}) where {Nx, Ne}
    C = deepcopy(A)
    C .+= B
    return C
end


# """
#     Define substraction of two EnsembleState
# """
function (-)(A::EnsembleState{Nx, Ne}, B::EnsembleState{Nx, Ne}) where {Nx, Ne}
    C = deepcopy(A)
    C.S .-= B.S
    return C
end

# """
#     Define substraction of an Array from an EnsembleState
# """
function (-)(A::EnsembleState{Nx, Ne}, B::Array{Float64,1}) where {Nx, Ne}
    C = deepcopy(A)
    C .-= B
    return C
end

# """
#     Define norm of an EnsembleState variable, equivalent to norm(hcat(ens))
# """

norm(A::EnsembleState{Nx, Ne}) where {Nx, Ne} = norm(A.S)

sort(A::EnsembleState{Nx, Ne}, dims::Int64) where {Nx, Ne} = sort(A.S, dims = dims)
sort!(A::EnsembleState{Nx, Ne}, dims::Int64) where {Nx, Ne} = sort!(A.S, dims = dims)

member(ens::EnsembleState{Nx, Ne}, idx::Int64)  where{Nx, Ne}  = ens.S[:,idx]

mean(ens::EnsembleState{Nx, Ne}) where {Nx, Ne} = mean(ens.S, dims=2)[:,1]
# Rescaled by 1/(Ne-1)
cov(ens::EnsembleState{Nx, Ne}) where {Nx, Ne} = Statistics.cov(ens.S, dims=2)

function deviation!(fluc::EnsembleState{Nx, Ne}, ens::EnsembleState{Nx, Ne}) where {Nx, Ne}
S = mean(ens)
@. fluc.S = ens.S - S
return fluc
end

deviation(ens::EnsembleState{Nx, Ne}) where {Nx, Ne} = deviation!(EnsembleState(Nx, Ne), ens)


struct EnsembleStateMeas{Nx, Ny, Ne}
    " Array of state for the different ensemble members"
    state::EnsembleState{Nx, Ne}

    " Array of measurement for the different ensemble members"
    meas::EnsembleState{Ny, Ne}
end

EnsembleStateMeas(Nx, Ny, Ne) = EnsembleStateMeas{Nx, Ny, Ne}(EnsembleState(Nx, Ne), EnsembleState(Ny, Ne))

function EnsembleStateMeas(state::Array{Float64,2}, meas::Array{Float64,2})

    Nx, Nes = size(state)
    Ny, Nem = size(meas)
    @assert Nes==Nem "Error dimension of the ensemble"
    Ne = Nes

    return EnsembleStateMeas{Nx, Ny, Ne}(EnsembleState{Nx, Ne}(state), EnsembleState{Ny, Ne}(meas))
end

size(ens::EnsembleStateMeas{Nx, Ny, Ne}) where {Nx, Ny, Ne} = (Nx, Ny, Ne)
