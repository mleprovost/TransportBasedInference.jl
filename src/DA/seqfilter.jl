export SeqFilter, IdFilter, idfilter

"""
    SeqFilter

An abstract type for the ensemble filtering algorithms.

"""
abstract type SeqFilter end

"""
    IdFilter <: SeqFilter

An immutable structure for the identity filter.

## Fields

- `Δtdyn::Float64`: time-step of the dynamical model
- `Δtobs::Float64`: time-step between two observations

## Constructors

- `IdFilter(Δtdyn, Δtobs)`: set up an identity filter
"""
struct IdFilter<:SeqFilter
	"Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64
end


"""
	(filter::IdFilter)(X, ystar, t)

	Applies the identity transformation to the ensemble `X`.
"""
function (filter::IdFilter)(X, ystar, t)
	return X
end
