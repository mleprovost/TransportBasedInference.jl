export SeqFilter, IdFilter, idfilter

abstract type SeqFilter end

# Define identity filter
struct IdFilter<:SeqFilter
	"Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64
end

# const idfilter = IdFilter()

function (filter::IdFilter)(X, ystar, t)
	return X
end
