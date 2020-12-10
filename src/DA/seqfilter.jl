export SeqFilter, IdFilter, idfilter

abstract type SeqFilter end

# Define identity filter
struct IdFilter<:SeqFilter end

const idfilter = IdFilter()

function (filter::IdFilter)(X, ystar, t)
	return X
end
