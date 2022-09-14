export getreducedmargin, updatereducedmargin

"""
$(TYPEDSIGNATURES)

Computes the reduced margin of a set of multi-indices.
See Baptista, R., Zahm, O., & Marzouk, Y. (2020).
An adaptive transport framework for joint and conditional density estimation. arXiv preprint arXiv:2009.10303.
"""
function getreducedmargin(midx::Array{Int64,2})

    if isempty(midx)
        return zeros(Int64,0,0)
    end

    n, d = size(midx)

    # add [0 0 0] to compute the reduced margin
    if !any(zeros(Int64,d) in eachslice(midx; dims = 1))
            idx = vcat(zeros(Int64,1,d), midx)
    else
            idx = midx
    end

    n, d = size(idx)

    neighbours = zeros(Int64,(n,d,d))
    eye = Matrix(1*I,d,d)

    @inbounds for i=1:d
        neighbours[:,i,:]  .= idx
    end

    @inbounds for i=1:n
        neighbours[i,:,:]  .+= eye
    end

    neighbours = reshape(neighbours,(n*d,d))

    margintmp = setdiff(eachslice(neighbours; dims = 1), eachslice(idx;dims = 1))
    cardinal = size(margintmp,1)
    margin = zeros(Int64,cardinal,d)
    @inbounds for i=1:cardinal
        margin[i,:] = margintmp[i]
    end

    # margin = margin[sortperm(margin[:, 1]), :] # sorted by the 1st column
    margin = sortslices(margin; dims = 1)

    neighbours = zeros(Int64, (cardinal,d,d))

    @inbounds for i=1:d
        neighbours[:,i,:]  .= margin
    end

    @inbounds for i=1:cardinal
        neighbours[i,:,:]  .-= eye
    end

    neighbours = reshape(neighbours, cardinal*d,d)

    ok = Bool[any(x in eachslice(idx; dims = 1)) for x in eachslice(neighbours; dims = 1)]

    isout = Bool[any(x.<0) for x in eachslice(neighbours; dims = 1)]

    ok .|=  isout

    ok = reshape(ok, (cardinal, d))

    keep = Bool[all(x) for x in eachslice(ok; dims = 1)]



    return margin[keep,:]
end

"""
$(TYPEDSIGNATURES)

Updates the reduced margin.

See Baptista, R., Zahm, O., & Marzouk, Y. (2020).
An adaptive transport framework for joint and conditional density estimation. arXiv preprint arXiv:2009.10303.
"""
function updatereducedmargin(lowerset::Array{Int64,2}, reduced_margin::Array{Int64,2}, idx::Int64)
    d = size(lowerset, 2)
    nr = size(reduced_margin, 1)
    @assert idx <= nr "idx is larger than dimension of the reduced_margin"
    newidx = reduced_margin[idx:idx,:]
#     @show lowerset
#     @show reduced_margin
#     @show newidx
    lowerset = vcat(lowerset, newidx)
#     @show lowerset
    # Remove the idx-th line of reducedmargin
    toremove = Bool[!(i in idx) for i=1:nr]
    # reduced_margin = reduced_margin[1:end .!= idx,:]
    reduced_margin = reduced_margin[toremove,:]

#     @show toremove
#     @show reduced_margin

    eye =  Matrix(1*I,d,d)
    # Update the reduced margin
    candidate  = repeat(newidx, outer = (d,1)) + eye

#     @show candidate
    ok = falses(d)
#     @show ok

    @inbounds for i = 1:d
#         @show i
#         @show repeat(candidate[i,:]', outer = (d,1))
#         @show size(repeat(candidate[i,:]', outer = (d,1)))
        parents_of_candidate = repeat(candidate[i,:]', outer = (d,1)) - eye
#         @show parents_of_candidate
        tokeep = Bool[!any(x .< 0) for x in eachslice(parents_of_candidate; dims = 1)]
#         @show tokeep
        parents_of_candidate = parents_of_candidate[tokeep,:]
#         @show parents_of_candidate

#         @show [any(x in eachslice(lowerset; dims = 1)) for x in eachslice(parents_of_candidate; dims = 1)]
        ok[i] = all([any(x in eachslice(lowerset; dims = 1)) for x in eachslice(parents_of_candidate; dims = 1)])
#        @show ok
    end
    #
    candidate = candidate[ok,:]
    #
    # # Add the candidates to the reduced margin
    reduced_margin = vcat(reduced_margin, candidate)

    # reduced_margin  = sortslices(reduced_margin; dims = 1)
    # lowerset = sortslices(lowerset; dims = 1)
    return lowerset, reduced_margin
end
