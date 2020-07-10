export getreducedmargin#, updatereducedmargin

function getreducedmargin(idx::Array{Int64,2})

    if isempty(idx)
        return zeros(Int64,0,0)
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
