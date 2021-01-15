export quant!, quant

using Base: has_offset_axes, require_one_based_indexing
# Cpopy of quantile from Statistics.jl with alpha and beta version of quantile

"""
    quant!([q::AbstractArray, ] v::AbstractVector, p; sorted=false, alpha::Real=1.0, beta::Real=alpha)
Compute the quantile(s) of a vector `v` at a specified probability or vector or tuple of
probabilities `p` on the interval [0,1]. If `p` is a vector, an optional
output array `q` may also be specified. (If not provided, a new output array is created.)
The keyword argument `sorted` indicates whether `v` can be assumed to be sorted; if
`false` (the default), then the elements of `v` will be partially sorted in-place.
By default (`alpha = beta = 1`), quantiles are computed via linear interpolation between the points
`((k-1)/(n-1), v[k])`, for `k = 1:n` where `n = length(v)`. This corresponds to Definition 7
of Hyndman and Fan (1996), and is the same as the R and NumPy default.
The keyword arguments `alpha` and `beta` correspond to the same parameters in Hyndman and Fan,
setting them to different values allows to calculate quantiles with any of the methods 4-9
defined in this paper:
- Def. 4: `alpha=0`, `beta=1`
- Def. 5: `alpha=0.5`, `beta=0.5`
- Def. 6: `alpha=0`, `beta=0` (Excel `PERCENTILE.EXC`, Python default, Stata `altdef`)
- Def. 7: `alpha=1`, `beta=1` (Julia, R and NumPy default, Excel `PERCENTILE` and `PERCENTILE.INC`, Python `'inclusive'`)
- Def. 8: `alpha=1/3`, `beta=1/3`
- Def. 9: `alpha=3/8`, `beta=3/8`
!!! note
    An `ArgumentError` is thrown if `v` contains `NaN` or [`missing`](@ref) values.
# References
- Hyndman, R.J and Fan, Y. (1996) "Sample Quantiles in Statistical Packages",
  *The American Statistician*, Vol. 50, No. 4, pp. 361-365
- [Quantile on Wikipedia](https://en.m.wikipedia.org/wiki/Quantile) details the different quantile definitions
# Examples
```jldoctest
julia> using Statistics
julia> x = [3, 2, 1];
julia> quant!(x, 0.5)
2.0
julia> x
3-element Array{Int64,1}:
 1
 2
 3
julia> y = zeros(3);
julia> quant!(y, x, [0.1, 0.5, 0.9]) === y
true
julia> y
3-element Array{Float64,1}:
 1.2000000000000002
 2.0
 2.8000000000000003
```
"""
function quant!(q::AbstractArray, v::AbstractVector, p::AbstractArray;
                   sorted::Bool=false, alpha::Real=1.0, beta::Real=alpha)
    require_one_based_indexing(q, v, p)
    if size(p) != size(q)
        throw(DimensionMismatch("size of p, $(size(p)), must equal size of q, $(size(q))"))
    end
    isempty(q) && return q

    minp, maxp = extrema(p)
    _quantilesort!(v, sorted, minp, maxp)

    for (i, j) in zip(eachindex(p), eachindex(q))
        @inbounds q[j] = _quantile(v,p[i], alpha=alpha, beta=beta)
    end
    return q
end

function quant!(v::AbstractVector, p::Union{AbstractArray, Tuple{Vararg{Real}}};
                   sorted::Bool=false, alpha::Real=1., beta::Real=alpha)
    if !isempty(p)
        minp, maxp = extrema(p)
        _quantilesort!(v, sorted, minp, maxp)
    end
    return map(x->_quantile(v, x, alpha=alpha, beta=beta), p)
end

quant!(v::AbstractVector, p::Real; sorted::Bool=false, alpha::Real=1., beta::Real=alpha) =
    _quantile(_quantilesort!(v, sorted, p, p), p, alpha=alpha, beta=beta)

# Function to perform partial sort of v for quantiles in given range
function _quantilesort!(v::AbstractArray, sorted::Bool, minp::Real, maxp::Real)
    isempty(v) && throw(ArgumentError("empty data vector"))
    require_one_based_indexing(v)

    if !sorted
        lv = length(v)
        lo = floor(Int,minp*(lv))
        hi = ceil(Int,1+maxp*(lv))

        # only need to perform partial sort
        sort!(v, 1, lv, Base.Sort.PartialQuickSort(lo:hi), Base.Sort.Forward)
    end
    ismissing(v[end]) && throw(ArgumentError("quantiles are undefined in presence of missing values"))
    isnan(v[end]) && throw(ArgumentError("quantiles are undefined in presence of NaNs"))
    return v
end

# Core quantile lookup function: assumes `v` sorted
@inline function _quantile(v::AbstractVector, p::Real; alpha::Real=1.0, beta::Real=alpha)
    0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))
    0 <= alpha <= 1 || throw(ArgumentError("alpha parameter out of [0,1] range"))
    0 <= beta <= 1 || throw(ArgumentError("beta parameter out of [0,1] range"))
    require_one_based_indexing(v)

    n = length(v)
    m = alpha + p * (one(alpha) - alpha - beta)
    aleph = n*p + oftype(p, m)
    j = clamp(trunc(Int, aleph), 1, n-1)
    γ = clamp(aleph - j, 0, 1)

    a = v[j]
    b = v[j + 1]

    if isfinite(a) && isfinite(b)
        return a + γ*(b-a)
    else
        return (1-γ)*a + γ*b
    end
end

"""
    quant(itr, p; sorted=false, alpha::Real=1.0, beta::Real=alpha)
Compute the quantile(s) of a collection `itr` at a specified probability or vector or tuple of
probabilities `p` on the interval [0,1]. The keyword argument `sorted` indicates whether
`itr` can be assumed to be sorted.
Samples quantile are defined by `Q(p) = (1-γ)*x[j] + γ*x[j+1]`,
where ``x[j]`` is the j-th order statistic, and `γ` is a function of
`j = floor(n*p + m)`, `m = alpha + p*(1 - alpha - beta)` and
`g = n*p + m - j`.
By default (`alpha = beta = 1`), quantiles are computed via linear interpolation between the points
`((k-1)/(n-1), v[k])`, for `k = 1:n` where `n = length(itr)`. This corresponds to Definition 7
of Hyndman and Fan (1996), and is the same as the R and NumPy default.
The keyword arguments `alpha` and `beta` correspond to the same parameters in Hyndman and Fan,
setting them to different values allows to calculate quantiles with any of the methods 4-9
defined in this paper:
- Def. 4: `alpha=0`, `beta=1`
- Def. 5: `alpha=0.5`, `beta=0.5`
- Def. 6: `alpha=0`, `beta=0` (Excel `PERCENTILE.EXC`, Python default, Stata `altdef`)
- Def. 7: `alpha=1`, `beta=1` (Julia, R and NumPy default, Excel `PERCENTILE` and `PERCENTILE.INC`, Python `'inclusive'`)
- Def. 8: `alpha=1/3`, `beta=1/3`
- Def. 9: `alpha=3/8`, `beta=3/8`
!!! note
    An `ArgumentError` is thrown if `v` contains `NaN` or [`missing`](@ref) values.
    Use the [`skipmissing`](@ref) function to omit `missing` entries and compute the
    quantiles of non-missing values.
# References
- Hyndman, R.J and Fan, Y. (1996) "Sample Quantiles in Statistical Packages",
  *The American Statistician*, Vol. 50, No. 4, pp. 361-365
- [Quantile on Wikipedia](https://en.m.wikipedia.org/wiki/Quantile) details the different quantile definitions
# Examples
```jldoctest
julia> using Statistics
julia> quantile(0:20, 0.5)
10.0
julia> quantile(0:20, [0.1, 0.5, 0.9])
3-element Array{Float64,1}:
  2.0
 10.0
 18.000000000000004
julia> quantile(skipmissing([1, 10, missing]), 0.5)
5.5
```
"""
quant(itr, p; sorted::Bool=false, alpha::Real=1.0, beta::Real=alpha) =
    quant!(collect(itr), p, sorted=sorted, alpha=alpha, beta=beta)

quant(v::AbstractVector, p; sorted::Bool=false, alpha::Real=1.0, beta::Real=alpha) =
    quant!(sorted ? v : Base.copymutable(v), p; sorted=sorted, alpha=alpha, beta=beta)
