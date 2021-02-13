# AdaptiveTransportMap.jl

*a framework for estimation of transport maps between two densities from samples*

This code is based on the adaptive transport framework developed by Baptista et al [^1].

## Installation

This package works on Julia `1.4` and above and is registered in the general Julia registry. To install from the REPL, type
e.g.,
```julia
] add AdaptiveTransportMap
```

Then, in any version, type
```julia
julia> using AdaptiveTransportMap
```

The plots in this documentation are generated using [Plots.jl](http://docs.juliaplots.org/latest/).
You might want to install that, too, to follow the examples.

## References

[^1]: Baptista, R., Zahm, O., & Marzouk, Y. (2020). An adaptive transport framework for joint and conditional density estimation. arXiv preprint arXiv:2009.10303.
