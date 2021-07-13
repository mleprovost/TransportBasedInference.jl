# Contribute to TransportBasedInference.jl

We welcome contributions to TransportBasedInference.jl. The modularity of the package eases the integration of new parametrizations of transport maps, new ensemble filtering algorithms, new ensemble inflation scheme... To contribute a new feature, please submit a pull request.

New types of inflation can easily be created and integrated in `TransportBasedInference`, as long as they satisfy the following requirements:

* `MyInflationType <: InflationType`
* `(A::MyInflationType)(X::AbstractMatrix{Float64})` is defined

Similarly, new ensemble filter only need to satisfy the following requirements:

* `MyFilterType <: SeqFilter`
* `(A::MyFilterType)(X::AbstractMatrix{Float64}, ystar, t)` is defined, where `ystar` is the observation to assimilate in the forecast ensemble `X` at time `t`.

# Reporting issues or problems

If you find a bug in the code, please open an issue with a minimal working example.

# Seek support

To reach support with TransportBasedInference.jl, open a post on **[Julia discourse](https://discourse.julialang.org)**.
