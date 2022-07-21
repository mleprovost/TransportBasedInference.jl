using Documenter
using TransportBasedInference


makedocs(
    modules = [TransportBasedInference],
    sitename = "TransportBasedInference.jl",
    doctest = true,
    clean = true,
    pages = [
        "Home" => "index.md"
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax2(Dict(
            :TeX => Dict(
                :equationNumbers => Dict(:autoNumber => "AMS"),
                :Macros => Dict()
            )
        ))
    )
)
