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


# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/mleprovost/TransportBasedInference.jl.git"
)
