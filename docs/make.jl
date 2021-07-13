using Documenter, TransportBasedInference, Plots, PyPlot, ColorSchemes


ENV["GKSwstype"] = "nul"

makedocs(
    sitename = "TransportBasedInference.jl",
    doctest = true,
    clean = true,
    pages = [
        "Home" => "index.md",
        "Background" => "background.md",
        "Tutorials" => [
                     "manual/1.-Estimation-of-the-Banana-distribution.md",
                     # "manual/2.-Basic-flow-with-a-stationary-body.md",
                     "manual/3.-Structure-discovery-of-the-Lorenz-96.md",
                     "manual/4.-Linear-ensemble-filtering-Lorenz-63.md",
                     "manual/5.-Linear-ensemble-filtering-Lorenz-96-with-localization.md",
                     "manual/6.-Radial-basis-nonlinear-ensemble-filtering-Lorenz-63.md",
                     "manual/7.-Radial-basis-nonlinear-ensemble-filtering-Lorenz-96.md"
                     ],
        "API Documentation" => ["apidoc.md"],
        "Community guidelines" => ["contribute.md"],
        "LICENSE.md"
    ],
    #format = Documenter.HTML(assets = ["assets/custom.css"])
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax(Dict(
            :TeX => Dict(
                :equationNumbers => Dict(:autoNumber => "AMS"),
                :Macros => Dict()
            )
        ))
    ),
    #assets = ["assets/custom.css"],
    #strict = true
)


#if "DOCUMENTER_KEY" in keys(ENV)
deploydocs(
     repo = "github.com/mleprovost/TransportBasedInference.jl.git",
     target = "build",
     deps = nothing,
     make = nothing
     #versions = "v^"
)
#end
