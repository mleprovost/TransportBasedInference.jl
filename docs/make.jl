using Documenter, TransportBasedInference, Plots

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "1.-Estimation-of-the-Banana-distribution.jl",
    "2.-Conditional-density-estimation-of-the-Banana-distribution.jl",
    "3.-Structure-discovery-of-the-Lorenz-96.jl",
    "4.-Linear-ensemble-filtering-Lorenz-63.jl",
]

function uncomment_objects(str)
    str = replace(str, "###```@raw" => "```\n\n```@raw")
    str = replace(str, "###<object" => "<object")
    str = replace(str, "###```\n```" => "```")
    str
end

for example in examples
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR; execute=true, postprocess = uncomment_objects)
end

# makedocs(
#     modules = [TransportBasedInference],
#     sitename = "TransportBasedInference.jl",
#     doctest = true,
#     clean = true,
#     pages = [
#         "Home" => "index.md"
#     ],
#     format = Documenter.HTML(
#         prettyurls = get(ENV, "CI", nothing) == "true",
#         mathengine = MathJax2(Dict(
#             :TeX => Dict(
#                 :equationNumbers => Dict(:autoNumber => "AMS"),
#                 :Macros => Dict()
#             )
#         ))
#     )
# )

makedocs(
            doctest = false,
            format = Documenter.HTML(),
            sitename = "TransportBasedInference.jl",
            authors = "Mathieu Le Provost",
            pages = Any[
                    "Home" => "index.md",
                    # "Development" => "dev.md",
                    "Examples" => [
                        "generated/1.-Estimation-of-the-Banana-distribution.md",
                        "generated/2.-Conditional-density-estimation-of-the-Banana-distribution.md",
                        "generated/3.-Structure-discovery-of-the-Lorenz-96.md",
                        "generated/4.-Linear-ensemble-filtering-Lorenz-63.md",
                        ],
                    ]
        )


# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/mleprovost/TransportBasedInference.jl.git"
)