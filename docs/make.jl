using Documenter
using Arianna

readme = read(joinpath(@__DIR__, "..", "README.md"), String)
html_part = readme[1:findlast("</p>", readme)[end]]
html_part = replace(html_part, r"<div align=\"center\">[\s\S]*?</div>" => "")
md_part = readme[findlast("</p>", readme)[end]+1:end]
readme = "```@raw html\n" * html_part * "\n```\n" * md_part
write(joinpath(@__DIR__, "src", "index.md"), readme)

makedocs(
    sitename = "Arianna",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold_ignore = ["api.md"],
        sidebar_sitename = false
    ),
    modules = [Arianna],
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "man/montecarlo.md",
            "man/system.md",
            "man/policyguided.md"
        ],
        "Related packages" => "related.md",
        "API" => "api.md",
    ]
)

# Deploying to GitHub Pages
deploydocs(;
    repo = "github.com/TheDisorderedOrganization/Arianna.jl.git",
)