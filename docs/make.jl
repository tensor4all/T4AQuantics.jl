using T4AQuantics
using Documenter

DocMeta.setdocmeta!(T4AQuantics, :DocTestSetup, :(using T4AQuantics); recursive=true)

makedocs(;
    modules=[T4AQuantics],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename="T4AQuantics.jl",
    remotes=nothing,
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/T4AQuantics.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md"
    ])

deploydocs(;
    repo="github.com/tensor4all/T4AQuantics.jl.git",
    devbranch="main"
)
