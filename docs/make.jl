using Documenter
using ParticipationAnalytics

makedocs(
    modules = [ParticipationAnalytics],
    sitename = "ParticipationAnalytics.jl",
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(repo = "github.com/laschuet/ParticipationAnalytics.jl.git")
