using Documenter
using VisualParticipationAnalytics

makedocs(
    modules = [VisualParticipationAnalytics],
    sitename = "VisualParticipationAnalytics.jl",
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(repo = "github.com/laschuet/VisualParticipationAnalytics.jl.git")
