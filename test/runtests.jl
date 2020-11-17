using VisualParticipationAnalytics
using Clustering
using Languages
using SparseArrays
using SQLite
using Tables
using Test
using TextAnalysis

@testset "VisualParticipationAnalytics" begin
    include("io.jl")
    include("preprocessing.jl")

    @testset "similarities" begin
        doc1 = StringDocument("a b")
        doc2 = StringDocument("b c")

        crps = Corpus([doc1, doc1])
        sims = similarities(crps)
        @test sims[1] ≈ sims[4] ≈ 1.0 && isnan(sims[2]) && isnan(sims[3])
        @test similarities(crps, false) ≈ [1.0 1.0; 1.0 1.0]

        crps = Corpus([doc1, doc2])
        @test similarities(crps) ≈ [1.0 0.0; 0.0 1.0]
        @test similarities(crps, false) ≈ [1.0 0.5; 0.5 1.0]
    end

    @testset "clustering" begin
        doc1 = StringDocument("a")
        doc2 = StringDocument("b")
        doc3 = StringDocument("c")
        crps = Corpus([doc1, doc2, doc3])

        c = clustering(crps, 2)
        @test typeof(c) <: ClusteringResult && nclusters(c) == 2

        c = clustering(crps, 2, false)
        @test typeof(c) <: ClusteringResult && nclusters(c) == 2
    end

    @testset "topicmodel" begin
        doc1 = StringDocument("a")
        doc2 = StringDocument("b")
        crps = Corpus([doc1, doc2])
        topicword, topicdoc = topicmodel(crps, 2, 100, 0.1, 0.1)
        @test typeof(topicword) == SparseMatrixCSC{Float64,Int}
        @test typeof(topicdoc) == Array{Float64,2}
    end

    @testset "topkwords" begin
        doc1 = StringDocument("programming in julia")
        doc2 = StringDocument("python programming")
        doc3 = StringDocument("julia and ada")
        crps = Corpus([doc1, doc2, doc3])
        topicword, topicdoc = topicmodel(crps, 2, 100, 0.1, 0.1)
        words = topkwords(topicword, 1, crps, 2)
        @test typeof(words) == Array{Tuple{String,Float64},1}
        @test length(words) == 2
        @test words[1][2] >= words[2][2]
    end
end
