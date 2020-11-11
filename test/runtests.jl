using ParticipationAnalytics
using Clustering
using Languages
using SparseArrays
using Test
using TextAnalysis

@testset "slatetext" begin
    @test slatetext("{\"document\":{\"nodes\":[]}}") == ""

    str = """
        {
            \"document\": {
                \"nodes\": [{
                    \"kind\": \"text\",
                    \"leaves\": [{
                        \"text\": \"Julia\"
                    }]
                }, {
                    \"kind\": \"block\",
                    \"nodes\": [{
                        \"kind\": \"text\",
                        \"leaves\": [{
                            \"text\": \"is fun\"
                        }]
                    }]
                }]
            }
        }
    """
    @test slatetext(str) == "Julia is fun"
end

@testset "preprocess" begin
    function test(input::AbstractString, output::AbstractString, lang::Language)
        doc = StringDocument(input)
        language!(doc, lang)
        preprocess!(doc)
        @test text(doc) == output

        doc = StringDocument(input)
        crps = Corpus([doc])
        languages!(crps, lang)
        preprocess!(crps)
        @test text(crps[1]) == output

        @test preprocess(input) == output
    end

    @testset "english" begin
        input = " The 1Julia programming language .is,2 fun"
        output = "The Julia program languag is fun"
        test(input, output, Languages.English())
    end

    @testset "german" begin
        input = "Die Straßen sind leer, die 10 Blätter sind grün und die Röte fehlt. "
        output = "Die Strass sind leer die Blatt sind grun und die Rot fehlt"
        test(input, output, Languages.German())
    end
end

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
