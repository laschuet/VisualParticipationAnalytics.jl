@testset "preprocessing" begin
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
        @test preprocess(input, lang) == output
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
