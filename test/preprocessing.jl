@testset "preprocessing" begin
    @testset "english" begin
        input = "? [The] Julia ; (programming) language .is,: fun!"
        output = "The Julia program languag is fun"

        @test preprocess(input, Languages.English()) == output
        @test preprocess(input) == output
    end

    @testset "german" begin
        input = "Die Straßen sind leer, die 10 Blätter sind grün und die Röte fehlt. "
        output = "Die Strass sind leer die 10 Blatt sind grun und die Rot fehlt"

        @test preprocess(input, Languages.German()) == output
        @test preprocess(input) == output
    end
end
