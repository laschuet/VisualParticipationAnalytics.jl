@testset "io" begin
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

    @testset "corpus" begin
        db = SQLite.DB()
        tablename = "test"
        colname = "text"
        schema = Tables.Schema([colname], [String])
        SQLite.createtable!(db, tablename, schema)
        DBInterface.execute(db, """
            INSERT INTO $tablename ($colname) VALUES ("a"), ("b");
        """)

        crps = corpus(db, tablename, colname)
        @test isa(crps, Corpus)
        @test text(crps[1]) == "a" && text(crps[2]) == "b"
    end
end
