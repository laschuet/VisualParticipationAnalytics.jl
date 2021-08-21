@testset "io" begin
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
