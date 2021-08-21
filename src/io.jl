""""""
function corpus(db::SQLite.DB, tablename::AbstractString,
                colname::AbstractString)
    table = DBInterface.execute(db, """
        SELECT $colname FROM $tablename;
    """) |> Tables.columntable
    rows = table[Symbol(colname)]
    return Corpus(StringDocument.(rows))
end
