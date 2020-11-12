""""""
function corpus(db::SQLite.DB, tablename::AbstractString,
                colname::AbstractString)
    table = DBInterface.execute(db, """
        SELECT $colname FROM $tablename;
    """) |> Tables.columntable
    rows = table[Symbol(colname)]
    return Corpus(StringDocument.(rows))
end

""""""
function slatetext(nodes::AbstractArray)
    text = ""
    for node in nodes
        if node["kind"] == "text"
            leaves = get(node, "leaves", [])
            text *= mapreduce(leaf -> get(leaf, "text", ""), *, leaves; init="")
            text *= " "
        else
            text *= slatetext(get(node, "nodes", []))
        end
    end
    return strip(text)
end

""""""
function slatetext(state::Dict)
    document = get(state, "document", Dict())
    nodes = get(document, "nodes", [])
    return slatetext(nodes)
end

""""""
function slatetext(content::AbstractString)
    json = JSON.parse(content)
    return slatetext(json)
end
