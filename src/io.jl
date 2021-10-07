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
function readdb(filename::AbstractString, tablename::AbstractString)
    db = SQLite.DB(filename)
    df = DBInterface.execute(db, """
        SELECT * FROM $tablename;
    """) |> DataFrame
    return df
end

""""""
function save(M::AbstractMatrix, filename::AbstractString)
    result = Dict(enumerate(eachcol(M)))

    open(filename, "w") do io
        JSON3.write(io, result)
    end
    open(filename * ".pretty", "w") do io
        JSON3.pretty(io, result)
    end
end

""""""
function save(clustering::DbscanResult, filename::AbstractString)
    y = assignments(clustering) .+ 1
    k = nclusters(clustering) + 1 # unassigned instances / noise
    ns = counts(clustering)

    enumeratedy = enumerate(y)
    assigns = map(collect, enumeratedy)

    invassigns = Vector{Vector{Union{Int, Vector{Int}}}}(undef, k)
    for i = 1:k
        invassigns[i] = [i, Int[]]
    end
    for (i, yi) in enumeratedy
        push!(invassigns[yi][2], i)
    end

    clusters = Vector{Dict}(undef, k)
    proto = length(invassigns[1][2]) == 0 ? 0 : invassigns[1][2][end]
    clusters[1] = Dict("id" => 1, "n" => length(y) - sum(ns), "J" => 0.0, "proto" => proto)
    for i = 2:k
        proto = length(invassigns[i][2]) == 0 ? 0 : invassigns[i][2][end]
        clusters[i] = Dict("id" => i, "n"  => ns[i - 1] , "J" => 0.0, "proto" => proto)
    end

    result = Dict("clusters" => clusters, "assignments" => assigns, "invAssignments" => invassigns)

    open(filename, "w") do io
        JSON3.write(io, result)
    end
    open(filename * ".pretty", "w") do io
        JSON3.pretty(io, result)
    end
end
