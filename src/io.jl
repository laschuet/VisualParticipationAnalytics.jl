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
function save(clustering::DbscanResult, filename::AbstractString)
    y = assignments(clustering) .+ 1
    k = nclusters(clustering) + 1 # unassigned instances / noise
    ns = counts(clustering)

    clusters = Vector{Dict}(undef, k)
    clusters[1] = Dict("id" => 1, "n" => length(y) - sum(ns), "J" => 0.0)
    for i = 2:k
        clusters[i] = Dict("id" => i, "n"  => ns[i - 1] , "J" => 0.0)
    end

    enumeratedy = enumerate(y)
    assigns = map(collect, enumeratedy)

    invassigns = Vector{Vector{Union{Int, Vector{Int}}}}(undef, k)
    for i = 1:k
        invassigns[i] = [i, Int[]]
    end
    for (i, yi) in enumeratedy
        push!(invassigns[yi][2], i)
    end

    result = Dict("clusters" => clusters, "assignments" => assigns, "invAssignments" => invassigns)

    open(filename, "w") do io
        JSON3.write(io, result)
    end
    open("pretty_" * filename, "w") do io
        JSON3.pretty(io, result)
    end
end
