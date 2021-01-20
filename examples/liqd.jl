using VisualParticipationAnalytics
using Clustering
using Distances
using NearestNeighbors
using PGFPlotsX
using Random
using SQLite
using Statistics
using Tables

Random.seed!(1)

empty!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"""
\usepgfplotslibrary{colorbrewer}
%\usepgfplotslibrary{colormap}
\usepackage{libertine}
\usepackage{unicode-math}
\setmathfont[Scale=MatchUppercase]{libertinusmath-regular.otf}
\pgfplotsset{
    colormap/Paired-12,
    cycle list/Paired-12,
    cycle multiindex* list={
        mark list*\nextlist
        Paired-12\nextlist
    }
}
""")

const CONFIG = Dict("out_dir" => "out")

function assignmentplot(assignments, x, y, xlabel, ylabel)
    t = @pgf Table({ meta = "cluster" }, x=x, y=y, cluster=assignments)
    p = @pgf Plot({ scatter, "only marks", scatter_src = "explicit", mark_size = "1pt" }, t)
    return @pgf Axis({ xlabel = xlabel, ylabel = ylabel }, p)
end

function clusterkmeans(data, ks, dist)
    clusterings = []
    Js = []
    silhouette_coefficients = []

    distances = pairwise(dist, data')
    for k in ks
        clustering = kmeans(data', k, distance=dist)
        push!(clusterings, clustering)
        push!(Js, clustering.totalcost)
        push!(silhouette_coefficients, mean(silhouettes(clustering, distances)))
    end

    ## Assignment plots
    for k in ks
        plt = assignmentplot(assignments(clusterings[k - 1]), data[:, 1],data[:, 2], "longitude", "latitude")
        pgfsave(CONFIG["out_dir"] * "/kmeans_assignments_k_" * (k < 10 ? "0$k" : "$k") * ".pdf", plt)
    end

    ## Elbow method plot
    c = @pgf Coordinates(zip(ks, Js))
    p = @pgf Plot({ color = "red", mark = "x" }, c)
    plt = @pgf Axis({ xlabel = raw"\(k\)", ylabel = raw"\(J\)" }, p)
    pgfsave(CONFIG["out_dir"] * "/kmeans_elbow.pdf", plt)

    ## Silhouette coefficient plot
    c = @pgf Coordinates(zip(ks, silhouette_coefficients))
    p = @pgf Plot({ color = "red", mark = "x" }, c)
    plt = @pgf Axis({ xlabel = raw"\(k\)", ylabel = "silhouette coefficient" }, p)
    pgfsave(CONFIG["out_dir"] * "/kmeans_silhouette.pdf", plt)
end

function clusterdbscan(data, dist)
    minpts = 5

    tree = BallTree(data', dist)
    _, knn_distances = knn(tree, data', minpts + 1, true)
    avg_knn_distances = mean.(knn_distances)
    sort!(avg_knn_distances)

    c = @pgf Coordinates(zip(1:length(avg_knn_distances), avg_knn_distances))
    p = @pgf Plot({ color = "red" }, c)
    plt = @pgf Axis({ xlabel = "instance", ylabel = "$minpts-nn distance" }, p)
    pgfsave(CONFIG["out_dir"] * "/dbscan_eps.pdf", plt)

    clusterings = []

    distances = pairwise(dist, data')
    clustering = dbscan(distances, 0.7, minpts)
    push!(clusterings, clustering)

    ## Assignment plots
    plt = assignmentplot(assignments(clusterings[1]), data[:, 1], data[:, 2], "longitude", "latitude")
    pgfsave(CONFIG["out_dir"] * "/dbscan_assignments_eps_0.7_min_pts_$minpts.pdf", plt)
end

function main(dbpath, tablename)
    MEAN_EARTH_RADIUS = 6371
    earth_haversine = Haversine(MEAN_EARTH_RADIUS)
    !isdir(CONFIG["out_dir"]) && mkdir(CONFIG["out_dir"])

    # Load data
    db = SQLite.DB(dbpath)
    table = DBInterface.execute(db, """
        SELECT * FROM $tablename;
    """) |> Tables.columntable

    # Pre-process data
    # Text column title
    # Text column content

    # Cluster data
    longitude = table[Symbol("long")]
    latitude = table[Symbol("lat")]
    data = [longitude latitude]
    # display(data)

    clusterkmeans(data, 2:12, earth_haversine)
    clusterdbscan(data, earth_haversine)
end

main("~/datasets/participation/liqd_laermorte_melden.sqlite", "contribution")
