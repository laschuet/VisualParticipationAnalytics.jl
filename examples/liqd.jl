using VisualParticipationAnalytics
using Clustering
using Distances
using LinearAlgebra
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
    for minpts = 2:12
        tree = BallTree(data', dist)
        _, knn_distances = knn(tree, data', minpts + 1, true)
        mean_knn_distances = mean.(knn_distances)
        sort!(mean_knn_distances)

        n = length(mean_knn_distances)
        ds = []
        first_to_last = [n, mean_knn_distances[n]] - [1, mean_knn_distances[1]]
        normalize!(first_to_last)
        for i = 1:n
            first_to_i = [i, mean_knn_distances[i]] - [1, mean_knn_distances[1]]
            d = norm(first_to_i - dot(first_to_i, first_to_last) * first_to_last)
            push!(ds, d)
        end
        eps = mean_knn_distances[argmax(ds)]

        c = @pgf Coordinates(zip(1:n, mean_knn_distances))
        p = @pgf Plot({ color = "red" }, c)
        plt = @pgf Axis({ xlabel = "instance", ylabel = "$minpts-nn distance" }, p,
                HLine({ dotted }, eps))
        pgfsave(CONFIG["out_dir"] * "/dbscan_eps_min_pts_$minpts.pdf", plt)

        clusterings = []

        distances = pairwise(dist, data')
        clustering = dbscan(distances, eps, minpts)
        push!(clusterings, clustering)

        ## Assignment plots
        plt = assignmentplot(assignments(clusterings[1]), data[:, 1], data[:, 2], "longitude", "latitude")
        pgfsave(CONFIG["out_dir"] * "/dbscan_assignments_eps_" * string(floor(eps, digits=4)) * "_min_pts_$minpts.pdf", plt)
    end
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

initplots()
main("~/datasets/participation/liqd_laermorte_melden.sqlite", "contribution")
