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

function assignmentplot(assignments, x, y, xlabel, ylabel)
    t = @pgf Table({ meta = "cluster" }, x=x, y=y, cluster=assignments)
    p = @pgf Plot({ scatter, "only marks", scatter_src = "explicit", mark_size = "1pt" }, t)
    return @pgf Axis({ xlabel = xlabel, ylabel = ylabel }, p)
end

function assignmentplot(assignments, x, y, xlabel, ylabel)
    return @pgf Axis(
        {
            xlabel = xlabel,
            ylabel = ylabel,
        },
        Plot(
            {
                scatter,
                "only marks",
                scatter_src = "explicit",
                mark_size = "1pt"
            },
            Table(
                {
                    meta = "cluster",
                },
                x=x,
                y=y,
                cluster=assignments
            )
        )
    )
end

function main(dbpath, tablename)
    MEAN_EARTH_RADIUS = 6371
    earth_haversine = Haversine(MEAN_EARTH_RADIUS)
    OUT_PATH = "out"
    !isdir(OUT_PATH) && mkdir(OUT_PATH)

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

    # k-means
    k_range = 2:2
    distances = pairwise(earth_haversine, data')
    clusterings = []
    js = []
    silhouette_coefficients = []
    for k in k_range
        clustering = kmeans(data', k, distance=earth_haversine)
        push!(clusterings, clustering)

        push!(js, clustering.totalcost)

        mean_silhouette = mean(silhouettes(clustering, distances))
        push!(silhouette_coefficients, mean_silhouette)
    end

    # Evaluate k-means
    ## Assignment plots
    for k in k_range
        plt = assignmentplot(assignments(clusterings[k - 1]), longitude, latitude, "longitude", "latitude")
        pgfsave("$OUT_PATH/kmeans_assignments_k_" * (k < 10 ? "0$k" : "$k") * ".pdf", plt)
    end
    ## Elbow method
    plt = @pgf Axis({
        xlabel = raw"\(k\)",
        ylabel = raw"\(J\)",
    }, Plot({
        color = "blue",
        mark = "x",
    }, Coordinates(zip(k_range, js))))
    pgfsave("$OUT_PATH/kmeans_elbow.pdf", plt)
    ## Silhouette coefficient
    plt = @pgf Axis({
        xlabel = raw"\(k\)",
        ylabel = "silhouette coefficient",
    }, Plot({
        color = "red",
        mark = "x",
    }, Coordinates(zip(k_range, silhouette_coefficients))))
    pgfsave("$OUT_PATH/kmeans_silhouette.pdf", plt)

    # DBSCAN
    distances = pairwise(earth_haversine, data')
    tree = BallTree(data', earth_haversine)
    k = 5
    _, knn_distances = knn(tree, data', k + 1, true)
    avg_knn_distances = mean.(knn_distances)
    sort!(avg_knn_distances)
    plt = @pgf Axis({
        xlabel = "instance",
        ylabel = "$k-nn distance",
    }, Plot({
        color = "green"
    }, Coordinates(zip(1:length(avg_knn_distances), avg_knn_distances))))
    pgfsave("$OUT_PATH/dbscan_eps.pdf", plt)

    clusterings = []
    clustering = dbscan(distances, 0.7, k)
    push!(clusterings, clustering)

    # Evaluate DBSCAN
    ## Assignment plots
    plt = assignmentplot(assignments(clusterings[1]), longitude, latitude, "longitude", "latitude")
    pgfsave("$OUT_PATH/dbscan_assignments_eps_0.7_min_pts_5.pdf", plt)
end

main("~/datasets/participation/liqd_laermorte_melden.sqlite", "contribution")
