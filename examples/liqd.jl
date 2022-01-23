using VisualParticipationAnalytics
using Clustering
using DataFrames
using Distances
using Languages
using LinearAlgebra
using MLJ
using MultivariateStats
using NearestNeighbors
using PGFPlotsX
using Random
using SQLite
using Statistics
using StatsBase
using TextAnalysis

Random.seed!(1)

CONFIG = Dict(:out_dir=>"out", :do_lsa=>false, :do_lda=>false, :do_cluster=>true)

"""
"""
function initplots()
    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"""
        \usepgfplotslibrary{colorbrewer}
        \pgfplotsset{
            colormap/Paired-12,
            cycle list/Paired-12,
            cycle multiindex* list={
                mark list*\nextlist
                Paired-12\nextlist
            }
        }
        \usepackage{libertine}
        \usepackage{unicode-math}
        \setmathfont[Scale=MatchUppercase]{libertinusmath-regular.otf}
        \newcommand{\mlv}[1]{\mathit{#1}}
    """)
end

"""
"""
function creategraph(dbpath, tablename)
    # Load data
    db = SQLite.DB(dbpath)
    df = DBInterface.execute(db, """
        SELECT * FROM $tablename;
    """) |> DataFrame

    for i = 1:5
        println(df[i])
    end
end

"""
"""
function preparedata(dbpath, tablename)
    # Load data
    df = readdb(dbpath, tablename)
    display(describe(df))

    # Pre-process data
    # Textual data
    lang = Languages.German()
    authors = df[:, ["author"]]
    #categorical!(authors)
    #ohe = OneHotEncoder()
    #mach = MLJ.fit!(machine(ohe, authors))
    #authors_ohe = MLJ.transform(mach, authors)
    #df = [df authors_ohe]

    titles = df[:, "title"]
    titles = preprocess.(titles, [lang])
    tfidf_titles = tf_idf(titles)
    crps_titles = Corpus(StringDocument.(titles))
    update_lexicon!(crps_titles)
    dtm_titles = DocumentTermMatrix(crps_titles)

    contents = df[:, "content"]
    contents = preprocess.(contents, [lang])
    tfidf_contents = tf_idf(contents)
    crps_contents = Corpus(StringDocument.(contents))
    update_lexicon!(crps_contents)
    dtm_contents = DocumentTermMatrix(crps_contents)

    # Temporal data
    # created_at

    # Other data
    #
    #category = df[:, ["category"]]
    #categorical!(category)
    #ohe = OneHotEncoder()
    #mach = MLJ.fit!(machine(ohe, category))
    #category_ohe = MLJ.transform(mach, category)
    #df = [df category_ohe]

    return df, crps_titles, crps_contents, dtm_titles, dtm_contents, tfidf_titles, tfidf_contents
end

"""
"""
function assignmentplot(assignments, x, y, xlabel, ylabel)
    t = @pgf PGFPlotsX.Table({ meta="cluster" }, x=x, y=y, cluster=assignments)
    p = @pgf Plot({
        scatter,
        "only marks",
        scatter_src="explicit",
        mark_size="1pt"
    }, t)
    return @pgf Axis({ xlabel=xlabel, ylabel=ylabel }, p)
end

"""
"""
function clusterkmeans(data, ks, dist, x, y, name)
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
        plt = assignmentplot(assignments(clusterings[k - 1]), x, y, "longitude",
                "latitude")
        pgfsave(CONFIG[:out_dir] * "/kmeans_$(name)_assignments_k_" * (k < 10 ? "0$k" : "$k") * ".pdf", plt)
    end

    ## Elbow method plot
    c = @pgf Coordinates(zip(ks, Js))
    p = @pgf Plot({ color="red", mark="x" }, c)
    plt = @pgf Axis({ xlabel=raw"\(k\)", ylabel=raw"\(J\)" }, p)
    pgfsave(CONFIG[:out_dir] * "/kmeans_$(name)_elbow.pdf", plt)

    ## Silhouette coefficient plot
    c = @pgf Coordinates(zip(ks, silhouette_coefficients))
    p = @pgf Plot({ color="red", mark="x" }, c)
    plt = @pgf Axis({ xlabel=raw"\(k\)", ylabel="silhouette coefficient" }, p)
    pgfsave(CONFIG[:out_dir] * "/kmeans_$(name)_silhouette.pdf", plt)
end

"""
"""
function clusterdbscan(data, minpoints, epsilons, dist, optimize, x, y, name)
    eps = 0.5825
    #silhouette_coefficients = []
    for minpts in minpoints
        if optimize
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
            p = @pgf Plot({ color="red" }, c)
            plt = @pgf Axis({ xlabel="instance", ylabel="$minpts-nn distance" }, p,
                    HLine({ dotted }, eps))
            pgfsave(CONFIG[:out_dir] * "/dbscan_$(name)_eps_min_pts_" * (minpts < 10 ? "0" : "") * "$minpts.pdf", plt)
        end

        distances = pairwise(dist, data')
        for ϵ in epsilons
            clustering = dbscan(distances, ϵ, minpts)
            assigns = assignments(clustering)
            #filter!(!iszero, assigns)
            assigns = assigns .+ 1
            #push!(silhouette_coefficients, mean(silhouettes(assigns, distances)))

            save(clustering, CONFIG[:out_dir] * "/" * (minpts < 10 ? "0" : "") * "$(minpts)_$(ϵ).json")

            ## Assignment plots
            plt = assignmentplot(assignments(clustering), x, y, "longitude", "latitude")
            pgfsave(CONFIG[:out_dir] * "/dbscan_$(name)_assignments_min_pts_" * (minpts < 10 ? "0" : "")
                    * "$(minpts)" * "_eps_" * string(floor(ϵ, digits=3)) * ".pdf" , plt)
        end
    end
end

"""
"""
function process(dbpath, tablename)
    CONFIG[:out_dir] = CONFIG[:out_dir] * "/" * split(dbpath, "/")[end]
    !ispath(CONFIG[:out_dir]) && mkpath(CONFIG[:out_dir])

    df, crps_titles, crps_contents, dtm_titles, dtm_contents, tfidf_titles, tfidf_contents = preparedata(dbpath, tablename)
    display(dtm_titles)
    display(dtm_contents)
    display(Array(tfidf_titles))
    display(Array(tfidf_contents))

    #M = fit(PCA, Array(tfidf_contents)'; maxoutdim=1000)
    #tfidf_contents = MultivariateStats.transform(M, tfidf_contents')'

    if CONFIG[:do_lsa]
        lsa(tfidf_titles)
        lsa(tfidf_contents)
        M = fit(UnitRangeTransform, tfidf_contents, dims=2)
        tfidf_contents = StatsBase.transform(M, tfidf_contents)
    end
    if CONFIG[:do_lda]
        lda_titles = lda(dtm_titles, 4, 1000, 0.1, 0.1)
        lda_contents = lda(dtm_contents, 4, 1000, 0.1, 0.1)
        for i = 1:4
            println("TOPIC $i")
            display(topkwords(lda_titles[1], i, crps_titles, 10))
            display(topkwords(lda_contents[1], i, crps_contents, 10))
        end
    end

    # Cluster data
    if CONFIG[:do_cluster]
        longitude = df[:, "long"]
        latitude = df[:, "lat"]
        clusterdbscan([longitude latitude], 2:12, 5:5:200, Haversine(), false, longitude, latitude, "long_lat")
        #clusterkmeans(tfidf_contents, 2:15, Euclidean(), longitude, latitude, "tfidf_contents_euclidean")
        #clusterkmeans(tfidf_contents, 2:15, CosineDist(), longitude, latitude, "tfidf_contents_cosine")
        #clusterkmedoids(tfidf_contents, 2:12, CosineDist(), longitude, latitude, "tfidf_contents_cosine")
    end
end

"""
"""
function main()
    initplots()
    #process("~/Datasets/participation/databases/liqd_laermorte_melden.sqlite", "contribution")
    process("~/Datasets/participation/databases/liqd_mauerpark.sqlite", "contribution")
    #process("~/Datasets/participation/databases/liqd_blankenburger_sueden.sqlite", "comment_a")
    #process("~/Datasets/participation/databases/liqd_blankenburger_sueden.sqlite", "comment_b")
    #process("~/Datasets/participation/databases/liqd_blankenburger_sueden.sqlite", "comment_c")
end

"""
"""
function distances()
    df = readdb("~/Datasets/participation/databases/liqd_mauerpark.sqlite", "contribution")
    longitude = df[:, "long"]
    latitude = df[:, "lat"]
    distances = pairwise(Haversine(), [longitude latitude]')
    save(distances, CONFIG[:out_dir] * "/long_lat_haversine.json")
end
