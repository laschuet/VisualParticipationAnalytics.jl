module VisualParticipationAnalytics

using Clustering
using Distances
using JSON3
using Languages
using LinearAlgebra
using SparseArrays
using SQLite
using Tables
using TextAnalysis

export clustering,
        corpus,
        preprocess,
        save,
        similarities,
        slatetext,
        topicmodel,
        topkwords

include("io.jl")
include("preprocessing.jl")

""""""
function docterm(crps::Corpus, tfidf::Bool)
    update_lexicon!(crps)
    dt = DocumentTermMatrix(crps)
    m = dtm(dt)
    if tfidf
        m = tf_idf(m)
    end
    return m
end

""""""
function similarities(crps::Corpus, tfidf=true)
    dt = docterm(crps, tfidf)
    return sparse(1 .- pairwise(CosineDist(), Matrix(dt'), dims=2))
end

""""""
function clustering(crps::Corpus, k::Int, tfidf=true)
    dt = docterm(crps, tfidf)
    return kmeans(Matrix(dt'), k; init=:kmpp)
end

""""""
function topicmodel(crps::Corpus, k::Int, iter::Int, alpha::Float64,
                beta::Float64)
    update_lexicon!(crps)
    return lda(DocumentTermMatrix(crps), k, iter, alpha, beta)
end

""""""
function topkwords(topicword::SparseMatrixCSC{Float64,Int}, i::Int,
                    crps::Corpus, k::Int=10)
    terms = collect(keys(lexicon(crps)))
    topicword = permutedims(topicword)
    rows = rowvals(topicword)
    vals = nonzeros(topicword)
    wordprobs = Array{Tuple{String,Float64},1}()
    for j in nzrange(topicword, i)
        push!(wordprobs, (terms[rows[j]], vals[j]))
    end
    wordprobs = sort(wordprobs; by=w -> w[2], rev=true)
    n = k < length(wordprobs) ? k : length(wordprobs)
    return wordprobs[1:n]
end

end # module VisualParticipationAnalytics
