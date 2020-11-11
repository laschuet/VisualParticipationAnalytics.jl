module ParticipationAnalytics

using Clustering
using Distances
using JSON
using SparseArrays
using TextAnalysis

export clustering,
        preprocess,
        preprocess!,
        similarities,
        slatetext,
        topicmodel,
        topkwords

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

""""""
function preprocess!(entity::Union{AbstractDocument,Corpus})
    #=
        (November 11th, 2020)
        The following flags for `prepare!` are available:
        - strip_corrupt_utf8
        - strip_case
        - strip_articles
        - strip_indefinite_articles
        - strip_definite_articles
        - strip_preposition
        - strip_pronouns
        - strip_non_letters
        - strip_numbers
        - strip_punctuation
        - strip_whitespace
        - strip_frequent_terms
        - strip_sparse_terms
        - strip_stopwords
    =#
    prepare!(entity, strip_corrupt_utf8)
    remove_patterns!(entity, r"[^a-zA-ZäöüÄÖÜß\s]")
    prepare!(entity, strip_whitespace)
    stem!(entity)
end

""""""
function preprocess(txt::AbstractString)
    doc = StringDocument(txt)
    preprocess!(doc)
    return text(doc)
end

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

end # module ParticipationAnalytics
