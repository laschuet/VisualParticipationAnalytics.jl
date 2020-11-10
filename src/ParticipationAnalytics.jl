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
    prepare!(entity, strip_corrupt_utf8)
    prepare!(entity, strip_whitespace)
    prepare!(entity, strip_non_letters)
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
    if typeof(m) == SparseMatrixCSC{Int,Int}
        m = convert(SparseMatrixCSC{Float64,Int}, m)
    end
    return m
end

""""""
function similarities(crps::Corpus, tfidf=true)
    dt = docterm(crps, tfidf)
    return sparse(1 .- pairwise(CosineDist(), Matrix(dt')))
end

""""""
function clustering(crps::Corpus, k::Int64, tfidf=true)
    dt = docterm(crps, tfidf)
    return kmeans(Matrix(dt'), k; init=:kmpp)
end

""""""
function topicmodel(crps::Corpus, k::Int64, iter::Int64, alpha::Float64,
                beta::Float64)
    update_lexicon!(crps)
    return lda(DocumentTermMatrix(crps), k, iter, alpha, beta)
end

""""""
function topkwords(topicword::SparseMatrixCSC{Float64,Int64}, i::Int64,
                    crps::Corpus, k::Int64=10)
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
