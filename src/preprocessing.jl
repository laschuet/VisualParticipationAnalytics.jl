""""""
function preprocess(txt::AbstractString, lang::Language; dostem=true)
    # We replace by " " to handle cases such as "end of sentence.begin of next sentence"
    txt = replace(txt, r"[.:,;?!()\[\]\\=*/+-]" => " ")
    tokens = tokenize(lang, txt)
    # TODO Stop word removal
    if dostem
        stemmer = Stemmer(isocode(lang))
        tokens = stem(stemmer, tokens)
    end
    return join(tokens, " ")
end
preprocess(txt::AbstractString; kwargs...) = preprocess(txt, LanguageDetector()(txt)[1]; kwargs...)

""""""
function TextAnalysis.tf_idf(crps::Corpus)
    update_lexicon!(crps)
    dtm = DocumentTermMatrix(crps)
    return tf_idf(dtm)
end
TextAnalysis.tf_idf(txts::Vector{<:AbstractString}) =
    tf_idf((Corpus(StringDocument.(txts))))

""""""
TextAnalysis.lsa(M::SparseMatrixCSC) = svd(Array(M))
