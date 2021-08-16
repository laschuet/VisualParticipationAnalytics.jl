""""""
function preprocess(txt::AbstractString, lang::Language)
    txt = replace(txt, r"[.:,;?!()\[\]=*/+-]" => " ")
    tokens = tokenize(lang, txt)
    # TODO Stop word removal
    stemmer = Stemmer(isocode(lang))
    tokens = stem(stemmer, tokens)
    return join(tokens, " ")
end
preprocess(txt::AbstractString) = preprocess(txt, LanguageDetector()(txt)[1])

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
