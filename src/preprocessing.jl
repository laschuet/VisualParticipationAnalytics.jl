""""""
preprocess(txt::AbstractString) = preprocess(txt, LanguageDetector()(txt)[1])
function preprocess(txt::AbstractString, lang::Language)
    doc = StringDocument(txt)
    language!(doc, lang)
    preprocess!(doc)
    return text(doc)
end

""""""
function preprocess!(entity::Union{AbstractDocument,Corpus})
    prepare!(entity, strip_corrupt_utf8)
    remove_patterns!(entity, r"[^a-zA-ZäöüÄÖÜß\s]")
    prepare!(entity, strip_whitespace)
    stem!(entity)
end
