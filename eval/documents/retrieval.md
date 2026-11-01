# Retrieval behaviour

## Ranking

Candidates are ranked by descending score and ties are broken by ascending item
index, so a query over an unchanged index returns a deterministic ordering.
Scores that are not finite are discarded rather than ranked, because a NaN
would otherwise make the ordering depend on the comparison implementation.

## Per-scale quotas

An index can mix chunk scales, and a quota limits how many results any one
scale contributes. The quota is applied after global ranking, which means the
returned set can be smaller than the requested top-k. It is not a re-ranking
step: it only removes surplus entries of an over-represented scale.

Rank within a scale is reported separately from the global rank, and remains
exact for every result returned under a quota.

## Chunking

Chunks are cut on UTF-8 code point boundaries so a multi-byte character is
never split. A chunk window prefers the last whitespace inside the second half
of the window, so the effective chunk size can be smaller than the configured
maximum. The overlap must be smaller than the chunk size, and a short chunk
carries a reduced overlap rather than being padded.

## Failure handling

If no candidate is retrieved the caller is expected to decline the request
instead of generating an answer without evidence. Trimming the context to fit a
generation window drops whole chunks from the least relevant end, and the
number of chunks that survived is reported to the caller.
