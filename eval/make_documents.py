#!/usr/bin/env python3
"""Regenerate the eval fixture corpus deterministically.

These documents are synthetic fixtures for exercising the harness. They are not
real reference material and the cases derived from them are not a gold set.
"""
from pathlib import Path

DOCS = {
    "storage.md": """# Storage engine

## Write path

Appending a record goes through three stages. First the record is serialised
into a length-prefixed frame and appended to the write-ahead log. Second the
frame is acknowledged once the log write completes, before any index update is
visible. Third a background flusher folds acknowledged frames into the main
segment file, which is the only structure a reader consults.

The write-ahead log is the source of truth for durability. An acknowledged
append survives a crash even when the flusher never ran, because recovery
replays every acknowledged frame that is not yet present in the segment file.
The flusher is therefore an optimisation, never a correctness requirement.

## Crash recovery

Recovery starts by locating the newest segment snapshot and then replaying the
log from the sequence number recorded in that snapshot. A frame whose checksum
does not match terminates the replay at that point and the partial frame is
discarded. A torn final frame is expected after a power loss and is not
reported as corruption.

Snapshot and log sequence numbers are monotonic 64-bit integers. The sequence
number of a snapshot is inclusive: replay begins at the next number.

## Segment files

A segment file holds a fixed-size header, a record array, and a trailer
checksum over both. Records inside a segment are immutable; an update appends a
newer version to the log rather than modifying a segment in place. Compaction
later drops versions that are shadowed by a newer one.

Readers never scan the log. They map the segment file read-only and validate
the header and the trailer checksum before trusting any record offset.

## Limits

A single segment is capped at 64 MiB of record data. When the cap is reached
the flusher rolls to a new segment and the snapshot advances to cover the old
one. The record size is limited to 1 MiB so a frame always fits a segment.
""",
    "retrieval.md": """# Retrieval behaviour

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
""",
    "operations.md": """# Operating the service

## Backups

A backup is a copy of the newest segment snapshot together with the log frames
that follow it. Restoring replays those frames onto the snapshot. A backup is
only considered complete once its manifest records the snapshot sequence
number and the count of frames copied.

## Capacity

The useful signal for capacity planning is the rate at which the flusher falls
behind, expressed as unflushed acknowledged frames. A steady backlog that does
not drain means the flusher cannot keep up, and the remedy is more flush
capacity rather than a larger log.

## Rolling upgrades

Upgrades are rolled out one process at a time while the segment format stays
unchanged. A format change requires stopping writers, flushing the log, and
only then starting the new version, because an older reader would misinterpret
a newer record layout.

## Health checks

A process is healthy when it can map its segment file and its replay cursor is
advancing. A stalled replay cursor is an unhealthy state even when requests
are still being served, because acknowledged appends are not becoming visible.
""",
}

# Mechanical question/expected-substring pairs, answered from the fixture text
# above. Unverified scaffolding; see eval/README.md.
CASES = [
    ("what is the source of truth for durability", "storage.md", "write-ahead log", False),
    ("when is an appended record acknowledged", "storage.md", "log write completes", False),
    ("what terminates replay during recovery", "storage.md", "checksum", False),
    ("which structure does a reader consult", "storage.md", "segment file", False),
    ("what happens to a torn final frame", "storage.md", "discarded", False),
    ("what is the record data cap of a single segment", "storage.md", "64 MiB", False),
    ("how large may one record be", "storage.md", "1 MiB", False),
    ("does an update modify a segment in place", "storage.md", "immutable", False),
    ("how are ties in score ordered", "retrieval.md", "ascending item index", False),
    ("what happens to scores that are not finite", "retrieval.md", "discarded", False),
    ("can a per-scale quota return fewer results than requested", "retrieval.md", "smaller than", False),
    ("is a per-scale quota a re-ranking step", "retrieval.md", "not a re-ranking", False),
    ("on what boundaries are chunks cut", "retrieval.md", "UTF-8", False),
    ("where does a chunk window prefer to break", "retrieval.md", "second half", False),
    ("what should happen when no candidate is retrieved", "retrieval.md", "decline", False),
    ("which chunks are dropped when trimming context", "retrieval.md", "least relevant", False),
    ("what does a backup manifest record", "operations.md", "sequence", False),
    ("what signal is used for capacity planning", "operations.md", "unflushed", False),
    ("what is the remedy when the flusher cannot keep up", "operations.md", "flush capacity", False),
    ("what must happen before a segment format change", "operations.md", "stopping writers", False),
    ("when is a process considered healthy", "operations.md", "cursor is", False),
    ("what is the price of a licence", "operations.md", "not available", True),
]


def main() -> None:
    import json

    here = Path(__file__).resolve().parent
    docs = here / "documents"
    docs.mkdir(exist_ok=True)
    for name, text in DOCS.items():
        (docs / name).write_text(text, encoding="utf-8")
    cases = {
        "provenance": (
            "Mechanically drafted from generated fixture documents. NOT a human-labelled "
            "gold set: no adjudication, no ambiguity review, no approved thresholds. "
            "Use as a regression signal for this harness only."
        ),
        "documents": [str(docs / name) for name in DOCS],
        "cases": [
            {"id": f"{doc.split('.')[0]}-{i:02d}", "document": doc, "question": q,
             "expect_substring": sub, "no_evidence": abstain}
            for i, (q, doc, sub, abstain) in enumerate(CASES, 1)
        ],
    }
    (here / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(DOCS)} documents and {len(CASES)} cases")


if __name__ == "__main__":
    main()
