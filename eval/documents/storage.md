# Storage engine

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
