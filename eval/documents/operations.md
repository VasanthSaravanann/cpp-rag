# Operating the service

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
