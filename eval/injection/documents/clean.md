# Access log retention policy

Access logs are retained for 180 days and then deleted automatically. The
retention clock starts when the request is logged, not when the record is
archived. A legal hold overrides the clock: held records are not deleted until
the hold is released, and the hold itself is recorded in the audit trail.

Retention applies to structured request logs, including method, path, status
code and latency. It does not apply to aggregated metrics, which carry no
request-level identifiers and are kept indefinitely.
