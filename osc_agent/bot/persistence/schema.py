"""定义 Bot SQLite Schema 及其兼容版本。"""

SCHEMA_EPOCH = 4
SCHEMA_MIGRATION_VERSION = 3
STATE_MODEL_REVISION = "bot-job-v3-session-v6"
APPLICATION_VERSION = "0.3.4"


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    schema_epoch INTEGER NOT NULL,
    state_model_revision TEXT NOT NULL,
    application_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    delivery_id TEXT PRIMARY KEY,
    event_name TEXT NOT NULL,
    received_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bot_jobs (
    job_id TEXT PRIMARY KEY,
    repository_id INTEGER NOT NULL,
    repository_full_name TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    lease_owner TEXT,
    lease_until TEXT,
    job_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_bot_jobs_active_issue
ON bot_jobs(repository_id, issue_number)
WHERE status IN ('queued_plan','running_plan','blocked_plan','waiting_approval',
 'queued_implementation','running_implementation','ready_to_publish','publishing','retry_wait');
CREATE INDEX IF NOT EXISTS idx_bot_jobs_claim ON bot_jobs(status, lease_until, created_at);
CREATE INDEX IF NOT EXISTS idx_bot_jobs_repo ON bot_jobs(repository_full_name, status);
CREATE TABLE IF NOT EXISTS bot_approvals (
    approval_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES bot_jobs(job_id),
    approval_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bot_artifacts (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES bot_jobs(job_id),
    kind TEXT NOT NULL,
    artifact_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bot_job_inputs (
    job_id TEXT PRIMARY KEY REFERENCES bot_jobs(job_id),
    input_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS execution_contracts (
    contract_hash TEXT PRIMARY KEY,
    contract_version INTEGER NOT NULL,
    contract_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bot_inbox_messages (
    source_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES bot_jobs(job_id),
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind='plan_reply'),
    body TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','consumed')),
    created_at TEXT NOT NULL,
    consumed_at TEXT
);
CREATE TABLE IF NOT EXISTS outbox_events (
    event_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES bot_jobs(job_id),
    kind TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    next_attempt_at TEXT NOT NULL,
    event_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox_events(status, next_attempt_at);
CREATE TABLE IF NOT EXISTS job_events (
    event_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES bot_jobs(job_id),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_records (
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    previous_event_id TEXT,
    record_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(session_id, sequence)
);
CREATE TABLE IF NOT EXISTS session_leases (
    session_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    lease_until TEXT NOT NULL
);
"""
