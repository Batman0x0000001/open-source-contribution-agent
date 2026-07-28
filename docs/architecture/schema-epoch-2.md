# Schema Epoch 2

Epoch 2 is a stop-the-world boundary identified by `schema_epoch=2` and
`state_model_revision=bot-job-v2`. Control and Worker refuse another epoch; mixed-version rolling
deployment is unsupported.

`ExecutionContract` is immutable canonical JSON. It binds repository/installation/Issue identity,
base branch and SHA, Issue input hash, provider/model/runtime/profile/Skill revisions, phase tool
allowlists, validation commands, denied paths, patch limits, immutable image, Docker resources and
Draft/ready publication mode. Tokens, keys, secret paths, prompts and host absolute paths are not
contract fields.

`bot_jobs` stores stable repository and Issue identity as indexed columns and the complete typed Job
record as canonical JSON. The partial unique index on `(repository_id, issue_number)` covers every
active state. Plan and implementation have separate attempts, Sessions, workspace paths/readiness,
and the Job separately records approval/artifact, lease, retry phase, branch/commit/PR, error,
version and timestamps. It also stores only a bounded RuntimeEvent type and timestamp for progress;
Prompt、Tool 输入输出和文件内容不会进入 Job。不存在通用 `workspace_path` 回退字段。

`bot_inbox_messages` deduplicates GitHub comment IDs. A Plan reply is appended to the SQLite Session
event chain and marked consumed in one `BEGIN IMMEDIATE` transaction. Plan, approval and delivery
artifacts carry the execution-contract hash and base SHA; approval evidence hashes the plan and its
contract binding (which itself contains the Issue input hash).

Epoch 1 data is archived rather than imported. The semantic names used for archive inspection are
`waiting_implementation → waiting_approval`, `blocked → blocked_plan`, and `failed → dead_letter`.
