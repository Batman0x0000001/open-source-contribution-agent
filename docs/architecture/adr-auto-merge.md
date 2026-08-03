# ADR: auto-merge is deferred

Open Source Contribution Agent `0.3.4` may create Draft or ready-for-review pull requests but never merges or
enables GitHub auto-merge. A future change requires an explicit repository-owner setting, protected
branches and required checks, a contract-bound approval, audit events, a kill switch, and a separate
security review. Unknown `auto_merge` repository configuration is rejected rather than ignored.
