# Explicit Task Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace thread-local task attribution with explicit ownership at every computational spawn while preserving metrics and query behavior.

**Architecture:** Retain TaskAttribution and per-task metrics in polars-async. Thread an optional ownership handle through node-specific StreamingExecutionState, reader/writer contexts, and spawn helpers. Each executor task retains its handle and uses it directly for registration and occupancy.

**Tech Stack:** Rust nightly-2026-09-01, Polars computational executor, Tokio, Cargo.

**Spec:** docs/superpowers/specs/2026-09-29-explicit-task-attribution.md

## Global Constraints

- Base adb73c51e5, branch refactor/explicit-task-attribution, worktree /tmp/polars-driver-phase-metrics.
- No ambient attribution lookup or TLS-restoring future wrapper remains.
- Every task registers once before scheduling, with separate TaskMetrics and the correct query/node owner.
- No indiscriminate empty ownership at node-owned spawn sites; unrelated scheduler TLS remains.
- Preserve priorities, lifetimes, cancellation, results, occupancy, and disabled-monitoring behavior.
- No graph-specific dependency in polars-async; no graph locking per poll; no new measurement of Tokio work.
- Run builds with CARGO_TARGET_DIR=/tmp/polars-explicit-attribution-target and keep test logs under /tmp.
- Do not push or publish. Stage only this task's files.

## Review Focus

- Nested tasks spawned after a parent yields, including a Tokio handoff, retain their explicit owner.
- Concurrent queries cannot attribute work to each other, even on the same worker.
- Inline helper futures are not double counted; cancelled and scoped tasks retain existing cleanup semantics.
- Work started during node update_state and pipeline setup has an owner, including optional reader/writer and spill paths.
- Disabled metrics and explicitly ownerless tasks do not accidentally register with a nearby node.

---

### Task 1: Migrate executor and streaming ownership end to end

**Files:**
- Modify: crates/polars-async/src/executor/mod.rs and crates/polars-async/src/primitives/opt_spawned_future.rs.
- Modify: crates/polars-stream/src/{execute.rs,graph.rs,metrics.rs,pipe.rs}.
- Modify: affected spawn sites and contexts under crates/polars-stream/src/nodes/ (sources, sinks, joins, aggregation and transformations).
- Modify: affected callers in crates/polars-ooc and any other callers found by compiler/inventory.
- Test: executor attribution tests under crates/polars-async; integration coverage under crates/polars-lazy/src/tests/observer.rs as appropriate.

**Interfaces:**
- Consumes: existing TaskAttribution::{task_spawned,poll_session}, GraphMetrics node attribution, and per-task TaskMetadata.
- Produces: cloneable optional attribution handle; spawn(priority, attribution, future); TaskScope::spawn_task(priority, attribution, future); attribution-aware LocalOrSpawnedFuture and parallelize_first_to_local; StreamingExecutionState.attribution.
- Graph/node attribution construction returns a handle, replacing the ambient guard. Any context additions must be carried through each constructor and helper call.

- [ ] Add executor behavioral tests first. Prove exact registration counts for a parent with explicitly assigned and unassigned children, distinct query/node owners, scoped tasks, and inline/spawned helpers. Exercise a yield and a Tokio handoff. Use distinct recorded metric Arcs to detect duplicate registration. Serialize tests that toggle global tracking.
- [ ] Run the tests against the old API and record the expected failure; where an API does not exist yet, document the compile failure and demonstrate behavioral failures during migration rather than claiming those errors alone prove behavior.
- [ ] Introduce explicit attribution in both spawn APIs; retain per-task ownership and registration-before-schedule. Remove attribution TLS, guard and restoring wrapper, preserving worker timing TLS and occupancy polling.
- [ ] Construct owner-specific execution state at graph update/spawn boundaries; preserve scoped borrow lifetimes. Explicitly attribute pipe tasks to the receiver. Propagate handles through all direct and helper spawns, source/sink initialization and nested helpers. Review every explicit empty owner against the call chain.
- [ ] Run cargo test --offline -p polars-async --lib; expect all tests pass. Run cargo test --offline -p polars-lazy --features test,streaming --no-default-features --lib; expect all tests pass.
- [ ] Run cargo check --offline -p polars-stream --all-features, or the supported feature matrix if mutually incompatible features prevent that combination. Include parquet,csv,ipc,json,scan_lines,python,object,dtype-categorical and joins/aggregation optional paths. Record actual commands and any pre-existing build limitations.
- [ ] Format touched Rust files and run git diff --check. Audit the spawn inventory and removed ambient symbols. Self-review ownership/cancellation and commit the coherent migration.

### Task 2: Independent review and final verification

**Files:** Review all Task 1 changes; update this plan's completion record.

**Interfaces:** Consumes explicit ownership APIs and test evidence from Task 1. Produces a verified branch and a concise handoff.

- [ ] Review spec compliance and code quality against the full diff, with particular attention to every empty owner and helpers that spawn only under size thresholds.
- [ ] Address any important findings with a reproducing test and run affected tests; retain exact failures and outcomes in the ledger.
- [ ] Verify the changed branch is clean, record commit IDs, and report the plan path, branch, tests and any remaining limitations. Leave the requested branch available locally.
