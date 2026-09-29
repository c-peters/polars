# Explicit task attribution

Replace ambient node attribution in the driver-phase-metrics implementation with ownership supplied at each executor spawn. The user approved the in-chat design and requested a plan and execution on a new branch.

Base: adb73c51e5; branch: refactor/explicit-task-attribution; worktree: /tmp/polars-driver-phase-metrics.

## Contract

- Every computational executor spawn, scoped spawn, and helper that spawns must receive an explicit attribution handle. Each spawned task has its own TaskMetrics, registered once before scheduling.
- Attribution identifies both the query and node, remains in TaskMetadata across polling and thread migration, and preserves node occupancy sessions. Child tasks receive ownership explicitly; there is no ambient inheritance.
- Remove TLS_ATTRIBUTION, AttributionGuard, scoped_task_attribution, current_task_attribution, and the future wrapper that restores attribution via TLS. Keep unrelated scheduler and worker counter thread locals.
- Preserve attribution for work currently owned by nodes: update_state, node spawn, receiver-owned pipe tasks, nested executor work, Parquet decodes and their Tokio bridge, other scan/sink pipelines, and spilling where an owning node is known.
- Pass the handle through StreamingExecutionState and existing reader/writer contexts. Keep graph-specific types out of polars-async. No graph locks on the polling hot path.
- Explicitly unattributed work must be genuinely ownerless; do not migrate node work by indiscriminately passing an empty handle. Disabled monitoring remains cheap and produces no node metric registrations.
- Futures polled inline belong to their parent task and are not separately registered; separately spawned futures register once. Tokio work is not newly measured by this refactor.
- Do not change query results, scheduling priority, cancellation, or scoped task lifetimes. Do not push or publish.

## API

Use a cloneable attribution handle backed by Option<Arc<dyn TaskAttribution>>. A small wrapper with Default/Debug is allowed if reader contexts need Debug. The spawn APIs receive the handle explicitly between priority and future. StreamingExecutionState exposes the node handle; helper signatures carry the same handle. Remove ambient ownership setup at graph entry points and instead construct a node-specific state. Scoped execution must retain node state long enough for borrowed futures.

## Validation

Exercise real executor tasks for distinct owners, nested tasks, explicit unattributed children, scoped tasks, inline-versus-spawned helpers, cancellation, and monitoring disabled. Add streaming observer coverage proving nested scan/processing work remains assigned to the correct node without double registration. Run async tests, lazy library tests with streaming/test features, and reader/sink feature compilation. Use assertions on ownership and task counts rather than exact timings. Review the entire spawn inventory, including optional feature paths.
