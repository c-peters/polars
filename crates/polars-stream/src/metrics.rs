use std::sync::Arc;
use std::time::Duration;

use polars_async::executor::TaskMetrics;
pub use polars_io::metrics::{IOMetrics, OptIOMetrics};
use slotmap::{SecondaryMap, SlotMap};

use crate::LogicalPipe;
use crate::graph::{GraphNodeKey, LogicalPipeKey};
use crate::pipe::PipeMetrics;

#[derive(Default, Clone)]
pub struct NodeMetrics {
    pub total_polls: u64,
    pub total_stolen_polls: u64,
    pub total_poll_time_ns: u64,
    pub max_poll_time_ns: u64,

    pub total_state_updates: u64,
    pub total_state_update_time_ns: u64,
    pub max_state_update_time_ns: u64,

    pub morsels_sent: u64,
    pub rows_sent: u64,
    pub largest_morsel_sent: u64,
    pub morsels_received: u64,
    pub rows_received: u64,
    pub largest_morsel_received: u64,

    pub io_total_active_ns: u64,
    pub io_total_bytes_requested: u64,
    pub io_total_bytes_received: u64,
    pub io_total_bytes_sent: u64,

    /// How much decode time has already been folded into `total_poll_time_ns`.
    /// `IOMetrics` is cumulative and re-read on every flush, so only the delta
    /// may be added -- unlike the `io_total_*` fields, the poll totals are
    /// accumulated from drained task metrics and must not be reset.
    io_decode_applied_poll_ns: u64,

    pub state_update_in_progress: bool,
    pub num_running_tasks: u32,
    pub done: bool,
}

impl NodeMetrics {
    fn add_task(&mut self, task_metrics: &TaskMetrics) {
        self.total_polls += task_metrics.total_polls.load();
        self.total_stolen_polls += task_metrics.total_stolen_polls.load();
        self.total_poll_time_ns += task_metrics.total_poll_time_ns.load();
        self.max_poll_time_ns = self
            .max_poll_time_ns
            .max(task_metrics.max_poll_time_ns.load());
        self.num_running_tasks += (!task_metrics.done.load()) as u32;
    }

    fn add_io(&mut self, io_metrics: &IOMetrics) {
        self.io_total_active_ns += io_metrics.io_timer.total_time_live_ns();
        self.io_total_bytes_requested += io_metrics.bytes_requested.load();
        self.io_total_bytes_received += io_metrics.bytes_received.load();
        self.io_total_bytes_sent += io_metrics.bytes_sent.load();

        // A detached decode task is one of this node's tasks that happens to
        // carry no node key, so its metrics belong in the same counters
        // `add_task` feeds -- same unit, same meaning. Folding them here rather
        // than exposing them separately means every consumer of
        // `total_poll_time_ns` sees a scan's real CPU with no schema change.
        //
        // Only the delta: cumulative and re-read on every flush.
        //
        // Poll time only. `total_polls`, `total_stolen_polls` and
        // `max_poll_time_ns` deliberately still count streaming polls alone, so
        // on a scan the time and the count describe different populations.
        let poll_ns = io_metrics.decode_task_poll_ns.load();
        self.total_poll_time_ns += poll_ns.saturating_sub(self.io_decode_applied_poll_ns);
        self.io_decode_applied_poll_ns = poll_ns;
    }

    fn reset_io_metrics(&mut self) {
        self.io_total_active_ns = 0;
        self.io_total_bytes_requested = 0;
        self.io_total_bytes_received = 0;
        self.io_total_bytes_sent = 0;
        // Deliberately not the decode fold: it lives in the poll totals, which
        // accumulate rather than being recomputed, and `add_io` re-applies only
        // the delta.
    }

    fn start_state_update(&mut self) {
        self.state_update_in_progress = true;
    }

    fn stop_state_update(&mut self, time: Duration, is_done: bool) {
        let time_ns = time.as_nanos() as u64;
        self.total_state_updates += 1;
        self.total_state_update_time_ns += time_ns;
        self.max_state_update_time_ns = self.max_state_update_time_ns.max(time_ns);
        self.state_update_in_progress = false;
        self.done = is_done;
    }

    fn add_send_metrics(&mut self, pipe_metrics: &PipeMetrics) {
        self.morsels_sent += pipe_metrics.morsels_sent.load();
        self.rows_sent += pipe_metrics.rows_sent.load();
        self.largest_morsel_sent = self
            .largest_morsel_sent
            .max(pipe_metrics.largest_morsel_sent.load());
    }

    fn add_recv_metrics(&mut self, pipe_metrics: &PipeMetrics) {
        self.morsels_received += pipe_metrics.morsels_received.load();
        self.rows_received += pipe_metrics.rows_received.load();
        self.largest_morsel_received = self
            .largest_morsel_received
            .max(pipe_metrics.largest_morsel_received.load());
    }
}

#[derive(Default, Clone)]
pub struct GraphMetrics {
    node_metrics: SecondaryMap<GraphNodeKey, NodeMetrics>,
    in_progress_io_metrics: SecondaryMap<GraphNodeKey, Arc<IOMetrics>>,
    in_progress_task_metrics: SecondaryMap<GraphNodeKey, Vec<Arc<TaskMetrics>>>,
    in_progress_pipe_metrics: SecondaryMap<LogicalPipeKey, Vec<Arc<PipeMetrics>>>,
}

impl GraphMetrics {
    pub fn add_task(&mut self, key: GraphNodeKey, task_metrics: Arc<TaskMetrics>) {
        self.in_progress_task_metrics
            .entry(key)
            .unwrap()
            .or_default()
            .push(task_metrics);
    }

    pub fn add_pipe(&mut self, key: LogicalPipeKey, pipe_metrics: Arc<PipeMetrics>) {
        self.in_progress_pipe_metrics
            .entry(key)
            .unwrap()
            .or_default()
            .push(pipe_metrics);
    }

    pub fn start_state_update(&mut self, key: GraphNodeKey) {
        self.node_metrics
            .entry(key)
            .unwrap()
            .or_default()
            .start_state_update();
    }

    pub fn stop_state_update(&mut self, key: GraphNodeKey, time: Duration, is_done: bool) {
        self.node_metrics[key].stop_state_update(time, is_done);
    }

    pub fn flush(&mut self, pipes: &SlotMap<LogicalPipeKey, LogicalPipe>) {
        for (key, in_progress_task_metrics) in self.in_progress_task_metrics.iter_mut() {
            let this_node_metrics = self.node_metrics.entry(key).unwrap().or_default();
            this_node_metrics.num_running_tasks = 0;
            for task_metrics in in_progress_task_metrics.drain(..) {
                this_node_metrics.add_task(&task_metrics);
            }
        }

        for (key, io_metrics) in self.in_progress_io_metrics.iter_mut() {
            let this_node_metrics = self.node_metrics.entry(key).unwrap().or_default();
            this_node_metrics.reset_io_metrics();
            this_node_metrics.add_io(io_metrics);
        }

        for (key, in_progress_pipe_metrics) in self.in_progress_pipe_metrics.iter_mut() {
            for pipe_metrics in in_progress_pipe_metrics.drain(..) {
                let pipe = &pipes[key];
                self.node_metrics
                    .entry(pipe.receiver)
                    .unwrap()
                    .or_default()
                    .add_recv_metrics(&pipe_metrics);
                self.node_metrics
                    .entry(pipe.sender)
                    .unwrap()
                    .or_default()
                    .add_send_metrics(&pipe_metrics);
            }
        }
    }

    pub fn get(&self, key: GraphNodeKey) -> Option<&NodeMetrics> {
        self.node_metrics.get(key)
    }

    pub fn iter(&self) -> slotmap::secondary::Iter<'_, GraphNodeKey, NodeMetrics> {
        self.node_metrics.iter()
    }
}

/// Routes a spawned column-decode task's metrics onto its scan node's `IOMetrics`.
///
/// `row_group_decode` fans column decoding out with `parallelize_first_to_local`,
/// which runs the first chunk inline and spawns the rest. The inline chunk lands
/// in the parent decode task's poll time; the spawned ones are detached and reach
/// nothing. Handing this down puts them on the same node as the parent.
///
/// The two chunking rules differ, and only one is governed by the tuning target:
///
///   - unfiltered: `calc_cols_per_thread` chunks once a row group's projected
///     cell count passes `target_values_per_thread` (16Mi), i.e. above ~64
///     projected columns for polars-written files and ~16 for pyarrow's defaults
///   - prefiltered, i.e. any scan with a pushed-down predicate: chunks columns
///     across `num_pipelines` unconditionally, ignoring the target
///
/// So a filtered scan always fans out, whatever the row group geometry, and
/// attributing only the parent under-reports those substantially.
///
/// This path is more exposed to the executor's record-after-`run` race than the
/// parent is (see `await_decode`). A chunk future contains no `.await`, so it is
/// polled exactly once; `run` wakes the joining parent from inside itself, and
/// the executor adds the poll's time only after `run` returns. If the parent
/// wins that window the chunk folds in as zero -- losing all of it, not a
/// fraction, and biased toward the chunk whose completion did the waking.
/// Measured totals do not show it firing (forcing maximum fan-out reports the
/// same CPU as forcing none), but the fix is the same one named on
/// `await_decode`: record inside `run`, before the wake.
pub struct DecodeTaskObserver(pub Arc<IOMetrics>);

impl polars_async::executor::SpawnedTaskObserver for DecodeTaskObserver {
    fn task_finished(&self, metrics: &TaskMetrics) {
        OptIOMetrics(Some(self.0.clone())).add_decode_task(metrics.total_poll_time_ns.load());
    }
}

pub struct NodeMetricsRegistrator {
    pub graph_key: GraphNodeKey,
    pub graph_metrics: Arc<parking_lot::Mutex<GraphMetrics>>,
}

impl NodeMetricsRegistrator {
    /// # Panics
    /// When debug_assertions enabled, panics if called more than once for a node within a single
    /// phase.
    pub fn register_io_metrics(&self, io_metrics: Arc<IOMetrics>) {
        let mut guard = self.graph_metrics.lock();

        use slotmap::secondary::Entry;

        match guard.in_progress_io_metrics.entry(self.graph_key).unwrap() {
            Entry::Occupied(e) => {
                // Each node should only have 1 set of metrics, identified by the Arc address.
                // But the registration can be called multiple times (per phase).
                assert!(Arc::ptr_eq(&io_metrics, e.get()));
            },
            Entry::Vacant(e) => {
                e.insert(io_metrics);
            },
        };
    }
}
