use std::sync::Arc;

use polars_utils::live_timer::{LiveTimer, LiveTimerSession};
use polars_utils::relaxed_cell::RelaxedCell;

/// Padded to its own cache line like its siblings `TaskMetrics` and
/// `PipeMetrics`: the decode counters are written once per row group from
/// whichever executor thread ran that decode, so many threads write one instance
/// concurrently and two unrelated scan nodes' instances must not share a line.
#[derive(Debug, Default, Clone)]
#[repr(align(128))]
pub struct IOMetrics {
    pub io_timer: LiveTimer,
    pub bytes_requested: RelaxedCell<u64>,
    pub bytes_received: RelaxedCell<u64>,
    pub bytes_sent: RelaxedCell<u64>,

    /// Thread time of this node's detached decode tasks, aggregated from their
    /// `TaskMetrics` as each completes.
    ///
    /// Decode tasks are spawned detached, so they carry no `GraphNodeKey` and the
    /// executor's registration in `run_subgraph` never sees them; without this a
    /// scan's CPU is almost entirely invisible. Same unit as a node's poll time,
    /// so the two are additive.
    pub decode_task_poll_ns: RelaxedCell<u64>,
}

#[derive(Debug, Clone)]
pub struct OptIOMetrics(pub Option<Arc<IOMetrics>>);

impl OptIOMetrics {
    pub fn start_io_session(&self) -> Option<LiveTimerSession> {
        self.0.as_ref().map(|x| x.io_timer.start_session())
    }

    /// Folds one finished decode task's poll time onto this node.
    ///
    /// Takes a plain integer rather than `&TaskMetrics` so polars-io need not
    /// depend on polars-async. Call only once the task has completed, or the
    /// value read will be partial.
    pub fn add_decode_task(&self, poll_time_ns: u64) {
        let Some(m) = self.0.as_ref() else {
            return;
        };
        m.decode_task_poll_ns.fetch_add(poll_time_ns);
    }

    pub fn add_bytes_requested(&self, bytes_requested: u64) {
        self.0
            .as_ref()
            .map(|x| x.bytes_requested.fetch_add(bytes_requested));
    }

    pub fn add_bytes_received(&self, bytes_received: u64) {
        self.0
            .as_ref()
            .map(|x| x.bytes_received.fetch_add(bytes_received));
    }

    pub fn add_bytes_sent(&self, bytes_sent: u64) {
        self.0.as_ref().map(|x| x.bytes_sent.fetch_add(bytes_sent));
    }

    pub async fn record_io_read<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        self.add_bytes_requested(num_bytes);

        let io_session = self.start_io_session();

        let out = fut.await;

        drop(io_session);

        self.add_bytes_received(num_bytes);

        out
    }

    pub async fn record_bytes_tx<F, O>(&self, num_bytes: u64, fut: F) -> O
    where
        F: Future<Output = O>,
    {
        let io_session = self.start_io_session();

        let out = fut.await;

        drop(io_session);

        self.add_bytes_sent(num_bytes);

        out
    }
}
