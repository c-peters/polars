use std::sync::Arc;

use pin_project_lite::pin_project;
use polars_utils::{UnitVec, unitvec};

use crate::executor::{AbortOnDropHandle, SpawnedTaskObserver, TaskPriority, spawn};

pin_project! {
    /// Represents a future that may either be local or spawned.
    ///
    /// `Spawned` carries an optional observer, folded on completion so the
    /// spawned task's CPU can be attributed to whoever fanned it out. It is
    /// `None` in the common case, costing one null check per poll.
    #[project = LocalOrSpawnedFutureProj]
    pub enum LocalOrSpawnedFuture<F, O> {
        Local { #[pin] fut: F },
        Spawned {
            #[pin] handle: AbortOnDropHandle<O>,
            observer: Option<Arc<dyn SpawnedTaskObserver>>,
        }
    }
}

impl<F, O> LocalOrSpawnedFuture<F, O>
where
    F: Future<Output = O>,
{
    /// Wraps the future in a `Local` variant.
    pub fn new_local(fut: F) -> Self {
        LocalOrSpawnedFuture::Local { fut }
    }
}

impl<F, O> LocalOrSpawnedFuture<F, O>
where
    F: Future<Output = O> + Send + 'static,
    O: Send + 'static,
{
    /// Spawns the future onto the async executor.
    pub fn spawn(task_priority: TaskPriority, fut: F) -> Self {
        Self::spawn_observed(task_priority, fut, None)
    }

    /// Spawns the future, reporting its `TaskMetrics` to `observer` once it
    /// completes.
    pub fn spawn_observed(
        task_priority: TaskPriority,
        fut: F,
        observer: Option<Arc<dyn SpawnedTaskObserver>>,
    ) -> Self {
        LocalOrSpawnedFuture::Spawned {
            handle: AbortOnDropHandle::new(spawn(task_priority, fut)),
            observer,
        }
    }
}

impl<F, O> Future for LocalOrSpawnedFuture<F, O>
where
    F: Future<Output = O>,
{
    type Output = O;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        match self.project() {
            LocalOrSpawnedFutureProj::Local { fut } => fut.poll(cx),
            LocalOrSpawnedFutureProj::Spawned {
                mut handle,
                observer,
            } => {
                // The `Local` arm needs no equivalent: it runs inline, so its CPU
                // is already inside the calling task's own poll time.
                let out = handle.as_mut().poll(cx);
                if out.is_ready()
                    && let Some(observer) = observer.take()
                    && let Some(metrics) = handle.metrics()
                {
                    observer.task_finished(metrics);
                }
                out
            },
        }
    }
}

/// Parallelizes futures across the computational async runtime.
///
/// As an optimization for cache access, the first future is kept on the current thread. If there
/// is only 1 future, then all data is kept on the current thread and spawn is not called at all.
///
/// Note this means the first future in the returned iterator does not run until polled.
///
/// Note that dropping the iterator will call abort on all spawned futures, as this is intended to be
/// used for compute.
pub fn parallelize_first_to_local<'i, 'o, I, F, O>(
    task_priority: TaskPriority,
    futures_iter: I,
) -> impl ExactSizeIterator<Item = impl Future<Output = O> + Send + 'static> + 'o
where
    I: Iterator<Item = F> + 'i,
    F: Future<Output = O> + Send + 'static,
    O: Send + 'static,
{
    parallelize_first_to_local_impl(task_priority, futures_iter, None).into_iter()
}

/// As [`parallelize_first_to_local`], but each *spawned* future reports its
/// `TaskMetrics` to `observer` on completion.
///
/// Without this the fan-out is invisible to the caller's own accounting: the
/// first future runs inline and lands in the caller's poll time, while the rest
/// become detached tasks that nothing collects. The wider the fan-out, the
/// larger the share that goes missing.
pub fn parallelize_first_to_local_observed<'i, 'o, I, F, O>(
    task_priority: TaskPriority,
    futures_iter: I,
    observer: Option<&Arc<dyn SpawnedTaskObserver>>,
) -> impl ExactSizeIterator<Item = impl Future<Output = O> + Send + 'static> + 'o
where
    I: Iterator<Item = F> + 'i,
    F: Future<Output = O> + Send + 'static,
    O: Send + 'static,
{
    parallelize_first_to_local_impl(task_priority, futures_iter, observer).into_iter()
}

fn parallelize_first_to_local_impl<I, F, O>(
    task_priority: TaskPriority,
    mut futures_iter: I,
    observer: Option<&Arc<dyn SpawnedTaskObserver>>,
) -> UnitVec<LocalOrSpawnedFuture<F, O>>
where
    I: Iterator<Item = F>,
    F: Future<Output = O> + Send + 'static,
    O: Send + 'static,
{
    let Some(first_fut) = futures_iter.next() else {
        return UnitVec::new();
    };

    let first_fut = LocalOrSpawnedFuture::new_local(first_fut);

    let Some(second_fut) = futures_iter.next() else {
        return unitvec![first_fut];
    };

    let mut futures = UnitVec::with_capacity(2 + futures_iter.size_hint().0);

    // Note:
    // * The local future must come first to ensure we don't block polling it.
    // * Remaining futures must all be spawned upfront into the Vec for them to run parallel.
    // Cloned only from here on: the single-future early return above is the
    // common case when a projection is too narrow to chunk, and it should not
    // pay a refcount bump for a fan-out that never happens.
    futures.extend([
        first_fut,
        LocalOrSpawnedFuture::spawn_observed(task_priority, second_fut, observer.cloned()),
    ]);
    futures.extend(
        futures_iter
            .map(|x| LocalOrSpawnedFuture::spawn_observed(task_priority, x, observer.cloned())),
    );

    futures
}
