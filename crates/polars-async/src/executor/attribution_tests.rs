use super::*;
use crate::primitives::opt_spawned_future::{LocalOrSpawnedFuture, parallelize_first_to_local};

#[derive(Default)]
struct Owner {
    tasks: Mutex<Vec<Arc<TaskMetrics>>>,
    polls: AtomicUsize,
}
impl TaskAttribution for Owner {
    fn task_spawned(&self, metrics: &Arc<TaskMetrics>) {
        let mut tasks = self.tasks.lock();
        assert!(!tasks.iter().any(|m| Arc::ptr_eq(m, metrics)));
        tasks.push(metrics.clone());
    }
    fn poll_session(&self) -> Option<LiveTimerSession> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        None
    }
}

// One test owns the process-wide tracking switches throughout all scenarios.
#[test]
fn explicit_ownership_survives_yields_handoffs_and_scopes() {
    track_task_metrics(true);
    let _reset = WithDrop::new((), |_| track_task_metrics(false));
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let a = Arc::new(Owner::default());
    let b = Arc::new(Owner::default());
    let owner_a = TaskAttributionHandle::new(a.clone());
    let owner_b = TaskAttributionHandle::new(b.clone());
    runtime.block_on(async {
        let tokio = tokio::runtime::Handle::current();
        let child_owner = owner_a.clone();
        let other_owner = owner_b.clone();
        let parent = spawn(TaskPriority::High, owner_a.clone(), async move {
            tokio::task::yield_now().await;
            spawn(TaskPriority::High, child_owner.clone(), async {}).await;
            spawn(
                TaskPriority::High,
                TaskAttributionHandle::default(),
                async {},
            )
            .await;
            spawn(TaskPriority::High, other_owner, async {}).await;
            tokio
                .spawn(async move {
                    spawn(TaskPriority::High, child_owner, async {}).await;
                })
                .await
                .unwrap();
        });
        // Registration is synchronous, before the task can be scheduled.
        assert!(
            a.tasks
                .lock()
                .iter()
                .any(|m| Arc::ptr_eq(m, parent.metrics().unwrap()))
        );
        parent.await;
        LocalOrSpawnedFuture::new_local(async {}).await;
        LocalOrSpawnedFuture::spawn(TaskPriority::High, owner_b.clone(), async {}).await;
        for f in parallelize_first_to_local(
            TaskPriority::High,
            owner_b.clone(),
            (0..3).map(|_| async {}),
        ) {
            f.await;
        }
    });
    let borrowed = String::from("scoped");
    task_scope(|scope| {
        runtime.block_on(
            scope.spawn_task(TaskPriority::High, owner_a.clone(), async {
                assert_eq!(&borrowed, "scoped");
            }),
        );
        let cancelled = scope.spawn_task(
            TaskPriority::High,
            owner_b.clone(),
            std::future::pending::<()>(),
        );
        cancelled.cancel_handle().cancel();
        drop(cancelled);
    });
    assert_eq!(a.tasks.lock().len(), 4);
    assert_eq!(b.tasks.lock().len(), 5);
    assert!(a.polls.load(Ordering::Relaxed) > 0);
    assert!(b.polls.load(Ordering::Relaxed) > 0);
    // Each task has a distinct metric allocation, even across owners.
    for a in a.tasks.lock().iter() {
        assert!(b.tasks.lock().iter().all(|b| !Arc::ptr_eq(a, b)));
    }
    track_task_metrics(false);
    runtime.block_on(spawn(TaskPriority::High, owner_a, async {}));
    assert_eq!(a.tasks.lock().len(), 4);
}
