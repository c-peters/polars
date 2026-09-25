//! CPU-time clocks, for work that no per-task counter covers.
//!
//! The streaming engine measures per-task CPU through the executor, but work
//! outside a task -- a node's `update_state` and whatever rayon or scoped tasks
//! it fans out to -- has no such counter. Bracketing a region with
//! [`process_cpu_ns`] gives its CPU without threading a handle into whatever
//! ran inside.
//!
//! Windows advances these counters on the scheduler tick (~15.6ms), so short
//! brackets read as zero or as a whole tick. That is why [`thread_cpu_ns`],
//! which times individual polls, reports nothing there.

/// CPU time consumed by this process (all threads, user + system) in
/// nanoseconds, or `None` if the platform clock is unavailable.
///
/// Only differences are meaningful; the origin is unspecified.
#[cfg(unix)]
pub fn process_cpu_ns() -> Option<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec for the duration of the call.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    (rc == 0).then(|| ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
}

#[cfg(windows)]
pub fn process_cpu_ns() -> Option<u64> {
    use std::mem::MaybeUninit;

    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    let mut creation = MaybeUninit::<FILETIME>::uninit();
    let mut exit = MaybeUninit::<FILETIME>::uninit();
    let mut kernel = MaybeUninit::<FILETIME>::uninit();
    let mut user = MaybeUninit::<FILETIME>::uninit();

    // SAFETY: all four out-params are valid writable FILETIMEs.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            creation.as_mut_ptr(),
            exit.as_mut_ptr(),
            kernel.as_mut_ptr(),
            user.as_mut_ptr(),
        )
    };
    if ok == 0 {
        return None;
    }

    // SAFETY: GetProcessTimes succeeded, so all four are initialised.
    let (kernel, user) = unsafe { (kernel.assume_init(), user.assume_init()) };
    let to_ns = |f: FILETIME| {
        let ticks = ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
        ticks * 100 // FILETIME is in 100ns units.
    };
    Some(to_ns(kernel) + to_ns(user))
}

#[cfg(not(any(unix, windows)))]
pub fn process_cpu_ns() -> Option<u64> {
    None
}

/// CPU time consumed by the calling thread (user + system) in nanoseconds, or
/// `None` where the platform cannot report it usefully.
///
/// Only differences are meaningful. Unlike a wall clock this does not advance
/// while the thread is descheduled, which is what makes it usable for timing a
/// poll under oversubscription.
#[cfg(unix)]
pub fn thread_cpu_ns() -> Option<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec for the duration of the call.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    (rc == 0).then(|| ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
}

/// Always `None` on Windows: `GetThreadTimes` has scheduler-tick granularity,
/// which cannot resolve a poll, and reporting nothing beats reporting that.
#[cfg(windows)]
pub fn thread_cpu_ns() -> Option<u64> {
    None
}

#[cfg(not(any(unix, windows)))]
pub fn thread_cpu_ns() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_busy_work() {
        let Some(before) = process_cpu_ns() else {
            // Platform without the clock: returning None rather than a wrong
            // number is the contract, so there is nothing to assert.
            return;
        };
        let sum = (0u64..20_000_000).fold(0u64, u64::wrapping_add);
        let after = process_cpu_ns().expect("clock available before but not after");
        assert!(sum > 0);
        assert!(
            after.saturating_sub(before) > 0,
            "busy loop consumed no measurable CPU"
        );
    }
}
