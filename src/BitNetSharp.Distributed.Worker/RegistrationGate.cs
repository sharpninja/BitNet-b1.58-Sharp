using System;
using System.Threading;
using System.Threading.Tasks;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Thread-safe latch that tracks whether the worker currently holds a
/// live registration with the coordinator. Used by the registration
/// supervisor, the heartbeat loop, and the work loop to coordinate
/// automatic reconnection when the coordinator restarts, drops the
/// worker row, or becomes temporarily unreachable.
///
/// <para>
/// The class exposes two primitives the supervisor needs:
///   <list type="bullet">
///     <item><see cref="IsRegistered"/> — fast non-blocking check.</item>
///     <item><see cref="WaitForLossAsync"/> — awaits the next transition
///       from Registered to Lost so the supervisor only wakes up when
///       there is work to do.</item>
///   </list>
/// </para>
/// </summary>
internal sealed class RegistrationGate
{
    private readonly object _lock = new();
    private TaskCompletionSource _lostTcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private volatile bool _isRegistered;

    /// <summary>True between a successful register and the next loss signal.</summary>
    public bool IsRegistered => _isRegistered;

    /// <summary>
    /// Called by the registration supervisor after <c>/register</c>
    /// returns a non-null response. Safe to call repeatedly; does not
    /// reset the wait task.
    /// </summary>
    public void MarkRegistered()
    {
        lock (_lock)
        {
            _isRegistered = true;
        }
    }

    /// <summary>
    /// Called by the heartbeat or work loop when the coordinator
    /// signals that the worker row is gone (HTTP 410) or when a long
    /// run of transient failures strongly suggests the coordinator is
    /// down. Safe to call from any thread; only the first call per
    /// cycle fires the waiter.
    /// </summary>
    public void MarkLost()
    {
        TaskCompletionSource toFire;
        lock (_lock)
        {
            if (!_isRegistered)
            {
                return;
            }
            _isRegistered = false;
            toFire = _lostTcs;
            _lostTcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        }
        toFire.TrySetResult();
    }

    /// <summary>
    /// Returns a task that completes the next time <see cref="MarkLost"/>
    /// is called. If the gate is already in the Lost state the returned
    /// task still only completes on the NEXT loss, so callers should
    /// check <see cref="IsRegistered"/> first.
    /// </summary>
    public Task WaitForLossAsync(CancellationToken cancellationToken)
    {
        TaskCompletionSource tcs;
        lock (_lock)
        {
            tcs = _lostTcs;
        }
        return tcs.Task.WaitAsync(cancellationToken);
    }
}
