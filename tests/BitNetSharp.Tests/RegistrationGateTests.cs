using System;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Worker;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Unit tests covering the automatic-reconnect latch the worker uses
/// to coordinate its supervisor / heartbeat / work loops. The gate
/// must:
///   <list type="bullet">
///     <item>Start in the Lost state so the supervisor tries to
///       register first.</item>
///     <item>Fire <c>WaitForLossAsync</c> exactly once per Registered
///       to Lost transition.</item>
///     <item>Be safe against repeated <c>MarkLost</c> calls (no double
///       signal).</item>
///     <item>Surface cancellation without hanging.</item>
///   </list>
/// </summary>
public sealed class RegistrationGateTests
{
    [Fact]
    public void Starts_unregistered()
    {
        var gate = new RegistrationGate();
        Assert.False(gate.IsRegistered);
    }

    [Fact]
    public void Mark_registered_flips_flag()
    {
        var gate = new RegistrationGate();
        gate.MarkRegistered();
        Assert.True(gate.IsRegistered);
    }

    [Fact]
    public async Task Wait_for_loss_completes_on_first_loss_after_registered()
    {
        var gate = new RegistrationGate();
        gate.MarkRegistered();

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        var waiter = gate.WaitForLossAsync(cts.Token);
        Assert.False(waiter.IsCompleted);

        gate.MarkLost();

        await waiter;
        Assert.False(gate.IsRegistered);
    }

    [Fact]
    public async Task Second_loss_requires_another_registered_cycle()
    {
        var gate = new RegistrationGate();
        gate.MarkRegistered();
        gate.MarkLost();

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(500));
        var waiter = gate.WaitForLossAsync(cts.Token);

        // No Register call — second MarkLost is a no-op because the
        // gate is already in the Lost state. WaitForLossAsync must
        // NOT complete from this redundant call.
        gate.MarkLost();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await waiter);
    }

    [Fact]
    public async Task Loss_signal_is_cycle_scoped()
    {
        var gate = new RegistrationGate();
        gate.MarkRegistered();
        gate.MarkLost();
        // Re-register, then loss again — new waiter should fire.
        gate.MarkRegistered();

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        var waiter = gate.WaitForLossAsync(cts.Token);
        Assert.False(waiter.IsCompleted);

        gate.MarkLost();
        await waiter;
    }

    [Fact]
    public async Task Cancellation_surfaces_on_waiter()
    {
        var gate = new RegistrationGate();
        gate.MarkRegistered();

        using var cts = new CancellationTokenSource();
        var waiter = gate.WaitForLossAsync(cts.Token);
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await waiter);
    }
}
