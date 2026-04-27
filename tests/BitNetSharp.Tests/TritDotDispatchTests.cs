using System.Reflection;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

/// <summary>
/// G0 dispatcher contract. The dispatcher inside
/// <see cref="TritPacking.TernaryDotSimdUnpacked"/> branches on a small set
/// of static flags read at startup; these tests assert each flag matches the
/// underlying intrinsic capability and that <c>ForceGeneric</c> short-circuits
/// correctly.
/// </summary>
public sealed class TritDotDispatchTests
{
    [Fact]
    public void HasFlags_AgreeWithRuntimeIntrinsicsCapabilities()
    {
        // Reflection lets the test compare against the same intrinsic types
        // without requiring the production code to expose each predicate as
        // a separate property. If the runtime ever changes shape (e.g. drops
        // V512 nesting), this test is the canary.
        var dispatchType = typeof(TritPacking).Assembly.GetType("BitNetSharp.Core.Quantization.TritDotDispatch")!;
        Assert.NotNull(dispatchType);

        var hasV512 = (bool)dispatchType.GetField("HasAvxVnniInt8V512", BindingFlags.Public | BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        var hasInt8 = (bool)dispatchType.GetField("HasAvxVnniInt8", BindingFlags.Public | BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        var hasAvx2 = (bool)dispatchType.GetField("HasAvx2", BindingFlags.Public | BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;

        Assert.Equal(System.Runtime.Intrinsics.X86.AvxVnniInt8.V512.IsSupported, hasV512);
        Assert.Equal(System.Runtime.Intrinsics.X86.AvxVnniInt8.IsSupported, hasInt8);
        Assert.Equal(System.Runtime.Intrinsics.X86.Avx2.IsSupported, hasAvx2);

        // Inclusion: V512 implies Int8 implies Avx2 (any host that reports V512
        // necessarily reports the lower tiers).
        if (hasV512) Assert.True(hasInt8);
        if (hasInt8) Assert.True(hasAvx2);
    }

    [Fact]
    public void ForceGeneric_ShortCircuitsAllAcceleratedFlags()
    {
        var dispatchType = typeof(TritPacking).Assembly.GetType("BitNetSharp.Core.Quantization.TritDotDispatch")!;
        var force = dispatchType.GetField("ForceGeneric", BindingFlags.NonPublic | BindingFlags.Static)!;
        var orig = (bool)force.GetValue(null)!;
        try
        {
            force.SetValue(null, true);
            var useV512 = (bool)dispatchType.GetProperty("UseAvxVnniInt8V512")!.GetValue(null)!;
            var useInt8 = (bool)dispatchType.GetProperty("UseAvxVnniInt8")!.GetValue(null)!;
            var useAvx2 = (bool)dispatchType.GetProperty("UseAvx2Sign")!.GetValue(null)!;
            Assert.False(useV512);
            Assert.False(useInt8);
            Assert.False(useAvx2);
        }
        finally
        {
            force.SetValue(null, orig);
        }
    }
}
