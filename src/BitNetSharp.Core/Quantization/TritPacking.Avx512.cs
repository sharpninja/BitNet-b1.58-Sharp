using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Core.Quantization;

/// <summary>
/// G-series hardware-targeted ternary dot kernels. Each kernel is gated by
/// the corresponding <see cref="TritDotDispatch"/> flag; callers should
/// reach these only via the dispatcher in <see cref="TernaryDotSimdUnpacked"/>.
///
/// Every kernel is bit-exact with <see cref="TernaryDotScalar"/>; equivalence
/// is enforced by the <c>TritDotAvx2Tests</c> and <c>TritDotAvxVnniInt8Tests</c>
/// suites.
/// </summary>
public static partial class TritPacking
{
    /// <summary>
    /// AVX2 ternary dot using <c>VPSIGNB</c>: <c>Avx2.Sign(act, trit)</c>
    /// produces <c>act * trit</c> per byte directly because trit ∈ {-1, 0, +1}.
    /// Cuts the per-32-lane chunk from ~11 ops (compare/select/subtract/widen×3/add×3)
    /// to ~6 ops (sign + widen×2 + add×3). Universal on every modern x64
    /// (Haswell+, Excavator+); the only kernel measurable on Zen 1-3 hosts.
    ///
    /// <para>
    /// <b>Activation domain.</b> Requires <c>activations[i] ∈ [-127, +127]</c>:
    /// <c>VPSIGNB</c> with <c>sign &lt; 0</c> produces <c>-act</c> in sbyte
    /// arithmetic, so <c>act = -128</c> would wrap to <c>-128</c> instead of
    /// the scalar oracle's <c>+128</c>. BitNet's
    /// <see cref="BitNetSharp.Core.Layers.BitLinear"/> quantiser clamps to
    /// <c>±127</c>, so this constraint is always satisfied in production.
    /// </para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int TernaryDotAvx2Sign(
        ReadOnlySpan<sbyte> trits,
        ReadOnlySpan<sbyte> activations)
    {
        if (!Avx2.IsSupported)
        {
            return TernaryDotSimdUnpackedGeneric(trits, activations);
        }

        var length = trits.Length;
        var sum = 0;
        var processed = 0;

        if (length >= Vector256<sbyte>.Count)
        {
            var laneCount = Vector256<sbyte>.Count; // 32
            var chunks = length / laneCount;
            var acc = Vector256<int>.Zero;

            // H4: ref-base load lifts the per-iteration span bounds check
            // (Vector256.Create<sbyte>(span.Slice(offset, lane)) emits a length
            // check; LoadUnsafe trusts the caller). The outer (length >= laneCount)
            // and (chunks * laneCount <= length) gates already guarantee in-bounds.
            ref var tritRef = ref MemoryMarshal.GetReference(trits);
            ref var actRef = ref MemoryMarshal.GetReference(activations);

            for (var c = 0; c < chunks; c++)
            {
                var offset = (nuint)(c * laneCount);
                var tritVec = Vector256.LoadUnsafe(ref tritRef, offset);
                var actVec = Vector256.LoadUnsafe(ref actRef, offset);

                // VPSIGNB: contribution[i] = trit[i] > 0 ? act[i]
                //                          : trit[i] == 0 ? 0
                //                          : -act[i].
                // Trit ∈ {-1, 0, +1} ⇒ contribution[i] = act[i] * trit[i] exactly.
                var contrib = Avx2.Sign(actVec, tritVec);

                // Widen 32 sbyte → 16 short + 16 short, sum the two halves, then
                // widen 16 short → 8 int + 8 int. Worst-case contribution
                // magnitude is 127, so the short sum across the 32-lane chunk
                // stays well inside int16 range.
                (Vector256<short> sLo, Vector256<short> sHi) = Vector256.Widen(contrib);
                var sumShort = Avx2.Add(sLo, sHi);
                (Vector256<int> iLo, Vector256<int> iHi) = Vector256.Widen(sumShort);
                acc = Avx2.Add(acc, Avx2.Add(iLo, iHi));
            }

            sum = Vector256.Sum(acc);
            processed = chunks * laneCount;
        }

        for (var i = processed; i < length; i++)
        {
            var t = trits[i];
            if (t > 0) sum += activations[i];
            else if (t < 0) sum -= activations[i];
        }

        return sum;
    }

    /// <summary>
    /// AVX-VNNI-INT8 ternary dot using <c>VPDPBSSD</c> at 256-bit. One fused
    /// instruction multiplies pairs of signed bytes and accumulates into 32-bit
    /// lanes, replacing the entire sign+widen+add chain. Hardware: Intel
    /// Sapphire Rapids+, Granite Rapids; AMD Zen 5+. Falls through to
    /// <see cref="TernaryDotAvx2Sign"/> when unavailable.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int TernaryDotAvxVnniInt8(
        ReadOnlySpan<sbyte> trits,
        ReadOnlySpan<sbyte> activations)
    {
        if (!AvxVnniInt8.IsSupported)
        {
            return TernaryDotAvx2Sign(trits, activations);
        }

        var length = trits.Length;
        var sum = 0;
        var processed = 0;

        if (length >= Vector256<sbyte>.Count)
        {
            var laneCount = Vector256<sbyte>.Count; // 32
            var chunks = length / laneCount;
            var acc = Vector256<int>.Zero;

            ref var tritRef = ref MemoryMarshal.GetReference(trits);
            ref var actRef = ref MemoryMarshal.GetReference(activations);

            for (var c = 0; c < chunks; c++)
            {
                var offset = (nuint)(c * laneCount);
                var tritVec = Vector256.LoadUnsafe(ref tritRef, offset);
                var actVec = Vector256.LoadUnsafe(ref actRef, offset);
                acc = AvxVnniInt8.MultiplyWideningAndAdd(acc, tritVec, actVec);
            }

            sum = Vector256.Sum(acc);
            processed = chunks * laneCount;
        }

        for (var i = processed; i < length; i++)
        {
            var t = trits[i];
            if (t > 0) sum += activations[i];
            else if (t < 0) sum -= activations[i];
        }

        return sum;
    }

    /// <summary>
    /// AVX-VNNI-INT8 ternary dot at 512-bit width (V512). 64 sbyte lanes per
    /// chunk fused via <c>VPDPBSSD ZMM</c>. Hardware: Sapphire Rapids+,
    /// Granite Rapids; AMD Zen 5+ that expose AVX-512 alongside AVX-VNNI-INT8.
    /// Falls through to the 256-bit kernel when unavailable.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int TernaryDotAvxVnniInt8V512(
        ReadOnlySpan<sbyte> trits,
        ReadOnlySpan<sbyte> activations)
    {
        if (!AvxVnniInt8.V512.IsSupported)
        {
            return TernaryDotAvxVnniInt8(trits, activations);
        }

        var length = trits.Length;
        var sum = 0;
        var processed = 0;

        if (length >= Vector512<sbyte>.Count)
        {
            var laneCount = Vector512<sbyte>.Count; // 64
            var chunks = length / laneCount;
            var acc = Vector512<int>.Zero;

            ref var tritRef = ref MemoryMarshal.GetReference(trits);
            ref var actRef = ref MemoryMarshal.GetReference(activations);

            for (var c = 0; c < chunks; c++)
            {
                var offset = (nuint)(c * laneCount);
                var tritVec = Vector512.LoadUnsafe(ref tritRef, offset);
                var actVec = Vector512.LoadUnsafe(ref actRef, offset);
                acc = AvxVnniInt8.V512.MultiplyWideningAndAdd(acc, tritVec, actVec);
            }

            sum = Vector512.Sum(acc);
            processed = chunks * laneCount;
        }

        // Tail of >= 32 lanes goes through the 256-bit kernel; remainder falls
        // through to scalar.
        if (length - processed >= Vector256<sbyte>.Count && AvxVnniInt8.IsSupported)
        {
            var tail = TernaryDotAvxVnniInt8(
                trits.Slice(processed),
                activations.Slice(processed));
            return sum + tail;
        }

        for (var i = processed; i < length; i++)
        {
            var t = trits[i];
            if (t > 0) sum += activations[i];
            else if (t < 0) sum -= activations[i];
        }

        return sum;
    }
}
