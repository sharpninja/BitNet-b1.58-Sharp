using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Core.Layers;

/// <summary>
/// SIMD helpers for the inner loops of attention. Both operations run once per
/// (head, target, source) triple, so on a Bonsai-sized layer the dot runs
/// 32 * seqLen * seqLen times per attention layer. The scalar version was the
/// single biggest contributor to decode-step cost even after KV caching.
/// </summary>
internal static class AttentionMath
{
    /// <summary>
    /// Dot product of two aligned spans using <see cref="Vector{T}"/>. Falls
    /// back to a scalar tail for the trailing <c>headDim % Vector.Count</c>
    /// lanes. Lengths are read from <paramref name="headDim"/> rather than
    /// span lengths so callers can pass wider buffers (e.g. full kv rows) and
    /// only sum the head slice.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> q, ReadOnlySpan<float> k, int headDim)
    {
        var width = Vector<float>.Count;
        var vectorEnd = headDim - (headDim % width);

        var acc = Vector<float>.Zero;
        var i = 0;
        while (i < vectorEnd)
        {
            var qv = new Vector<float>(q.Slice(i, width));
            var kv = new Vector<float>(k.Slice(i, width));
            acc += qv * kv;
            i += width;
        }

        var sum = Vector.Sum(acc);
        for (; i < headDim; i++)
        {
            sum += q[i] * k[i];
        }

        return sum;
    }

    /// <summary>
    /// <c>target += weight * source</c> over <paramref name="headDim"/> lanes.
    /// SIMD fma equivalent with a scalar tail.
    /// </summary>
    /// <summary>
    /// View a <c>float[rows, cols]</c> as a contiguous row-major span. Allows
    /// SIMD kernels to slice (row, head) windows without copying. The span is
    /// only valid for the lifetime of <paramref name="array"/> since it aliases
    /// the array storage directly.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Span<float> AsFlatSpan(float[,] array)
    {
        ref var first = ref Unsafe.As<byte, float>(ref MemoryMarshal.GetArrayDataReference(array));
        return MemoryMarshal.CreateSpan(ref first, array.Length);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AccumulateWeighted(Span<float> target, ReadOnlySpan<float> source, float weight, int headDim)
    {
        var width = Vector<float>.Count;
        var vectorEnd = headDim - (headDim % width);

        var scale = new Vector<float>(weight);
        var i = 0;
        while (i < vectorEnd)
        {
            var sv = new Vector<float>(source.Slice(i, width));
            var tv = new Vector<float>(target.Slice(i, width));
            var updated = tv + sv * scale;
            updated.CopyTo(target.Slice(i, width));
            i += width;
        }

        for (; i < headDim; i++)
        {
            target[i] += weight * source[i];
        }
    }

    /// <summary>
    /// Section B (KV cache quantization) - KV3: dot product of an fp32 query
    /// against a per-row absmax-quantised int8 key. Multiplies by the per-row
    /// <paramref name="kScale"/> once after the SIMD accumulation so the inner
    /// loop stays in fp32 and we pay one mul per row instead of one per lane.
    /// <para>
    /// Inner loop processes <see cref="Vector{T}"/>.Count sbytes per chunk
    /// (32 on AVX2, 64 on AVX-512). Each chunk widens sbyte -> short via
    /// <see cref="Vector.Widen(Vector{sbyte}, out Vector{short}, out Vector{short})"/>
    /// then short -> int then int -> float, producing four
    /// <see cref="Vector{float}"/> accumulators that pair against four
    /// query vectors. JIT emits VPMOVSXBD + VCVTDQ2PS on AVX2.
    /// </para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotInt8(ReadOnlySpan<float> q, ReadOnlySpan<sbyte> k, float kScale, int headDim)
    {
        if (AdvSimd.IsSupported && headDim >= 16)
        {
            return DotInt8Arm(q, k, kScale, headDim);
        }
        if (Avx2.IsSupported && headDim >= 16)
        {
            return DotInt8Avx2(q, k, kScale, headDim);
        }
        return DotInt8Portable(q, k, kScale, headDim);
    }

    /// <summary>
    /// ARM NEON fast-path: process 16 sbytes per chunk via two-stage
    /// Vector128 sign-extend (SXTL/SXTL2 + SXTL/SXTL2) + SCVTF + FMLA.
    /// Vector128 abstractions emit native NEON ops on ARM hosts; on x86 they
    /// would emit 128-bit SSE which is suboptimal, so this path is gated on
    /// AdvSimd.IsSupported and only fires on ARM. The Avx2 path stays for
    /// x86 hosts (uses 256-bit ymm registers via Avx2.ConvertToVector256Int32).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotInt8Arm(ReadOnlySpan<float> q, ReadOnlySpan<sbyte> k, float kScale, int headDim)
    {
        const int chunk = 16; // 4 x Vector128<float>.Count
        var vectorEnd = headDim - (headDim % chunk);

        var acc = Vector128<float>.Zero;
        ref var qRef = ref MemoryMarshal.GetReference(q);
        ref var kRef = ref MemoryMarshal.GetReference(k);

        var i = 0;
        while (i < vectorEnd)
        {
            // Load 16 sbytes; sign-extend to 16 shorts (two Vector128<short>),
            // then to 16 ints (four Vector128<int>), then to 16 floats. NEON
            // emits SXTL / SXTL2 / SCVTF / FMLA via the Vector128 abstractions.
            var bytes = Vector128.LoadUnsafe(ref kRef, (nuint)i);
            var shortsLo = Vector128.WidenLower(bytes);    // SXTL
            var shortsHi = Vector128.WidenUpper(bytes);    // SXTL2
            var i0 = Vector128.WidenLower(shortsLo);
            var i1 = Vector128.WidenUpper(shortsLo);
            var i2 = Vector128.WidenLower(shortsHi);
            var i3 = Vector128.WidenUpper(shortsHi);
            var f0 = Vector128.ConvertToSingle(i0);
            var f1 = Vector128.ConvertToSingle(i1);
            var f2 = Vector128.ConvertToSingle(i2);
            var f3 = Vector128.ConvertToSingle(i3);

            var q0 = Vector128.LoadUnsafe(ref qRef, (nuint)i);
            var q1 = Vector128.LoadUnsafe(ref qRef, (nuint)(i + 4));
            var q2 = Vector128.LoadUnsafe(ref qRef, (nuint)(i + 8));
            var q3 = Vector128.LoadUnsafe(ref qRef, (nuint)(i + 12));

            // AdvSimd.FusedMultiplyAdd(addend, left, right) emits FMLA on ARM.
            acc = AdvSimd.FusedMultiplyAdd(acc, q0, f0);
            acc = AdvSimd.FusedMultiplyAdd(acc, q1, f1);
            acc = AdvSimd.FusedMultiplyAdd(acc, q2, f2);
            acc = AdvSimd.FusedMultiplyAdd(acc, q3, f3);

            i += chunk;
        }

        var sum = Vector128.Sum(acc);
        for (; i < headDim; i++)
        {
            sum += q[i] * (float)k[i];
        }

        return sum * kScale;
    }

    /// <summary>
    /// AVX2 fast-path: process 16 sbytes per chunk via two VPMOVSXBD ymm
    /// (sbyte -> int) + two VCVTDQ2PS ymm (int -> float) sequences. Skips
    /// the framework's three-stage Vector.Widen detour. Bench-confirmed to
    /// recover the L1-resident-decode regression vs the portable path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotInt8Avx2(ReadOnlySpan<float> q, ReadOnlySpan<sbyte> k, float kScale, int headDim)
    {
        const int chunk = 16; // 2 x Vector256<float>.Count
        var vectorEnd = headDim - (headDim % chunk);

        var acc = Vector256<float>.Zero;
        ref var qRef = ref MemoryMarshal.GetReference(q);
        ref var kRef = ref MemoryMarshal.GetReference(k);

        var i = 0;
        while (i < vectorEnd)
        {
            // Load 16 sbytes; sign-extend lower 8 -> 8 ints (VPMOVSXBD ymm),
            // shift right 8 bytes for upper half, repeat. Then VCVTDQ2PS each
            // and FMA against q.
            var bytes = Vector128.LoadUnsafe(ref kRef, (nuint)i);
            var bytesUpper = Sse2.ShiftRightLogical128BitLane(bytes.AsByte(), 8).AsSByte();

            var intsLo = Avx2.ConvertToVector256Int32(bytes);       // VPMOVSXBD ymm
            var intsHi = Avx2.ConvertToVector256Int32(bytesUpper);  // VPMOVSXBD ymm
            var floatsLo = Avx.ConvertToVector256Single(intsLo);    // VCVTDQ2PS
            var floatsHi = Avx.ConvertToVector256Single(intsHi);

            var qLo = Vector256.LoadUnsafe(ref qRef, (nuint)i);
            var qHi = Vector256.LoadUnsafe(ref qRef, (nuint)(i + 8));

            acc = Fma.IsSupported
                ? Fma.MultiplyAdd(qHi, floatsHi, Fma.MultiplyAdd(qLo, floatsLo, acc))
                : acc + qLo * floatsLo + qHi * floatsHi;

            i += chunk;
        }

        var sum = Vector256.Sum(acc);
        for (; i < headDim; i++)
        {
            sum += q[i] * (float)k[i];
        }

        return sum * kScale;
    }

    /// <summary>
    /// Portable path: framework Vector.Widen (sbyte -> short -> int -> float)
    /// for non-AVX2 hosts and headDim &lt; 16 tails.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotInt8Portable(ReadOnlySpan<float> q, ReadOnlySpan<sbyte> k, float kScale, int headDim)
    {
        var floatWidth = Vector<float>.Count;
        var byteWidth = Vector<sbyte>.Count;
        var vectorEnd = headDim - (headDim % byteWidth);

        var acc = Vector<float>.Zero;
        var i = 0;
        while (i < vectorEnd)
        {
            var bytes = new Vector<sbyte>(k.Slice(i, byteWidth));
            Vector.Widen(bytes, out var s0, out var s1);
            Vector.Widen(s0, out var i0, out var i1);
            Vector.Widen(s1, out var i2, out var i3);
            var f0 = Vector.ConvertToSingle(i0);
            var f1 = Vector.ConvertToSingle(i1);
            var f2 = Vector.ConvertToSingle(i2);
            var f3 = Vector.ConvertToSingle(i3);

            var q0 = new Vector<float>(q.Slice(i + 0 * floatWidth, floatWidth));
            var q1 = new Vector<float>(q.Slice(i + 1 * floatWidth, floatWidth));
            var q2 = new Vector<float>(q.Slice(i + 2 * floatWidth, floatWidth));
            var q3 = new Vector<float>(q.Slice(i + 3 * floatWidth, floatWidth));
            acc += q0 * f0 + q1 * f1 + q2 * f2 + q3 * f3;
            i += byteWidth;
        }

        var sum = Vector.Sum(acc);
        for (; i < headDim; i++)
        {
            sum += q[i] * (float)k[i];
        }

        return sum * kScale;
    }

    /// <summary>
    /// Section B (KV cache quantization) - KV3: <c>target += weight * (source * scale)</c>
    /// where <paramref name="source"/> is per-row absmax-quantised int8.
    /// Folds <paramref name="weight"/> and <paramref name="vScale"/> into one
    /// fp32 multiplier outside the SIMD body. Inner loop matches DotInt8's
    /// stride: <see cref="Vector{T}"/>.Count sbytes per chunk via four
    /// Widen+ConvertToSingle stages.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AccumulateWeightedInt8(
        Span<float> target,
        ReadOnlySpan<sbyte> source,
        float vScale,
        float weight,
        int headDim)
    {
        if (AdvSimd.IsSupported && headDim >= 16)
        {
            AccumulateWeightedInt8Arm(target, source, vScale, weight, headDim);
            return;
        }
        if (Avx2.IsSupported && headDim >= 16)
        {
            AccumulateWeightedInt8Avx2(target, source, vScale, weight, headDim);
            return;
        }
        AccumulateWeightedInt8Portable(target, source, vScale, weight, headDim);
    }

    /// <summary>
    /// ARM NEON path: <c>target += folded * dequant(source)</c> in 16-byte
    /// chunks producing 4 Vector128&lt;float&gt; FMLAs per chunk. Mirrors
    /// <see cref="DotInt8Arm"/>'s widen ladder. Activates whenever
    /// <see cref="AdvSimd.IsSupported"/> is true (any ARMv8 host).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateWeightedInt8Arm(
        Span<float> target,
        ReadOnlySpan<sbyte> source,
        float vScale,
        float weight,
        int headDim)
    {
        var folded = weight * vScale;
        var foldedVec = Vector128.Create(folded);
        const int chunk = 16;
        var vectorEnd = headDim - (headDim % chunk);

        ref var sRef = ref MemoryMarshal.GetReference(source);
        ref var tRef = ref MemoryMarshal.GetReference(target);

        var i = 0;
        while (i < vectorEnd)
        {
            var bytes = Vector128.LoadUnsafe(ref sRef, (nuint)i);
            var shortsLo = Vector128.WidenLower(bytes);
            var shortsHi = Vector128.WidenUpper(bytes);
            var f0 = Vector128.ConvertToSingle(Vector128.WidenLower(shortsLo));
            var f1 = Vector128.ConvertToSingle(Vector128.WidenUpper(shortsLo));
            var f2 = Vector128.ConvertToSingle(Vector128.WidenLower(shortsHi));
            var f3 = Vector128.ConvertToSingle(Vector128.WidenUpper(shortsHi));

            var t0 = Vector128.LoadUnsafe(ref tRef, (nuint)i);
            var t1 = Vector128.LoadUnsafe(ref tRef, (nuint)(i + 4));
            var t2 = Vector128.LoadUnsafe(ref tRef, (nuint)(i + 8));
            var t3 = Vector128.LoadUnsafe(ref tRef, (nuint)(i + 12));

            AdvSimd.FusedMultiplyAdd(t0, f0, foldedVec).StoreUnsafe(ref tRef, (nuint)i);
            AdvSimd.FusedMultiplyAdd(t1, f1, foldedVec).StoreUnsafe(ref tRef, (nuint)(i + 4));
            AdvSimd.FusedMultiplyAdd(t2, f2, foldedVec).StoreUnsafe(ref tRef, (nuint)(i + 8));
            AdvSimd.FusedMultiplyAdd(t3, f3, foldedVec).StoreUnsafe(ref tRef, (nuint)(i + 12));

            i += chunk;
        }

        for (; i < headDim; i++)
        {
            target[i] += folded * source[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateWeightedInt8Avx2(
        Span<float> target,
        ReadOnlySpan<sbyte> source,
        float vScale,
        float weight,
        int headDim)
    {
        var folded = weight * vScale;
        var foldedVec = Vector256.Create(folded);
        const int chunk = 16;
        var vectorEnd = headDim - (headDim % chunk);

        ref var sRef = ref MemoryMarshal.GetReference(source);
        ref var tRef = ref MemoryMarshal.GetReference(target);

        var i = 0;
        while (i < vectorEnd)
        {
            var bytes = Vector128.LoadUnsafe(ref sRef, (nuint)i);
            var bytesUpper = Sse2.ShiftRightLogical128BitLane(bytes.AsByte(), 8).AsSByte();

            var intsLo = Avx2.ConvertToVector256Int32(bytes);
            var intsHi = Avx2.ConvertToVector256Int32(bytesUpper);
            var floatsLo = Avx.ConvertToVector256Single(intsLo);
            var floatsHi = Avx.ConvertToVector256Single(intsHi);

            var tLo = Vector256.LoadUnsafe(ref tRef, (nuint)i);
            var tHi = Vector256.LoadUnsafe(ref tRef, (nuint)(i + 8));

            if (Fma.IsSupported)
            {
                Fma.MultiplyAdd(floatsLo, foldedVec, tLo).StoreUnsafe(ref tRef, (nuint)i);
                Fma.MultiplyAdd(floatsHi, foldedVec, tHi).StoreUnsafe(ref tRef, (nuint)(i + 8));
            }
            else
            {
                (tLo + floatsLo * foldedVec).StoreUnsafe(ref tRef, (nuint)i);
                (tHi + floatsHi * foldedVec).StoreUnsafe(ref tRef, (nuint)(i + 8));
            }
            i += chunk;
        }

        for (; i < headDim; i++)
        {
            target[i] += folded * source[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateWeightedInt8Portable(
        Span<float> target,
        ReadOnlySpan<sbyte> source,
        float vScale,
        float weight,
        int headDim)
    {
        var folded = weight * vScale;
        var floatWidth = Vector<float>.Count;
        var byteWidth = Vector<sbyte>.Count;
        var vectorEnd = headDim - (headDim % byteWidth);

        var scale = new Vector<float>(folded);
        var i = 0;
        while (i < vectorEnd)
        {
            var bytes = new Vector<sbyte>(source.Slice(i, byteWidth));
            Vector.Widen(bytes, out var s0, out var s1);
            Vector.Widen(s0, out var i0, out var i1);
            Vector.Widen(s1, out var i2, out var i3);
            var f0 = Vector.ConvertToSingle(i0);
            var f1 = Vector.ConvertToSingle(i1);
            var f2 = Vector.ConvertToSingle(i2);
            var f3 = Vector.ConvertToSingle(i3);

            var t0 = new Vector<float>(target.Slice(i + 0 * floatWidth, floatWidth));
            var t1 = new Vector<float>(target.Slice(i + 1 * floatWidth, floatWidth));
            var t2 = new Vector<float>(target.Slice(i + 2 * floatWidth, floatWidth));
            var t3 = new Vector<float>(target.Slice(i + 3 * floatWidth, floatWidth));
            (t0 + f0 * scale).CopyTo(target.Slice(i + 0 * floatWidth, floatWidth));
            (t1 + f1 * scale).CopyTo(target.Slice(i + 1 * floatWidth, floatWidth));
            (t2 + f2 * scale).CopyTo(target.Slice(i + 2 * floatWidth, floatWidth));
            (t3 + f3 * scale).CopyTo(target.Slice(i + 3 * floatWidth, floatWidth));
            i += byteWidth;
        }

        for (; i < headDim; i++)
        {
            target[i] += folded * source[i];
        }
    }

    /// <summary>
    /// In-place scalar multiply for the accumulator correction step of online
    /// softmax. <c>target *= scalar</c> over the first <paramref name="headDim"/>
    /// lanes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ScaleInPlace(Span<float> target, float scalar, int headDim)
    {
        var width = Vector<float>.Count;
        var vectorEnd = headDim - (headDim % width);

        var scale = new Vector<float>(scalar);
        var i = 0;
        while (i < vectorEnd)
        {
            var tv = new Vector<float>(target.Slice(i, width));
            (tv * scale).CopyTo(target.Slice(i, width));
            i += width;
        }

        for (; i < headDim; i++)
        {
            target[i] *= scalar;
        }
    }

    /// <summary>
    /// Online-softmax attention for a single query row against a contiguous
    /// key/value prefix. Streams QK/softmax/AV in one pass so no
    /// [headCount, 1, pastLength] weight matrix is materialised. Called per
    /// head by <see cref="FlashAttention.ForwardDecode"/>.
    /// </summary>
    /// <param name="query">Query vector slice of length <paramref name="headDim"/>.</param>
    /// <param name="keysBase">Flat span over K cache rows [0, pastLength), stride <paramref name="rowStride"/>.</param>
    /// <param name="valuesBase">Flat span over V cache rows, same stride.</param>
    /// <param name="output">Output buffer of length <paramref name="headDim"/>. Cleared before accumulation.</param>
    /// <param name="headOffsetWithinRow">Column offset to this head inside each kv row.</param>
    /// <param name="rowStride">kvDim = bytes-per-row for K and V.</param>
    /// <param name="headDim">Per-head vector size.</param>
    /// <param name="pastLength">Number of K/V rows to attend over (inclusive of the just-appended row).</param>
    /// <param name="scale">Attention scale factor, typically 1/sqrt(headDim).</param>
    public static void OnlineSoftmaxAttendSingleRow(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> keysBase,
        ReadOnlySpan<float> valuesBase,
        Span<float> output,
        int headOffsetWithinRow,
        int rowStride,
        int headDim,
        int pastLength,
        float scale)
    {
        output.Slice(0, headDim).Clear();
        if (pastLength <= 0)
        {
            return;
        }

        var maxScore = float.NegativeInfinity;
        var partition = 0f;

        for (var source = 0; source < pastLength; source++)
        {
            var rowStart = source * rowStride + headOffsetWithinRow;
            var kSlice = keysBase.Slice(rowStart, headDim);
            var score = Dot(query, kSlice, headDim) * scale;

            float correction;
            float weight;
            if (score > maxScore)
            {
                correction = maxScore == float.NegativeInfinity ? 0f : MathF.Exp(maxScore - score);
                partition = partition * correction + 1f;
                ScaleInPlace(output, correction, headDim);
                maxScore = score;
                weight = 1f;
            }
            else
            {
                weight = MathF.Exp(score - maxScore);
                partition += weight;
            }

            var vSlice = valuesBase.Slice(rowStart, headDim);
            AccumulateWeighted(output, vSlice, weight, headDim);
        }

        if (partition > 0f)
        {
            ScaleInPlace(output, 1f / partition, headDim);
        }
    }
}

/// <summary>
/// Fused flash-style attention for the decode case (single query row). Avoids
/// allocating the [headCount, 1, pastLength] attention-weights tensor that the
/// standard path produces.
/// </summary>
internal static class FlashAttention
{
    /// <summary>
    /// Attend a single query row against <paramref name="pastLength"/> cached
    /// K/V rows using online softmax per head. Writes the weighted-value sum
    /// for every head into <paramref name="attendedOutput"/>.
    /// </summary>
    /// <param name="query">Row-major <c>[1, dim]</c> (interpreted as flat headCount*headDim).</param>
    /// <param name="cacheK">KV cache keys as flat span [capacity * kvDim].</param>
    /// <param name="cacheV">KV cache values as flat span [capacity * kvDim].</param>
    /// <param name="attendedOutput">Flat span [dim] to receive output. Cleared then filled.</param>
    /// <param name="headCount">Number of query heads.</param>
    /// <param name="kvHeadCount">Number of key/value heads (MHA: == headCount, GQA: less).</param>
    /// <param name="headDim">Per-head vector size.</param>
    /// <param name="pastLength">Number of cached rows to attend over.</param>
    /// <param name="scale">Attention scale factor.</param>
    public static void ForwardDecode(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> cacheK,
        ReadOnlySpan<float> cacheV,
        Span<float> attendedOutput,
        int headCount,
        int kvHeadCount,
        int headDim,
        int pastLength,
        float scale)
    {
        var groupSize = headCount / kvHeadCount;
        var kvDim = kvHeadCount * headDim;

        attendedOutput.Clear();

        for (var head = 0; head < headCount; head++)
        {
            var kvHead = head / groupSize;
            var qOffset = head * headDim;
            var kvOffset = kvHead * headDim;
            var qSlice = query.Slice(qOffset, headDim);
            var outSlice = attendedOutput.Slice(qOffset, headDim);

            AttentionMath.OnlineSoftmaxAttendSingleRow(
                qSlice, cacheK, cacheV, outSlice,
                kvOffset, kvDim, headDim, pastLength, scale);
        }
    }

    /// <summary>
    /// Section B (KV cache quantization) - KV4: int8 K/V variant of
    /// <see cref="ForwardDecode"/>. Per-row absmax scales for K and V are
    /// applied lazily inside the per-head online-softmax loop so the dot
    /// stays in fp32 and the value accumulation pays one fold-multiply.
    /// </summary>
    public static void ForwardDecodeInt8(
        ReadOnlySpan<float> query,
        ReadOnlySpan<sbyte> cacheK,
        ReadOnlySpan<float> cacheKScale,
        ReadOnlySpan<sbyte> cacheV,
        ReadOnlySpan<float> cacheVScale,
        Span<float> attendedOutput,
        int headCount,
        int kvHeadCount,
        int headDim,
        int pastLength,
        float scale)
    {
        var groupSize = headCount / kvHeadCount;
        var kvDim = kvHeadCount * headDim;

        attendedOutput.Clear();
        if (pastLength <= 0)
        {
            return;
        }

        for (var head = 0; head < headCount; head++)
        {
            var kvHead = head / groupSize;
            var qOffset = head * headDim;
            var kvOffset = kvHead * headDim;
            var qSlice = query.Slice(qOffset, headDim);
            var outSlice = attendedOutput.Slice(qOffset, headDim);

            OnlineSoftmaxAttendSingleRowInt8(
                qSlice,
                cacheK, cacheKScale,
                cacheV, cacheVScale,
                outSlice,
                kvOffset, kvDim, headDim, pastLength, scale);
        }
    }

    private static void OnlineSoftmaxAttendSingleRowInt8(
        ReadOnlySpan<float> query,
        ReadOnlySpan<sbyte> keysBase,
        ReadOnlySpan<float> kScales,
        ReadOnlySpan<sbyte> valuesBase,
        ReadOnlySpan<float> vScales,
        Span<float> output,
        int headOffsetWithinRow,
        int rowStride,
        int headDim,
        int pastLength,
        float scale)
    {
        output.Slice(0, headDim).Clear();

        var maxScore = float.NegativeInfinity;
        var partition = 0f;

        for (var source = 0; source < pastLength; source++)
        {
            var rowStart = source * rowStride + headOffsetWithinRow;
            var kSlice = keysBase.Slice(rowStart, headDim);
            var vSlice = valuesBase.Slice(rowStart, headDim);
            var score = AttentionMath.DotInt8(query, kSlice, kScales[source], headDim) * scale;

            float correction;
            float weight;
            if (score > maxScore)
            {
                correction = maxScore == float.NegativeInfinity ? 0f : MathF.Exp(maxScore - score);
                partition = partition * correction + 1f;
                AttentionMath.ScaleInPlace(output, correction, headDim);
                maxScore = score;
                weight = 1f;
            }
            else
            {
                weight = MathF.Exp(score - maxScore);
                partition += weight;
            }

            AttentionMath.AccumulateWeightedInt8(output, vSlice, vScales[source], weight, headDim);
        }

        if (partition > 0f)
        {
            AttentionMath.ScaleInPlace(output, 1f / partition, headDim);
        }
    }
}
