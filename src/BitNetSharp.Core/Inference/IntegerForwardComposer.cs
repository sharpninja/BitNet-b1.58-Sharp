using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Core.Inference;

/// <summary>
/// Composes the integer-inference primitives (I3 RmsNorm, I4 RoPE,
/// I5 BitLinear.ForwardInt32, I6 softmax, I7 SwiGLU, I8 residual adder)
/// into a <see cref="BitNetLayer"/>-equivalent forward pass while keeping the
/// float[,] public surface. Caller is the bridge between the existing V1
/// primitive-composition test and the production forward hot path: downstream
/// we can grow <see cref="Models.BitNetTransformer"/> overloads that route
/// every layer through this composer without touching the float training
/// path.
///
/// Contract (verified by IntegerForwardComposerTests): output stays within the
/// integer-precision floor (5e-2 per element) of
/// <see cref="BitNetLayer.Forward(float[,])"/> for MHA and GQA configurations.
/// </summary>
public static class IntegerForwardComposer
{
    /// <summary>
    /// Runs a single <see cref="BitNetLayer"/> forward pass through the
    /// integer-inference primitives. Mirrors the prefill/training-style
    /// overload <see cref="BitNetLayer.Forward(float[,])"/>: per-head causal
    /// attention, no KV cache.
    /// </summary>
    public static float[,] ForwardFullSeq(BitNetLayer layer, float[,] input)
    {
        ArgumentNullException.ThrowIfNull(layer);
        ArgumentNullException.ThrowIfNull(input);

        var config = layer.Config;
        int dim = config.Dimension;
        int headDim = config.HeadDimension;
        int headCount = config.HeadCount;
        int kvHeadCount = config.KvHeadCount;
        int seqLen = input.GetLength(0);
        float attentionScale = 1f / MathF.Sqrt(headDim);

        if (input.GetLength(1) != dim)
        {
            throw new ArgumentException(
                $"Expected input dimension {dim}, received {input.GetLength(1)}.",
                nameof(input));
        }

        int ropeMaxSeq = Math.Max(seqLen, 128);

        // RMSNorm -> Q/K/V projections (int32 accumulators) -> RoPE on dequantised
        // Q/K (integer rotor uses Q1.15 sin/cos but keeps float tensor surface).
        // Primitives cached per layer so the RoPE sin/cos table (and the
        // RmsNorm scale import) are paid once per layer, not per call.
        var (intAttnRms, intFfnRms, intRope) = IntegerLayerPrimitiveCache.Get(layer, ropeMaxSeq);
        var intSoftmax = IntegerLayerPrimitiveCache.Softmax;
        var intSwiGLU = IntegerLayerPrimitiveCache.SwiGLU;

        float[,] normed = intAttnRms.Forward(input);
        var quantAttn = QuantizedActivationBlock.FromFloat(normed);

        int kvDim = kvHeadCount * headDim;
        float[,] queries = layer.Attention.QueryProjection.ForwardInt32(quantAttn).ToFloat();
        float[,] keys = layer.Attention.KeyProjection.ForwardInt32(quantAttn).ToFloat();
        float[,] values = layer.Attention.ValueProjection.ForwardInt32(quantAttn).ToFloat();

        intRope.ApplyInPlace(queries, headCount);
        // For GQA, keys/values carry kvHeadCount head-sized rotor slots.
        intRope.ApplyInPlace(keys, kvHeadCount);

        // Per-head causal scaled dot-product using IntegerSoftmax on the
        // materialised logits. GQA maps each query head to a KV head by
        // integer division (queryHead / headGroupSize). Logits buffer is
        // sized to the maximum causal length (seqLen) and reused across every
        // (head, target) tuple to keep the hot loop allocation-free.
        int headGroupSize = headCount / kvHeadCount;
        var attended = new float[seqLen, dim];
        var logitsBuf = new float[seqLen];
        for (int queryHead = 0; queryHead < headCount; queryHead++)
        {
            int kvHead = queryHead / headGroupSize;
            int queryOffset = queryHead * headDim;
            int kvOffset = kvHead * headDim;

            for (int target = 0; target < seqLen; target++)
            {
                // Score only up to the causal cursor.
                int causalLen = target + 1;
                var logitsSpan = logitsBuf.AsSpan(0, causalLen);
                for (int source = 0; source < causalLen; source++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        dot += queries[target, queryOffset + d] * keys[source, kvOffset + d];
                    }
                    logitsSpan[source] = dot * attentionScale;
                }

                intSoftmax.ApplyRowInPlace(logitsSpan, logitsSpan);
                for (int source = 0; source < causalLen; source++)
                {
                    float w = logitsSpan[source];
                    for (int d = 0; d < headDim; d++)
                    {
                        attended[target, queryOffset + d] += w * values[source, kvOffset + d];
                    }
                }
            }
        }

        // Output projection (int32 -> float) + residual.
        var quantAttended = QuantizedActivationBlock.FromFloat(attended);
        float[,] attnProjected = layer.Attention.OutputProjection.ForwardInt32(quantAttended).ToFloat();
        float[,] residual = TensorMath.Add(input, attnProjected);

        // FFN: RmsNorm -> gate/up (int32) -> integer SwiGLU -> down (int32) -> residual.
        float[,] normedFfn = intFfnRms.Forward(residual);
        var quantFfn = QuantizedActivationBlock.FromFloat(normedFfn);
        float[,] gate = layer.FeedForward.GateProjection.ForwardInt32(quantFfn).ToFloat();
        float[,] up = layer.FeedForward.UpProjection.ForwardInt32(quantFfn).ToFloat();
        float[,] swish = intSwiGLU.ApplyToFloat(gate, up);
        var quantDown = QuantizedActivationBlock.FromFloat(swish);
        float[,] down = layer.FeedForward.DownProjection.ForwardInt32(quantDown).ToFloat();

        return TensorMath.Add(residual, down);
    }

    /// <summary>
    /// Cache-aware integer forward: mirrors
    /// <see cref="BitNetLayer.Forward(float[,], LayerKvCache, int)"/> but runs
    /// every op through the integer primitives. Handles both single-row
    /// decode (the per-token hot path) and multi-row prefill-with-cache
    /// (warm path). Writes new K/V rows into <paramref name="cache"/> starting
    /// at <paramref name="positionOffset"/> and scores each target row against
    /// the causal prefix in the cache.
    /// </summary>
    public static float[,] ForwardWithCache(
        BitNetLayer layer,
        float[,] input,
        LayerKvCache cache,
        int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(layer);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentOutOfRangeException.ThrowIfNegative(positionOffset);

        var config = layer.Config;
        int dim = config.Dimension;
        int headDim = config.HeadDimension;
        int headCount = config.HeadCount;
        int kvHeadCount = config.KvHeadCount;
        int kvDim = kvHeadCount * headDim;
        int newRows = input.GetLength(0);
        float attentionScale = 1f / MathF.Sqrt(headDim);

        if (input.GetLength(1) != dim)
        {
            throw new ArgumentException(
                $"Expected input dimension {dim}, received {input.GetLength(1)}.",
                nameof(input));
        }
        if (cache.KvDimension != kvDim)
        {
            throw new ArgumentException(
                $"Cache kv dimension {cache.KvDimension} does not match expected {kvDim}.",
                nameof(cache));
        }

        int totalLength = positionOffset + newRows;
        if (totalLength > cache.Capacity)
        {
            throw new ArgumentException(
                $"Cache capacity {cache.Capacity} too small for total length {totalLength}.",
                nameof(cache));
        }

        int ropeMaxSeq = Math.Max(totalLength, 128);
        var (intAttnRms, intFfnRms, intRope) = IntegerLayerPrimitiveCache.Get(layer, ropeMaxSeq);
        var intSoftmax = IntegerLayerPrimitiveCache.Softmax;
        var intSwiGLU = IntegerLayerPrimitiveCache.SwiGLU;

        float[,] normed = intAttnRms.Forward(input);
        var quantAttn = QuantizedActivationBlock.FromFloat(normed);

        float[,] queries = layer.Attention.QueryProjection.ForwardInt32(quantAttn).ToFloat();
        float[,] newKeys = layer.Attention.KeyProjection.ForwardInt32(quantAttn).ToFloat();
        float[,] newValues = layer.Attention.ValueProjection.ForwardInt32(quantAttn).ToFloat();

        intRope.ApplyInPlace(queries, headCount, positionOffset);
        intRope.ApplyInPlace(newKeys, kvHeadCount, positionOffset);

        // Write new K/V into the cache at [positionOffset ... positionOffset + newRows).
        for (int row = 0; row < newRows; row++)
        {
            for (int c = 0; c < kvDim; c++)
            {
                cache.K[positionOffset + row, c] = newKeys[row, c];
                cache.V[positionOffset + row, c] = newValues[row, c];
            }
        }

        // Per-head causal attention against the cache prefix. Single logits
        // buffer sized to the deepest causal length (totalLength) is reused
        // across every (head, target) tuple in this call.
        int headGroupSize = headCount / kvHeadCount;
        var attended = new float[newRows, dim];
        var logitsBuf = new float[totalLength];
        for (int queryHead = 0; queryHead < headCount; queryHead++)
        {
            int kvHead = queryHead / headGroupSize;
            int queryOffset = queryHead * headDim;
            int cacheOffset = kvHead * headDim;

            for (int targetRow = 0; targetRow < newRows; targetRow++)
            {
                int causalLen = positionOffset + targetRow + 1;
                var logitsSpan = logitsBuf.AsSpan(0, causalLen);
                for (int source = 0; source < causalLen; source++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        dot += queries[targetRow, queryOffset + d] * cache.K[source, cacheOffset + d];
                    }
                    logitsSpan[source] = dot * attentionScale;
                }

                intSoftmax.ApplyRowInPlace(logitsSpan, logitsSpan);
                for (int source = 0; source < causalLen; source++)
                {
                    float w = logitsSpan[source];
                    for (int d = 0; d < headDim; d++)
                    {
                        attended[targetRow, queryOffset + d] += w * cache.V[source, cacheOffset + d];
                    }
                }
            }
        }

        var quantAttended = QuantizedActivationBlock.FromFloat(attended);
        float[,] attnProjected = layer.Attention.OutputProjection.ForwardInt32(quantAttended).ToFloat();
        float[,] residual = TensorMath.Add(input, attnProjected);

        float[,] normedFfn = intFfnRms.Forward(residual);
        var quantFfn = QuantizedActivationBlock.FromFloat(normedFfn);
        float[,] gate = layer.FeedForward.GateProjection.ForwardInt32(quantFfn).ToFloat();
        float[,] up = layer.FeedForward.UpProjection.ForwardInt32(quantFfn).ToFloat();
        float[,] swish = intSwiGLU.ApplyToFloat(gate, up);
        var quantDown = QuantizedActivationBlock.FromFloat(swish);
        float[,] down = layer.FeedForward.DownProjection.ForwardInt32(quantDown).ToFloat();

        return TensorMath.Add(residual, down);
    }
}
