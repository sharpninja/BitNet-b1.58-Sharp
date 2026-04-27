using System.Runtime.InteropServices;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Core.Layers;

public sealed class MultiHeadAttention : AttentionModule
{
    private readonly RotaryPositionEmbedding _rotaryPositionEmbedding;
    private readonly float _attentionScale;

    public MultiHeadAttention(BitNetConfig config, Random random)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(random);

        Config = config;
        QueryProjection = ParameterInitializer.CreateBitLinear(new BitLinearConfig(config.Dimension, config.Dimension), random);
        KeyProjection = ParameterInitializer.CreateBitLinear(new BitLinearConfig(config.Dimension, config.Dimension), random);
        ValueProjection = ParameterInitializer.CreateBitLinear(new BitLinearConfig(config.Dimension, config.Dimension), random);
        OutputProjection = ParameterInitializer.CreateBitLinear(new BitLinearConfig(config.Dimension, config.Dimension), random);
        _rotaryPositionEmbedding = new RotaryPositionEmbedding(config.HeadDimension);
        _attentionScale = 1f / MathF.Sqrt(config.HeadDimension);
    }

    public BitNetConfig Config { get; }

    public override BitLinear QueryProjection { get; }

    public override BitLinear KeyProjection { get; }

    public override BitLinear ValueProjection { get; }

    public override BitLinear OutputProjection { get; }

    public override bool UsesRotaryPositionEmbedding => true;

    public override bool AppliesRotaryPositionEmbeddingToQueriesAndKeysOnly => true;

    public override bool UsesCausalAttentionMask => true;

    public override float AttentionScale => _attentionScale;

    public override long EstimateResidentParameterBytes() =>
        QueryProjection.EstimateResidentParameterBytes()
        + KeyProjection.EstimateResidentParameterBytes()
        + ValueProjection.EstimateResidentParameterBytes()
        + OutputProjection.EstimateResidentParameterBytes();

    // Cached for backward pass
    private float[,]? _cachedQueries;
    private float[,]? _cachedKeys;
    private float[,]? _cachedValues;
    private float[,,]? _cachedAttentionWeights; // [head, target, source]

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.GetLength(1) != Config.Dimension)
        {
            throw new ArgumentException($"Expected input dimension {Config.Dimension}, but received {input.GetLength(1)}.", nameof(input));
        }

        var sharedQuant = QuantizedActivationBlock.FromFloat(input);
        var queries = QueryProjection.ForwardQuantized(sharedQuant, input);
        var keys = KeyProjection.ForwardQuantized(sharedQuant, input);
        var values = ValueProjection.ForwardQuantized(sharedQuant, input);

        _rotaryPositionEmbedding.ApplyInPlace(queries, Config.HeadCount);
        _rotaryPositionEmbedding.ApplyInPlace(keys, Config.HeadCount);

        var seqLen = input.GetLength(0);
        var attended = new float[seqLen, Config.Dimension];
        var attentionWeights = new float[Config.HeadCount, seqLen, seqLen];

        for (var head = 0; head < Config.HeadCount; head++)
        {
            var scores = new float[seqLen];
            ApplyHeadWithCache(attended, queries, keys, values, head, scores, attentionWeights);
        }

        _cachedQueries = queries;
        _cachedKeys = keys;
        _cachedValues = values;
        _cachedAttentionWeights = attentionWeights;

        return OutputProjection.Forward(attended);
    }

    public override float[,] Forward(float[,] input, LayerKvCache cache, int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentOutOfRangeException.ThrowIfNegative(positionOffset);

        if (input.GetLength(1) != Config.Dimension)
        {
            throw new ArgumentException(
                $"Expected input dimension {Config.Dimension}, received {input.GetLength(1)}.", nameof(input));
        }

        var newRows = input.GetLength(0);
        var dim = Config.Dimension;
        var headDim = Config.HeadDimension;
        var headCount = Config.HeadCount;

        if (cache.KvDimension != dim)
        {
            throw new ArgumentException(
                $"Cache kv dimension {cache.KvDimension} does not match expected {dim}.", nameof(cache));
        }

        var totalLength = positionOffset + newRows;
        if (totalLength > cache.Capacity)
        {
            throw new ArgumentException(
                $"Cache capacity {cache.Capacity} too small for total length {totalLength}.", nameof(cache));
        }

        var sharedQuant = QuantizedActivationBlock.FromFloat(input);
        var queries = QueryProjection.ForwardQuantized(sharedQuant);
        var newKeys = KeyProjection.ForwardQuantized(sharedQuant);
        var newValues = ValueProjection.ForwardQuantized(sharedQuant);

        _rotaryPositionEmbedding.ApplyInPlace(queries, headCount, positionOffset);
        _rotaryPositionEmbedding.ApplyInPlace(newKeys, headCount, positionOffset);

        for (var row = 0; row < newRows; row++)
        {
            for (var col = 0; col < dim; col++)
            {
                cache.K[positionOffset + row, col] = newKeys[row, col];
                cache.V[positionOffset + row, col] = newValues[row, col];
            }
        }

        var attended = new float[newRows, dim];
        var attendedFlat = AttentionMath.AsFlatSpan(attended);
        var queriesFlat = AttentionMath.AsFlatSpan(queries);
        var cacheKFlat = AttentionMath.AsFlatSpan(cache.K);
        var cacheVFlat = AttentionMath.AsFlatSpan(cache.V);

        for (var head = 0; head < headCount; head++)
        {
            var headOffset = head * headDim;

            for (var targetRow = 0; targetRow < newRows; targetRow++)
            {
                var absoluteTarget = positionOffset + targetRow;
                var scoreCount = absoluteTarget + 1;
                var scores = new float[scoreCount];
                var maxScore = float.NegativeInfinity;
                var qSlice = queriesFlat.Slice(targetRow * dim + headOffset, headDim);

                for (var source = 0; source < scoreCount; source++)
                {
                    var kSlice = cacheKFlat.Slice(source * dim + headOffset, headDim);
                    var score = AttentionMath.Dot(qSlice, kSlice, headDim) * _attentionScale;
                    scores[source] = score;
                    if (score > maxScore)
                    {
                        maxScore = score;
                    }
                }

                var partition = 0f;
                for (var source = 0; source < scoreCount; source++)
                {
                    scores[source] = MathF.Exp(scores[source] - maxScore);
                    partition += scores[source];
                }

                if (partition <= 0f)
                {
                    continue;
                }

                var attendedSlice = attendedFlat.Slice(targetRow * dim + headOffset, headDim);
                var invPartition = 1f / partition;
                for (var source = 0; source < scoreCount; source++)
                {
                    var weight = scores[source] * invPartition;
                    var vSlice = cacheVFlat.Slice(source * dim + headOffset, headDim);
                    AttentionMath.AccumulateWeighted(attendedSlice, vSlice, weight, headDim);
                }
            }
        }

        return OutputProjection.Forward(attended);
    }

    public override float[,] Forward(float[,] input, QuantizedKvLayerCache cache, int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentOutOfRangeException.ThrowIfNegative(positionOffset);
        if (input.GetLength(1) != Config.Dimension)
        {
            throw new ArgumentException(
                $"Expected input dimension {Config.Dimension}, received {input.GetLength(1)}.", nameof(input));
        }

        var newRows = input.GetLength(0);
        var dim = Config.Dimension;
        var headDim = Config.HeadDimension;
        var headCount = Config.HeadCount;

        if (cache.KvDimension != dim)
        {
            throw new ArgumentException(
                $"Cache kv dimension {cache.KvDimension} does not match expected {dim}.", nameof(cache));
        }
        var totalLength = positionOffset + newRows;
        if (totalLength > cache.Capacity)
        {
            throw new ArgumentException(
                $"Cache capacity {cache.Capacity} too small for total length {totalLength}.", nameof(cache));
        }

        var sharedQuant = QuantizedActivationBlock.FromFloat(input);
        var queries = QueryProjection.ForwardQuantized(sharedQuant);
        var newKeys = KeyProjection.ForwardQuantized(sharedQuant);
        var newValues = ValueProjection.ForwardQuantized(sharedQuant);

        _rotaryPositionEmbedding.ApplyInPlace(queries, headCount, positionOffset);
        _rotaryPositionEmbedding.ApplyInPlace(newKeys, headCount, positionOffset);

        // KV5: write through IKvCache so int8 quantizer runs once per row.
        var newKeysFlat = AttentionMath.AsFlatSpan(newKeys);
        var newValuesFlat = AttentionMath.AsFlatSpan(newValues);
        for (var row = 0; row < newRows; row++)
        {
            cache.WriteKRow(positionOffset + row, newKeysFlat.Slice(row * dim, dim));
            cache.WriteVRow(positionOffset + row, newValuesFlat.Slice(row * dim, dim));
        }

        // Prefill (newRows > 1) dequantises the cache prefix once per call
        // and reuses the existing fp32 attention math; this keeps the
        // per-row absmax loss bounded the same way as decode and reuses the
        // already-validated SIMD path. Decode (newRows == 1) bypasses this
        // and goes straight to ForwardFlashDecode + ForwardDecodeInt8.
        var attended = new float[newRows, dim];
        var attendedFlat = AttentionMath.AsFlatSpan(attended);
        var queriesFlat = AttentionMath.AsFlatSpan(queries);

        var totalRows = positionOffset + newRows;
        var kFloat = new float[totalRows * dim];
        var vFloat = new float[totalRows * dim];
        var kFloatSpan = kFloat.AsSpan();
        var vFloatSpan = vFloat.AsSpan();
        for (var r = 0; r < totalRows; r++)
        {
            cache.DequantizeKRow(r, kFloatSpan.Slice(r * dim, dim));
            cache.DequantizeVRow(r, vFloatSpan.Slice(r * dim, dim));
        }

        for (var head = 0; head < headCount; head++)
        {
            var headOffset = head * headDim;
            for (var targetRow = 0; targetRow < newRows; targetRow++)
            {
                var absoluteTarget = positionOffset + targetRow;
                var scoreCount = absoluteTarget + 1;
                var scores = new float[scoreCount];
                var maxScore = float.NegativeInfinity;
                var qSlice = queriesFlat.Slice(targetRow * dim + headOffset, headDim);

                for (var source = 0; source < scoreCount; source++)
                {
                    var kSlice = kFloatSpan.Slice(source * dim + headOffset, headDim);
                    var score = AttentionMath.Dot(qSlice, kSlice, headDim) * _attentionScale;
                    scores[source] = score;
                    if (score > maxScore)
                    {
                        maxScore = score;
                    }
                }

                var partition = 0f;
                for (var source = 0; source < scoreCount; source++)
                {
                    scores[source] = MathF.Exp(scores[source] - maxScore);
                    partition += scores[source];
                }

                if (partition <= 0f)
                {
                    continue;
                }

                var attendedSlice = attendedFlat.Slice(targetRow * dim + headOffset, headDim);
                var invPartition = 1f / partition;
                for (var source = 0; source < scoreCount; source++)
                {
                    var weight = scores[source] * invPartition;
                    var vSlice = vFloatSpan.Slice(source * dim + headOffset, headDim);
                    AttentionMath.AccumulateWeighted(attendedSlice, vSlice, weight, headDim);
                }
            }
        }

        return OutputProjection.Forward(attended);
    }

    public override float[,] ForwardFlashDecode(float[,] input, QuantizedKvLayerCache cache, int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentOutOfRangeException.ThrowIfNegative(positionOffset);
        if (input.GetLength(0) != 1)
        {
            throw new ArgumentException("Flash decode expects exactly one query row.", nameof(input));
        }
        if (input.GetLength(1) != Config.Dimension)
        {
            throw new ArgumentException(
                $"Expected input dimension {Config.Dimension}, received {input.GetLength(1)}.", nameof(input));
        }

        var dim = Config.Dimension;
        var headDim = Config.HeadDimension;
        var headCount = Config.HeadCount;

        if (cache.KvDimension != dim)
        {
            throw new ArgumentException(
                $"Cache kv dimension {cache.KvDimension} does not match expected {dim}.", nameof(cache));
        }
        var totalLength = positionOffset + 1;
        if (totalLength > cache.Capacity)
        {
            throw new ArgumentException(
                $"Cache capacity {cache.Capacity} too small for total length {totalLength}.", nameof(cache));
        }

        var sharedQuant = QuantizedActivationBlock.FromFloat(input);
        var queries = QueryProjection.ForwardQuantized(sharedQuant);
        var newKeys = KeyProjection.ForwardQuantized(sharedQuant);
        var newValues = ValueProjection.ForwardQuantized(sharedQuant);

        _rotaryPositionEmbedding.ApplyInPlace(queries, headCount, positionOffset);
        _rotaryPositionEmbedding.ApplyInPlace(newKeys, headCount, positionOffset);

        cache.WriteKRow(positionOffset, AttentionMath.AsFlatSpan(newKeys).Slice(0, dim));
        cache.WriteVRow(positionOffset, AttentionMath.AsFlatSpan(newValues).Slice(0, dim));

        var attended = new float[1, dim];
        var attendedFlat = AttentionMath.AsFlatSpan(attended);
        var queriesFlat = AttentionMath.AsFlatSpan(queries);
        ref var kFirst = ref System.Runtime.CompilerServices.Unsafe.As<byte, sbyte>(
            ref MemoryMarshal.GetArrayDataReference(cache.K));
        ref var vFirst = ref System.Runtime.CompilerServices.Unsafe.As<byte, sbyte>(
            ref MemoryMarshal.GetArrayDataReference(cache.V));
        var cacheKFlat = MemoryMarshal.CreateSpan(ref kFirst, cache.K.Length);
        var cacheVFlat = MemoryMarshal.CreateSpan(ref vFirst, cache.V.Length);

        FlashAttention.ForwardDecodeInt8(
            queriesFlat,
            cacheKFlat,
            cache.KScale.AsSpan(0, totalLength),
            cacheVFlat,
            cache.VScale.AsSpan(0, totalLength),
            attendedFlat,
            headCount,
            headCount,
            headDim,
            totalLength,
            _attentionScale);

        return OutputProjection.Forward(attended);
    }

    public override float[,] ForwardFlashDecode(float[,] input, LayerKvCache cache, int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentOutOfRangeException.ThrowIfNegative(positionOffset);

        if (input.GetLength(0) != 1)
        {
            throw new ArgumentException("Flash decode expects exactly one query row.", nameof(input));
        }
        if (input.GetLength(1) != Config.Dimension)
        {
            throw new ArgumentException(
                $"Expected input dimension {Config.Dimension}, received {input.GetLength(1)}.", nameof(input));
        }

        var dim = Config.Dimension;
        var headDim = Config.HeadDimension;
        var headCount = Config.HeadCount;

        if (cache.KvDimension != dim)
        {
            throw new ArgumentException(
                $"Cache kv dimension {cache.KvDimension} does not match expected {dim}.", nameof(cache));
        }

        var totalLength = positionOffset + 1;
        if (totalLength > cache.Capacity)
        {
            throw new ArgumentException(
                $"Cache capacity {cache.Capacity} too small for total length {totalLength}.", nameof(cache));
        }

        var sharedQuant = QuantizedActivationBlock.FromFloat(input);
        var queries = QueryProjection.ForwardQuantized(sharedQuant);
        var newKeys = KeyProjection.ForwardQuantized(sharedQuant);
        var newValues = ValueProjection.ForwardQuantized(sharedQuant);

        _rotaryPositionEmbedding.ApplyInPlace(queries, headCount, positionOffset);
        _rotaryPositionEmbedding.ApplyInPlace(newKeys, headCount, positionOffset);

        for (var col = 0; col < dim; col++)
        {
            cache.K[positionOffset, col] = newKeys[0, col];
            cache.V[positionOffset, col] = newValues[0, col];
        }

        var attended = new float[1, dim];
        var attendedFlat = AttentionMath.AsFlatSpan(attended);
        var queriesFlat = AttentionMath.AsFlatSpan(queries);
        var cacheKFlat = AttentionMath.AsFlatSpan(cache.K);
        var cacheVFlat = AttentionMath.AsFlatSpan(cache.V);

        FlashAttention.ForwardDecode(
            queriesFlat,
            cacheKFlat,
            cacheVFlat,
            attendedFlat,
            headCount,
            headCount,
            headDim,
            totalLength,
            _attentionScale);

        return OutputProjection.Forward(attended);
    }

    public override float[,] BackwardSTE(float[,] gradientOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientOutput);

        if (_cachedQueries is null || _cachedKeys is null || _cachedValues is null || _cachedAttentionWeights is null)
        {
            return (float[,])gradientOutput.Clone();
        }

        // Backward through OutputProjection
        var gradAttended = OutputProjection.BackwardSTE(gradientOutput);

        var seqLen = gradAttended.GetLength(0);
        var dim = Config.Dimension;
        var headDim = Config.HeadDimension;
        var headCount = Config.HeadCount;

        var gradQueries = new float[seqLen, dim];
        var gradKeys = new float[seqLen, dim];
        var gradValues = new float[seqLen, dim];

        for (var head = 0; head < headCount; head++)
        {
            var headOffset = head * headDim;

            for (var target = 0; target < seqLen; target++)
            {
                // Step 1: dL/d_values and dL/d_attn_weights
                var gradAttnWeights = new float[target + 1];

                for (var source = 0; source <= target; source++)
                {
                    var attnWeight = _cachedAttentionWeights[head, target, source];

                    var gradWeight = 0f;
                    for (var d = 0; d < headDim; d++)
                    {
                        gradValues[source, headOffset + d] += attnWeight * gradAttended[target, headOffset + d];
                        gradWeight += gradAttended[target, headOffset + d] * _cachedValues[source, headOffset + d];
                    }

                    gradAttnWeights[source] = gradWeight;
                }

                // Step 2: Softmax backward
                // dL/d_score[s] = attn[s] * (dL/d_attnWeight[s] - sum_j(attn[j] * dL/d_attnWeight[j]))
                var weightedSum = 0f;
                for (var source = 0; source <= target; source++)
                {
                    weightedSum += _cachedAttentionWeights[head, target, source] * gradAttnWeights[source];
                }

                for (var source = 0; source <= target; source++)
                {
                    var gradScore = _cachedAttentionWeights[head, target, source]
                        * (gradAttnWeights[source] - weightedSum)
                        * _attentionScale;

                    // Step 3: dL/d_queries and dL/d_keys from score gradient
                    for (var d = 0; d < headDim; d++)
                    {
                        gradQueries[target, headOffset + d] += gradScore * _cachedKeys[source, headOffset + d];
                        gradKeys[source, headOffset + d] += gradScore * _cachedQueries[target, headOffset + d];
                    }
                }
            }
        }

        // Backward through RoPE (inverse rotation)
        _rotaryPositionEmbedding.ApplyInverseInPlace(gradQueries, headCount);
        _rotaryPositionEmbedding.ApplyInverseInPlace(gradKeys, headCount);

        // Backward through Q/K/V projections
        var gradInputFromQ = QueryProjection.BackwardSTE(gradQueries);
        var gradInputFromK = KeyProjection.BackwardSTE(gradKeys);
        var gradInputFromV = ValueProjection.BackwardSTE(gradValues);

        // Sum gradients from all three paths (shared input)
        var gradInput = new float[seqLen, dim];
        for (var row = 0; row < seqLen; row++)
        {
            for (var col = 0; col < dim; col++)
            {
                gradInput[row, col] = gradInputFromQ[row, col] + gradInputFromK[row, col] + gradInputFromV[row, col];
            }
        }

        return gradInput;
    }

    private void ApplyHeadWithCache(float[,] attended, float[,] queries, float[,] keys, float[,] values, int head, float[] scores, float[,,] attentionWeights)
    {
        var headOffset = head * Config.HeadDimension;

        for (var targetPosition = 0; targetPosition < queries.GetLength(0); targetPosition++)
        {
            var maxScore = float.NegativeInfinity;

            for (var sourcePosition = 0; sourcePosition <= targetPosition; sourcePosition++)
            {
                var score = 0f;
                for (var dimension = 0; dimension < Config.HeadDimension; dimension++)
                {
                    score += queries[targetPosition, headOffset + dimension] * keys[sourcePosition, headOffset + dimension];
                }

                score *= _attentionScale;
                scores[sourcePosition] = score;
                maxScore = MathF.Max(maxScore, score);
            }

            var partition = 0f;
            for (var sourcePosition = 0; sourcePosition <= targetPosition; sourcePosition++)
            {
                scores[sourcePosition] = MathF.Exp(scores[sourcePosition] - maxScore);
                partition += scores[sourcePosition];
            }

            if (partition <= 0f)
            {
                continue;
            }

            for (var sourcePosition = 0; sourcePosition <= targetPosition; sourcePosition++)
            {
                var weight = scores[sourcePosition] / partition;
                attentionWeights[head, targetPosition, sourcePosition] = weight;
                for (var dimension = 0; dimension < Config.HeadDimension; dimension++)
                {
                    attended[targetPosition, headOffset + dimension] += weight * values[sourcePosition, headOffset + dimension];
                }
            }
        }
    }
}
