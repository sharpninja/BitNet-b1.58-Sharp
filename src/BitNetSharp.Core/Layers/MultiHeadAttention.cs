using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Core.Layers;

public sealed class MultiHeadAttention : Module
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

    public BitLinear QueryProjection { get; }

    public BitLinear KeyProjection { get; }

    public BitLinear ValueProjection { get; }

    public BitLinear OutputProjection { get; }

    public bool UsesRotaryPositionEmbedding => true;

    public bool AppliesRotaryPositionEmbeddingToQueriesAndKeysOnly => true;

    public bool UsesCausalAttentionMask => true;

    public float AttentionScale => _attentionScale;

    public long EstimateResidentParameterBytes() =>
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

        var queries = QueryProjection.Forward(input);
        var keys = KeyProjection.Forward(input);
        var values = ValueProjection.Forward(input);

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
