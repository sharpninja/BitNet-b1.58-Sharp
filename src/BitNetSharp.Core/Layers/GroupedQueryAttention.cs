using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Core.Layers;

/// <summary>
/// Grouped-query attention: Q projects to headCount*headDim, K/V project to
/// kvHeadCount*headDim. Each Q head in a group of size (headCount/kvHeadCount)
/// shares the same KV head. When kvHeadCount == headCount, behavior matches
/// plain MHA bit-for-bit given identical weights.
/// </summary>
public sealed class GroupedQueryAttention : AttentionModule
{
    private readonly RotaryPositionEmbedding _rotaryPositionEmbedding;
    private readonly float _attentionScale;
    private readonly int _headCount;
    private readonly int _kvHeadCount;
    private readonly int _headDim;
    private readonly int _groupSize;

    public GroupedQueryAttention(BitNetConfig config, Random random)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(random);

        Config = config;
        _headCount = config.HeadCount;
        _kvHeadCount = config.KvHeadCount;
        _headDim = config.HeadDimension;
        _groupSize = _headCount / _kvHeadCount;
        _attentionScale = 1f / MathF.Sqrt(_headDim);

        int kvDim = _kvHeadCount * _headDim;
        QueryProjection = ParameterInitializer.CreateBitLinear(
            new BitLinearConfig(config.Dimension, config.Dimension), random);
        KeyProjection = ParameterInitializer.CreateBitLinear(
            new BitLinearConfig(config.Dimension, kvDim), random);
        ValueProjection = ParameterInitializer.CreateBitLinear(
            new BitLinearConfig(config.Dimension, kvDim), random);
        OutputProjection = ParameterInitializer.CreateBitLinear(
            new BitLinearConfig(config.Dimension, config.Dimension), random);

        _rotaryPositionEmbedding = new RotaryPositionEmbedding(_headDim, config.RopeTheta);
    }

    public BitNetConfig Config { get; }

    public override BitLinear QueryProjection { get; }

    public override BitLinear KeyProjection { get; }

    public override BitLinear ValueProjection { get; }

    public override BitLinear OutputProjection { get; }

    public int HeadCount => _headCount;

    public int KvHeadCount => _kvHeadCount;

    public int GroupSize => _groupSize;

    public override float AttentionScale => _attentionScale;

    public override long EstimateResidentParameterBytes() =>
        QueryProjection.EstimateResidentParameterBytes()
        + KeyProjection.EstimateResidentParameterBytes()
        + ValueProjection.EstimateResidentParameterBytes()
        + OutputProjection.EstimateResidentParameterBytes();

    private float[,]? _cachedQueries;
    private float[,]? _cachedKeys;
    private float[,]? _cachedValues;
    private float[,,]? _cachedAttentionWeights;

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.GetLength(1) != Config.Dimension)
        {
            throw new ArgumentException(
                $"Expected input dimension {Config.Dimension}, received {input.GetLength(1)}.", nameof(input));
        }

        var queries = QueryProjection.Forward(input);
        var keys = KeyProjection.Forward(input);
        var values = ValueProjection.Forward(input);

        _rotaryPositionEmbedding.ApplyInPlace(queries, _headCount);
        _rotaryPositionEmbedding.ApplyInPlace(keys, _kvHeadCount);

        int seqLen = input.GetLength(0);
        var attended = new float[seqLen, Config.Dimension];
        var attentionWeights = new float[_headCount, seqLen, seqLen];

        for (int head = 0; head < _headCount; head++)
        {
            int kvHead = head / _groupSize;
            int qOffset = head * _headDim;
            int kvOffset = kvHead * _headDim;

            for (int target = 0; target < seqLen; target++)
            {
                float maxScore = float.NegativeInfinity;
                var scores = new float[target + 1];

                for (int source = 0; source <= target; source++)
                {
                    float score = 0f;
                    for (int d = 0; d < _headDim; d++)
                    {
                        score += queries[target, qOffset + d] * keys[source, kvOffset + d];
                    }
                    score *= _attentionScale;
                    scores[source] = score;
                    if (score > maxScore)
                    {
                        maxScore = score;
                    }
                }

                float partition = 0f;
                for (int source = 0; source <= target; source++)
                {
                    scores[source] = MathF.Exp(scores[source] - maxScore);
                    partition += scores[source];
                }

                if (partition <= 0f)
                {
                    continue;
                }

                for (int source = 0; source <= target; source++)
                {
                    float weight = scores[source] / partition;
                    attentionWeights[head, target, source] = weight;
                    for (int d = 0; d < _headDim; d++)
                    {
                        attended[target, qOffset + d] += weight * values[source, kvOffset + d];
                    }
                }
            }
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

        var gradAttended = OutputProjection.BackwardSTE(gradientOutput);

        int seqLen = gradAttended.GetLength(0);
        int dim = Config.Dimension;
        int kvDim = _kvHeadCount * _headDim;

        var gradQueries = new float[seqLen, dim];
        var gradKeys = new float[seqLen, kvDim];
        var gradValues = new float[seqLen, kvDim];

        for (int head = 0; head < _headCount; head++)
        {
            int kvHead = head / _groupSize;
            int qOffset = head * _headDim;
            int kvOffset = kvHead * _headDim;

            for (int target = 0; target < seqLen; target++)
            {
                var gradAttnWeights = new float[target + 1];

                for (int source = 0; source <= target; source++)
                {
                    float attnWeight = _cachedAttentionWeights[head, target, source];
                    float gw = 0f;
                    for (int d = 0; d < _headDim; d++)
                    {
                        gradValues[source, kvOffset + d] += attnWeight * gradAttended[target, qOffset + d];
                        gw += gradAttended[target, qOffset + d] * _cachedValues[source, kvOffset + d];
                    }
                    gradAttnWeights[source] = gw;
                }

                float weightedSum = 0f;
                for (int source = 0; source <= target; source++)
                {
                    weightedSum += _cachedAttentionWeights[head, target, source] * gradAttnWeights[source];
                }

                for (int source = 0; source <= target; source++)
                {
                    float gs = _cachedAttentionWeights[head, target, source]
                        * (gradAttnWeights[source] - weightedSum)
                        * _attentionScale;
                    for (int d = 0; d < _headDim; d++)
                    {
                        gradQueries[target, qOffset + d] += gs * _cachedKeys[source, kvOffset + d];
                        gradKeys[source, kvOffset + d] += gs * _cachedQueries[target, qOffset + d];
                    }
                }
            }
        }

        _rotaryPositionEmbedding.ApplyInverseInPlace(gradQueries, _headCount);
        _rotaryPositionEmbedding.ApplyInverseInPlace(gradKeys, _kvHeadCount);

        var gradInputFromQ = QueryProjection.BackwardSTE(gradQueries);
        var gradInputFromK = KeyProjection.BackwardSTE(gradKeys);
        var gradInputFromV = ValueProjection.BackwardSTE(gradValues);

        var gradInput = new float[seqLen, dim];
        for (int r = 0; r < seqLen; r++)
        {
            for (int c = 0; c < dim; c++)
            {
                gradInput[r, c] = gradInputFromQ[r, c] + gradInputFromK[r, c] + gradInputFromV[r, c];
            }
        }

        return gradInput;
    }
}
