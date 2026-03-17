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

        var attended = new float[input.GetLength(0), Config.Dimension];

        for (var head = 0; head < Config.HeadCount; head++)
        {
            var scores = new float[queries.GetLength(0)];
            ApplyHead(attended, queries, keys, values, head, scores);
        }

        return OutputProjection.Forward(attended);
    }

    private void ApplyHead(float[,] attended, float[,] queries, float[,] keys, float[,] values, int head, float[] scores)
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
                for (var dimension = 0; dimension < Config.HeadDimension; dimension++)
                {
                    attended[targetPosition, headOffset + dimension] += weight * values[sourcePosition, headOffset + dimension];
                }
            }
        }
    }
}
