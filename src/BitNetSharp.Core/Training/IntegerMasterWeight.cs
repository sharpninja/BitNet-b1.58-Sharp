namespace BitNetSharp.Core.Training;

/// <summary>
/// Two-component fixed-point master weight accumulator for integer training.
/// master = (bucket * 65536 + delta) * epsilon
/// Delta absorbs gradient steps via exact integer addition.
/// Overflow carries into bucket (exact, no rounding).
/// </summary>
public sealed class IntegerMasterWeightLayer
{
    private readonly float _epsilon;
    private readonly float _epsilonInverse;
    private readonly int _ternaryThreshold;
    private readonly short[] _buckets;
    private readonly short[] _deltas;
    private readonly int _length;

    public IntegerMasterWeightLayer(LayerScaleProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);

        _length = profile.OutputDimension * profile.InputDimension;
        _epsilon = profile.Epsilon;
        _epsilonInverse = 1f / profile.Epsilon;
        _ternaryThreshold = profile.TernaryThreshold;
        _buckets = new short[_length];
        _deltas = new short[_length];
    }

    public int Length => _length;

    public int CarryCount { get; private set; }

    public void InitializeFromTernary(sbyte[] ternaryWeights)
    {
        ArgumentNullException.ThrowIfNull(ternaryWeights);

        for (var i = 0; i < _length && i < ternaryWeights.Length; i++)
        {
            var intValue = ternaryWeights[i] * _ternaryThreshold;
            _buckets[i] = (short)(intValue >> 16);
            _deltas[i] = (short)(intValue & 0xFFFF);
        }
    }

    public void ApplyDelta(int index, float gradient)
    {
        var intDelta = (int)MathF.Round(gradient * _epsilonInverse);
        var newDelta = (int)_deltas[index] + intDelta;

        // Normalize delta to short range with carry into bucket
        while (newDelta > short.MaxValue)
        {
            _buckets[index]++;
            newDelta -= 65536;
            CarryCount++;
        }

        while (newDelta < short.MinValue)
        {
            _buckets[index]--;
            newDelta += 65536;
            CarryCount++;
        }

        _deltas[index] = (short)newDelta;
    }

    public void ProjectToTernary(sbyte[] output)
    {
        ArgumentNullException.ThrowIfNull(output);

        for (var i = 0; i < _length && i < output.Length; i++)
        {
            var fullValue = (int)_buckets[i] * 65536 + _deltas[i];
            output[i] = fullValue > _ternaryThreshold ? (sbyte)1
                      : fullValue < -_ternaryThreshold ? (sbyte)-1
                      : (sbyte)0;
        }
    }

    public float ToFloat(int index)
    {
        var fullValue = (int)_buckets[index] * 65536 + _deltas[index];
        return fullValue * _epsilon;
    }

    public void ResetCarryCount() => CarryCount = 0;
}
