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
            SplitCanonical(intValue, out var bucket, out var delta);
            _buckets[i] = bucket;
            _deltas[i] = delta;
        }
    }

    /// <summary>
    /// Initialises the bucket/delta state from a float master-weight snapshot.
    /// Each float is quantised to the nearest int step of size <see cref="_epsilon"/>
    /// and split into bucket (upper 16 bits) + delta (lower 16 bits).
    /// Used by BitLinear.InitializeMasterWeights and ImportMasterWeights to
    /// migrate a float checkpoint into integer master state.
    /// </summary>
    public void InitializeFromFloats(float[] weights)
    {
        ArgumentNullException.ThrowIfNull(weights);

        for (var i = 0; i < _length && i < weights.Length; i++)
        {
            var intValue = (int)MathF.Round(weights[i] * _epsilonInverse);
            SplitCanonical(intValue, out var bucket, out var delta);
            _buckets[i] = bucket;
            _deltas[i] = delta;
        }
    }

    /// <summary>
    /// Splits an int32 into (bucket, delta) such that
    /// bucket * 65536 + delta == intValue AND delta is in signed-short range.
    /// The naive (int >> 16, int &amp; 0xFFFF) split corrupts values whose low
    /// 16 bits interpret as a negative short under the 65536 factor: e.g.
    /// 50000 -&gt; (0, -15536) -&gt; 0*65536 + (-15536) = -15536 (wrong).
    /// Canonicalising pushes the 65536 overflow into the bucket.
    /// </summary>
    private static void SplitCanonical(int intValue, out short bucket, out short delta)
    {
        var loUnsigned = intValue & 0xFFFF;
        var hi = intValue >> 16;
        if (loUnsigned > short.MaxValue)
        {
            delta = (short)(loUnsigned - 65536);
            hi += 1;
        }
        else
        {
            delta = (short)loUnsigned;
        }
        bucket = (short)hi;
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
