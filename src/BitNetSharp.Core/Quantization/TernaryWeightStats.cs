namespace BitNetSharp.Core.Quantization;

// Counts are long so aggregate transformer weight counts above 2^31 (multi-
// billion-parameter BitNet models) don't silently wrap into negative int32
// values in diagnostic output. Per-tensor counts still fit an int32 up to
// ~2.1B weights per BitLinear, so the widening is ~free at call sites.
public sealed record TernaryWeightStats(long NegativeCount, long ZeroCount, long PositiveCount)
{
    public long TotalCount => NegativeCount + ZeroCount + PositiveCount;
}
