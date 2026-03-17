namespace BitNetSharp.Core.Quantization;

public sealed record TernaryWeightStats(int NegativeCount, int ZeroCount, int PositiveCount)
{
    public int TotalCount => NegativeCount + ZeroCount + PositiveCount;
}
