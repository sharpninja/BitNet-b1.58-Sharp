namespace BitNetSharp.Core.Inference;

/// <summary>
/// Fixed-point integer helpers for zero-float inference. Q16.16 is the
/// default format: 16 integer bits, 16 fractional bits, long-sized so
/// intermediate multiplies do not overflow before the normalising shift.
/// </summary>
public static class IntegerMath
{
    public const int Q16_16_SHIFT = 16;
    public const long Q16_16_ONE = 1L << 16;

    /// <summary>
    /// Computes 1/sqrt(x) in Q16.16 fixed-point for a Q16.16 input. Uses a
    /// normalising shift plus four Newton-Raphson iterations so the result
    /// tracks the float equivalent within ~0.1 percent relative error across
    /// [1, 2^30]. Returns 0 for non-positive inputs (defensive).
    /// </summary>
    public static int RsqrtQ16_16(long xQ16_16)
    {
        if (xQ16_16 <= 0) return 0;

        // Normalise x into [0.25, 1.0) in Q16.16 by shifting out pairs of bits.
        // We track the shift count k so we can denormalise the result at the end:
        // rsqrt(x * 2^(2k)) = rsqrt(x) / 2^k.
        var k = 0;
        var xn = xQ16_16;
        while (xn >= Q16_16_ONE)
        {
            xn >>= 2;
            k++;
        }
        while (xn < (Q16_16_ONE >> 2) && k > -14)
        {
            xn <<= 2;
            k--;
        }

        // Initial guess y0 = 2 - x (Q16.16), valid on [0.25, 1.0)
        var y = (Q16_16_ONE << 1) - xn;

        // Newton-Raphson: y_{n+1} = y_n * (3 - x * y_n^2) / 2
        for (var iter = 0; iter < 4; iter++)
        {
            var y2 = (y * y) >> Q16_16_SHIFT;
            var xy2 = (xn * y2) >> Q16_16_SHIFT;
            var threeMinus = (3L << Q16_16_SHIFT) - xy2;
            y = (y * threeMinus) >> (Q16_16_SHIFT + 1);
        }

        if (k > 0) y >>= k;
        else if (k < 0) y <<= -k;

        return (int)Math.Clamp(y, int.MinValue, int.MaxValue);
    }

    public static long ToQ16_16(float value) => (long)(value * (double)Q16_16_ONE);

    public static float FromQ16_16(long q) => q / (float)Q16_16_ONE;
}
