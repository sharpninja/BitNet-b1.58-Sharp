using System.Runtime.Intrinsics.X86;

namespace BitNetSharp.Core.Quantization;

/// <summary>
/// Centralised hardware-feature dispatch for the ternary dot kernels. Read
/// once at startup; the dispatcher methods on <see cref="TritPacking"/>
/// branch on these flags without paying the per-call CPUID cost.
///
/// Order of preference (fastest first):
/// 1. <see cref="UseAvxVnniInt8V512"/> (Sapphire Rapids+, Zen 5+) - 64 lanes/chunk fused VPDPBSSD.
/// 2. <see cref="UseAvxVnniInt8"/> (Sapphire Rapids+, Zen 5+) - 32 lanes/chunk fused VPDPBSSD.
/// 3. <see cref="UseAvx2Sign"/> (Haswell+, Excavator+, every Zen) - 32 lanes/chunk via VPSIGNB + widen.
/// 4. Generic <see cref="System.Numerics.Vector{T}"/> - the existing AVX2-on-old-API loop.
///
/// Test override: set <see cref="ForceGeneric"/> to true and every dispatcher
/// short-circuits to the generic path. Equivalence tests use this to compare
/// the active-host kernel against the scalar oracle.
/// </summary>
internal static class TritDotDispatch
{
    public static readonly bool HasAvxVnniInt8V512 = AvxVnniInt8.V512.IsSupported;

    public static readonly bool HasAvxVnniInt8 = AvxVnniInt8.IsSupported;

    public static readonly bool HasAvx2 = Avx2.IsSupported;

    public static readonly bool HasSsse3 = Ssse3.IsSupported;

    /// <summary>
    /// When true, every dispatcher in <see cref="TritPacking"/> falls through
    /// to the generic <see cref="System.Numerics.Vector{T}"/> implementation.
    /// Test-only escape hatch; production code never sets this.
    /// </summary>
    [System.Diagnostics.CodeAnalysis.SuppressMessage(
        "Performance",
        "CA1802:Use literals where appropriate",
        Justification = "Mutable test override, not a constant.")]
    internal static bool ForceGeneric;

    /// <summary>
    /// When true, <see cref="TritPacking.SimdUnpackLayer"/> falls through to
    /// the legacy scalar shift/store loop instead of the SSSE3
    /// mask+shift+VPSHUFB+interleave fast path. Equivalence tests use this to
    /// pin both paths produce bit-identical output. Production never sets it.
    /// </summary>
    [System.Diagnostics.CodeAnalysis.SuppressMessage(
        "Performance",
        "CA1802:Use literals where appropriate",
        Justification = "Mutable test override, not a constant.")]
    internal static bool ForceScalarUnpack;

    /// <summary>
    /// When true, <c>BitLinear.ForwardQuantized</c> and
    /// <c>BitLinear.ForwardInt32</c> stay on the serial outer-column loop
    /// instead of dispatching to the Parallel.For column-stripe path.
    /// Equivalence tests use this to pin parallel and serial dispatch produce
    /// identical output; production never sets it.
    /// </summary>
    [System.Diagnostics.CodeAnalysis.SuppressMessage(
        "Performance",
        "CA1802:Use literals where appropriate",
        Justification = "Mutable test override, not a constant.")]
    internal static bool ForceSerial;

    public static bool UseAvxVnniInt8V512 => !ForceGeneric && HasAvxVnniInt8V512;
    public static bool UseAvxVnniInt8 => !ForceGeneric && HasAvxVnniInt8;
    public static bool UseAvx2Sign => !ForceGeneric && HasAvx2;

    /// <summary>
    /// True when the SSSE3 fast unpack path is available and not test-disabled.
    /// SSSE3 is universal on every modern x64 (Core 2+, Bulldozer+); the flag
    /// exists for symmetry with the AVX2/VNNI flags and for the test override.
    /// </summary>
    public static bool UseSsse3Unpack => !ForceScalarUnpack && HasSsse3;

    /// <summary>
    /// True when BitLinear's outer column loop should dispatch to
    /// Parallel.For with a per-worker decoded-buffer rent.
    /// Production: always true. Test override: <see cref="ForceSerial"/>.
    /// </summary>
    public static bool UseParallelColumnStripes => !ForceSerial;
}
