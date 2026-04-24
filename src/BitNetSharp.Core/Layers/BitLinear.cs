using System.Buffers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Training;

namespace BitNetSharp.Core.Layers;

public sealed class BitLinear : Module
{
    private const int ActivationQuantizationMaxMagnitude = 127;
    private const float WeightQuantizationEpsilon = 1e-6f;

    private readonly int _totalWeights;
    private readonly int _packedStride; // packed bytes per output row (5-trit base-3)
    private readonly int _simdPackedStride; // packed bytes per output row (4-trit 2-bit-signed)
    private byte[] _packedWeights;
    private byte[] _simdPackedWeights;

    // Row permutation for cache-aware token-row layout (null = identity)
    private int[]? _rowPermutation;

    // Training state (null until InitializeMasterWeights / ImportMasterWeights is called).
    // Master weights live in an integer bucket+delta accumulator (Epsilon-scaled);
    // float Export/Import is kept as a compat boundary for the existing trainer
    // (T2 replaces those callers with direct integer delta apply).
    private IntegerMasterWeightLayer? _intMasterWeights;
    private LayerScaleProfile? _scaleProfile;
    private float[]? _masterGradients;
    private float[,]? _cachedInput;

    // Default epsilon for integer master-weight resolution when no calibration
    // profile is supplied by the trainer. Small enough to resolve typical
    // gradient magnitudes; T1 sets it here, T2 switches to calibrated profiles.
    private const float DefaultMasterWeightEpsilon = 1e-5f;

    public BitLinear(BitLinearConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        Config = config;
        _totalWeights = config.OutputDimension * config.InputDimension;
        _packedStride = (config.InputDimension + 4) / 5;
        _simdPackedStride = (config.InputDimension + 3) / 4;
        _packedWeights = new byte[config.OutputDimension * _packedStride];
        _simdPackedWeights = new byte[config.OutputDimension * _simdPackedStride];
    }

    public BitLinearConfig Config { get; }

    public float Gamma { get; private set; }

    public bool HasBias => false;

    public int ActivationQuantizationBound => ActivationQuantizationMaxMagnitude;

    public int ActivationQuantizationBitWidth => 8;

    public long EstimateResidentParameterBytes() =>
        (long)_packedWeights.Length + (long)_simdPackedWeights.Length + sizeof(float);

    /// <summary>
    /// True when this layer participates in training (master weights were
    /// initialised). Inference-only models skip the backward cache so callers
    /// that share a pre-quantised block can safely bypass <see cref="Forward"/>.
    /// </summary>
    public bool IsTraining => _intMasterWeights is not null;

    /// <summary>
    /// Returns the <see cref="LayerScaleProfile"/> backing the integer master
    /// weight accumulator, or null if training has not been initialised.
    /// </summary>
    public LayerScaleProfile? GetMasterWeightScaleProfile() => _scaleProfile;

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var inputDim = Config.InputDimension;

        if (input.GetLength(1) != inputDim)
        {
            throw new ArgumentException($"Expected input dimension {inputDim}, but received {input.GetLength(1)}.", nameof(input));
        }

        return ForwardQuantized(QuantizedActivationBlock.FromFloat(input), input);
    }

    /// <summary>
    /// Fast path for callers that have already quantised the shared input
    /// (attention Q/K/V sharing the pre-norm output, FFN Gate/Up sharing the
    /// residual). Skips the per-row absmax scan, which is otherwise repeated
    /// 3x per attention layer and 2x per FFN layer on the same activations.
    /// Pass <paramref name="rawInputForBackward"/> when training so the
    /// backward cache can be populated; pure inference callers pass null.
    /// </summary>
    public float[,] ForwardQuantized(QuantizedActivationBlock input, float[,]? rawInputForBackward = null)
    {
        ArgumentNullException.ThrowIfNull(input);

        var inputDim = Config.InputDimension;
        if (input.Cols != inputDim)
        {
            throw new ArgumentException($"Expected input dimension {inputDim}, but received {input.Cols}.", nameof(input));
        }

        if (_intMasterWeights is not null && rawInputForBackward is not null)
        {
            _cachedInput = (float[,])rawInputForBackward.Clone();
        }

        var rows = input.Rows;
        var outDim = Config.OutputDimension;
        var output = new float[rows, outDim];

        var simdWeights = _simdPackedWeights.AsSpan();
        var quantizedSpan = input.Quantized.AsSpan();
        var decodedBuffer = ArrayPool<sbyte>.Shared.Rent(inputDim);
        try
        {
            var decodedSpan = decodedBuffer.AsSpan(0, inputDim);
            for (var outputColumn = 0; outputColumn < outDim; outputColumn++)
            {
                var physicalRow = _rowPermutation is not null ? _rowPermutation[outputColumn] : outputColumn;
                var simdRow = simdWeights.Slice(physicalRow * _simdPackedStride, _simdPackedStride);
                TritPacking.SimdUnpackLayer(simdRow, decodedSpan, inputDim);

                for (var row = 0; row < rows; row++)
                {
                    var activationSpan = quantizedSpan.Slice(row * inputDim, inputDim);
                    var isum = TritPacking.TernaryDotSimdUnpacked(decodedSpan, activationSpan);
                    output[row, outputColumn] = isum * Gamma * input.RowScales[row];
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(decodedBuffer);
        }

        return output;
    }

    public override float[,] BackwardSTE(float[,] gradientOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientOutput);

        var rows = gradientOutput.GetLength(0);
        var outDim = Config.OutputDimension;
        var inDim = Config.InputDimension;
        var gradInput = new float[rows, inDim];

        var unpackBuffer = ArrayPool<sbyte>.Shared.Rent(inDim);
        try
        {
            for (var row = 0; row < rows; row++)
            {
                for (var outCol = 0; outCol < outDim; outCol++)
                {
                    var grad = gradientOutput[row, outCol] * Gamma;
                    if (grad == 0f)
                    {
                        continue;
                    }

                    var physicalRow = _rowPermutation is not null ? _rowPermutation[outCol] : outCol;
                    TritPacking.UnpackRowInto(_packedWeights, physicalRow * _packedStride, _packedStride, unpackBuffer, inDim);

                    for (var inCol = 0; inCol < inDim; inCol++)
                    {
                        var w = unpackBuffer[inCol];
                        if (w > 0) gradInput[row, inCol] += grad;
                        else if (w < 0) gradInput[row, inCol] -= grad;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(unpackBuffer);
        }

        if (_masterGradients is not null && _cachedInput is not null)
        {
            for (var row = 0; row < rows; row++)
            {
                for (var outCol = 0; outCol < outDim; outCol++)
                {
                    var grad = gradientOutput[row, outCol];
                    if (grad == 0f)
                    {
                        continue;
                    }

                    var weightOffset = outCol * inDim;
                    for (var inCol = 0; inCol < inDim; inCol++)
                    {
                        _masterGradients[weightOffset + inCol] += grad * _cachedInput[row, inCol];
                    }
                }
            }
        }

        return gradInput;
    }

    public void InitializeMasterWeights()
    {
        _scaleProfile = BuildDefaultScaleProfile();
        _intMasterWeights = new IntegerMasterWeightLayer(_scaleProfile);
        _masterGradients = new float[_totalWeights];

        var inputDim = Config.InputDimension;
        var unpackBuffer = new sbyte[inputDim];
        var floatBuffer = new float[_totalWeights];
        for (var row = 0; row < Config.OutputDimension; row++)
        {
            TritPacking.UnpackRowInto(_packedWeights, row * _packedStride, _packedStride, unpackBuffer, inputDim);
            var offset = row * inputDim;
            for (var col = 0; col < inputDim; col++)
            {
                floatBuffer[offset + col] = unpackBuffer[col] * Gamma;
            }
        }

        _intMasterWeights.InitializeFromFloats(floatBuffer);
    }

    private LayerScaleProfile BuildDefaultScaleProfile()
    {
        var epsilon = DefaultMasterWeightEpsilon;
        var maxWeightMagnitude = MathF.Max(1f, Gamma * 2f);
        var bucketSize = epsilon * 65536f;
        var bucketCount = (int)MathF.Ceiling(MathF.Max(maxWeightMagnitude * 2f, 1f) / bucketSize);

        return new LayerScaleProfile
        {
            LayerName = $"bitlinear[{Config.InputDimension}->{Config.OutputDimension}]",
            OutputDimension = Config.OutputDimension,
            InputDimension = Config.InputDimension,
            MaxGradientMagnitude = epsilon * 1024f,
            P99GradientMagnitude = epsilon * 256f,
            MeanGradientMagnitude = epsilon * 64f,
            ObservedWeightRange = maxWeightMagnitude * 2f,
            MaxWeightMagnitude = maxWeightMagnitude,
            Epsilon = epsilon,
            BucketCount = bucketCount,
        };
    }

    public void ZeroGradients()
    {
        if (_masterGradients is not null)
        {
            Array.Clear(_masterGradients);
        }
    }

    public void SyncTernaryFromMaster()
    {
        if (_intMasterWeights is null)
        {
            return;
        }

        var projected = ProjectIntMastersToFloat();

        var absSum = 0f;
        for (var i = 0; i < projected.Length; i++)
        {
            absSum += MathF.Abs(projected[i]);
        }

        Gamma = projected.Length > 0 ? absSum / projected.Length : 0f;

        if (Gamma <= 0f)
        {
            Array.Clear(_packedWeights);
            Array.Clear(_simdPackedWeights);
            return;
        }

        var ternary = new sbyte[_totalWeights];
        for (var i = 0; i < projected.Length; i++)
        {
            var normalized = projected[i] / Gamma + WeightQuantizationEpsilon;
            ternary[i] = (sbyte)Math.Clamp(
                (int)MathF.Round(normalized, MidpointRounding.AwayFromZero), -1, 1);
        }

        PackRowMajor(ternary);
    }

    public float[]? ExportMasterWeights()
    {
        if (_intMasterWeights is null)
        {
            return null;
        }
        return ProjectIntMastersToFloat();
    }

    public float[]? ExportMasterGradients() => _masterGradients is null ? null : [.. _masterGradients];

    public void ImportMasterWeights(float[] weights)
    {
        ArgumentNullException.ThrowIfNull(weights);

        if (weights.Length != _totalWeights)
        {
            throw new ArgumentException(
                $"Expected {_totalWeights} weights, got {weights.Length}.",
                nameof(weights));
        }

        if (_scaleProfile is null || _intMasterWeights is null)
        {
            _scaleProfile = BuildDefaultScaleProfile();
            _intMasterWeights = new IntegerMasterWeightLayer(_scaleProfile);
        }

        _masterGradients ??= new float[weights.Length];
        _intMasterWeights.InitializeFromFloats(weights);
    }

    /// <summary>
    /// Applies a per-index float delta directly to the integer master-weight
    /// accumulator. Unlike <see cref="ImportMasterWeights"/>, this routes each
    /// delta through <see cref="IntegerMasterWeightLayer.ApplyDelta"/> so the
    /// bucket+delta state retains sub-Epsilon precision across steps instead
    /// of re-quantising the full tensor to the Epsilon grid every call. Used
    /// by the trainer after an optimizer step to apply (updated - previous)
    /// without the Import round-trip's quantisation loss.
    /// </summary>
    public void ApplyMasterWeightDeltas(float[] deltas)
    {
        ArgumentNullException.ThrowIfNull(deltas);

        if (deltas.Length != _totalWeights)
        {
            throw new ArgumentException(
                $"Expected {_totalWeights} deltas, got {deltas.Length}.",
                nameof(deltas));
        }

        if (_intMasterWeights is null)
        {
            throw new InvalidOperationException(
                "Master weights not initialised; call InitializeMasterWeights or ImportMasterWeights first.");
        }

        for (var i = 0; i < deltas.Length; i++)
        {
            _intMasterWeights.ApplyDelta(i, deltas[i]);
        }
    }

    private float[] ProjectIntMastersToFloat()
    {
        var buffer = new float[_totalWeights];
        if (_intMasterWeights is null)
        {
            return buffer;
        }

        for (var i = 0; i < _totalWeights; i++)
        {
            buffer[i] = _intMasterWeights.ToFloat(i);
        }

        return buffer;
    }

    /// <summary>
    /// Directly installs a pre-ternarized weight tensor plus per-tensor Gamma
    /// without running quantization. Used by external GGUF importers that have
    /// already decoded integer codes to ternary trits.
    /// </summary>
    /// <param name="ternary">
    /// Row-major flat trits (length == OutputDimension * InputDimension).
    /// Each value must be in {-1, 0, +1}.
    /// </param>
    /// <param name="gamma">Per-tensor absmean scale (must be >= 0).</param>
    public void ImportTernary(sbyte[] ternary, float gamma)
    {
        ArgumentNullException.ThrowIfNull(ternary);

        if (ternary.Length != _totalWeights)
        {
            throw new ArgumentException(
                $"Expected {_totalWeights} trits, got {ternary.Length}.",
                nameof(ternary));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(gamma);

        for (var i = 0; i < ternary.Length; i++)
        {
            var t = ternary[i];
            if (t < -1 || t > 1)
            {
                throw new ArgumentException(
                    $"Trit at index {i} is out of range: {t}. Must be in {{-1, 0, +1}}.",
                    nameof(ternary));
            }
        }

        Gamma = gamma;

        if (gamma == 0f)
        {
            Array.Clear(_packedWeights);
            Array.Clear(_simdPackedWeights);
            return;
        }

        PackRowMajor(ternary);
    }

    public void ApplyRowPermutation(int[] permutation)
    {
        ArgumentNullException.ThrowIfNull(permutation);

        if (permutation.Length != Config.OutputDimension)
        {
            throw new ArgumentException(
                $"Permutation length {permutation.Length} does not match output dimension {Config.OutputDimension}.",
                nameof(permutation));
        }

        // Physically reorder packed weight rows (both canonical + SIMD).
        var newPacked = new byte[_packedWeights.Length];
        var newSimdPacked = new byte[_simdPackedWeights.Length];
        for (var logical = 0; logical < Config.OutputDimension; logical++)
        {
            var physical = permutation[logical];
            Array.Copy(_packedWeights, logical * _packedStride, newPacked, physical * _packedStride, _packedStride);
            Array.Copy(_simdPackedWeights, logical * _simdPackedStride, newSimdPacked, physical * _simdPackedStride, _simdPackedStride);
        }

        _packedWeights = newPacked;
        _simdPackedWeights = newSimdPacked;
        _rowPermutation = (int[])permutation.Clone();
    }

    public int[]? ExportRowPermutation() => _rowPermutation is null ? null : (int[])_rowPermutation.Clone();

    public void QuantizeFromFullPrecision(float[,] fullPrecisionWeights)
    {
        ArgumentNullException.ThrowIfNull(fullPrecisionWeights);

        if (fullPrecisionWeights.GetLength(0) != Config.OutputDimension || fullPrecisionWeights.GetLength(1) != Config.InputDimension)
        {
            throw new ArgumentException(
                $"Expected weights with shape [{Config.OutputDimension}, {Config.InputDimension}], but received [{fullPrecisionWeights.GetLength(0)}, {fullPrecisionWeights.GetLength(1)}].",
                nameof(fullPrecisionWeights));
        }

        Gamma = ComputeAbsMean(fullPrecisionWeights);

        if (Gamma <= 0f)
        {
            Array.Clear(_packedWeights);
            Array.Clear(_simdPackedWeights);
            return;
        }

        var inputDim = Config.InputDimension;
        var ternary = new sbyte[_totalWeights];
        for (var row = 0; row < Config.OutputDimension; row++)
        {
            var offset = row * inputDim;
            for (var column = 0; column < inputDim; column++)
            {
                var normalized = fullPrecisionWeights[row, column] / Gamma;
                normalized += WeightQuantizationEpsilon;
                var quantized = Math.Clamp((int)MathF.Round(normalized, MidpointRounding.AwayFromZero), -1, 1);
                ternary[offset + column] = (sbyte)quantized;
            }
        }

        PackRowMajor(ternary);
    }

    public float[,] ToFullPrecision()
    {
        var result = new float[Config.OutputDimension, Config.InputDimension];
        var inputDim = Config.InputDimension;
        var buffer = new sbyte[inputDim];

        for (var row = 0; row < Config.OutputDimension; row++)
        {
            TritPacking.UnpackRowInto(_packedWeights, row * _packedStride, _packedStride, buffer, inputDim);
            for (var column = 0; column < inputDim; column++)
            {
                result[row, column] = buffer[column] * Gamma;
            }
        }

        return result;
    }

    /// <summary>
    /// Streams ternary weights as FP32 float values directly to a BinaryWriter,
    /// one output row at a time. Peak resident memory is O(InputDimension)
    /// instead of O(OutputDimension * InputDimension), so multi-GB projection
    /// matrices can be serialized without materializing a contiguous FP32 buffer.
    /// </summary>
    public void WriteFullPrecisionTo(BinaryWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        var inputDim = Config.InputDimension;
        var tritBuffer = new sbyte[inputDim];
        var rowBuffer = new float[inputDim];

        for (var row = 0; row < Config.OutputDimension; row++)
        {
            TritPacking.UnpackRowInto(_packedWeights, row * _packedStride, _packedStride, tritBuffer, inputDim);
            for (var column = 0; column < inputDim; column++)
            {
                rowBuffer[column] = tritBuffer[column] * Gamma;
            }

            writer.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(rowBuffer.AsSpan()));
        }
    }

    /// <summary>
    /// Number of packed bytes per output row (5 trits/byte, base-3 packing).
    /// Total bytes in the packed representation = <see cref="PackedStride"/>
    /// * <see cref="BitLinearConfig.OutputDimension"/>.
    /// </summary>
    public int PackedStride => _packedStride;

    /// <summary>
    /// Streams the raw packed-trit bytes (no Gamma, no FP32 expansion) directly
    /// to a BinaryWriter. Used by the v2 BitNetSharp GGUF serializer. Byte layout
    /// matches <see cref="ImportPacked"/>'s expected input: row-major, with
    /// <see cref="PackedStride"/> bytes per output row.
    /// </summary>
    public void WritePackedTritsTo(BinaryWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        writer.Write(_packedWeights);
    }

    /// <summary>
    /// Directly installs a pre-packed trit buffer plus per-tensor Gamma without
    /// any unpack/repack round trip. Used by the v2 BitNetSharp GGUF loader
    /// where the on-disk layout already matches our in-memory packing.
    /// </summary>
    /// <param name="packed">
    /// Raw packed bytes, length == <see cref="PackedStride"/> *
    /// <see cref="BitLinearConfig.OutputDimension"/>.
    /// </param>
    /// <param name="gamma">Per-tensor absmean scale (must be >= 0).</param>
    public void ImportPacked(byte[] packed, float gamma)
    {
        ArgumentNullException.ThrowIfNull(packed);

        if (packed.Length != _packedWeights.Length)
        {
            throw new ArgumentException(
                $"Expected {_packedWeights.Length} packed bytes, got {packed.Length}.",
                nameof(packed));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(gamma);

        Gamma = gamma;
        Buffer.BlockCopy(packed, 0, _packedWeights, 0, packed.Length);
        RegenerateSimdPacked();
    }

    public TernaryWeightStats GetTernaryStats()
    {
        var negativeCount = 0;
        var zeroCount = 0;
        var positiveCount = 0;
        var inputDim = Config.InputDimension;
        var buffer = new sbyte[inputDim];

        for (var row = 0; row < Config.OutputDimension; row++)
        {
            TritPacking.UnpackRowInto(_packedWeights, row * _packedStride, _packedStride, buffer, inputDim);
            for (var col = 0; col < inputDim; col++)
            {
                switch (buffer[col])
                {
                    case < 0:
                        negativeCount++;
                        break;
                    case > 0:
                        positiveCount++;
                        break;
                    default:
                        zeroCount++;
                        break;
                }
            }
        }

        return new TernaryWeightStats(negativeCount, zeroCount, positiveCount);
    }

    private void PackRowMajor(sbyte[] ternary)
    {
        var inputDim = Config.InputDimension;
        for (var row = 0; row < Config.OutputDimension; row++)
        {
            var srcOffset = row * inputDim;
            var dstOffset = row * _packedStride;
            for (var pi = 0; pi < _packedStride; pi++)
            {
                var baseIdx = srcOffset + pi * 5;
                sbyte t0 = baseIdx < ternary.Length ? ternary[baseIdx] : (sbyte)0;
                sbyte t1 = baseIdx + 1 < ternary.Length ? ternary[baseIdx + 1] : (sbyte)0;
                sbyte t2 = baseIdx + 2 < ternary.Length ? ternary[baseIdx + 2] : (sbyte)0;
                sbyte t3 = baseIdx + 3 < ternary.Length ? ternary[baseIdx + 3] : (sbyte)0;
                sbyte t4 = baseIdx + 4 < ternary.Length ? ternary[baseIdx + 4] : (sbyte)0;
                _packedWeights[dstOffset + pi] = TritPacking.PackFive(t0, t1, t2, t3, t4);
            }
        }

        RegenerateSimdPacked();
    }

    /// <summary>
    /// Rebuild the 4-trit SIMD-friendly workspace from the canonical
    /// 5-trit base-3 packed weights. Call after any mutation of
    /// <see cref="_packedWeights"/> so the Forward hot path reads a
    /// layout consistent with the on-disk GGUF representation.
    /// Logical row indexing (i.e., <see cref="_rowPermutation"/>) is
    /// applied identically: row r in SIMD workspace = row r in packed.
    /// </summary>
    private void RegenerateSimdPacked()
    {
        var inputDim = Config.InputDimension;
        var buffer = new sbyte[inputDim];
        for (var row = 0; row < Config.OutputDimension; row++)
        {
            TritPacking.UnpackRowInto(_packedWeights, row * _packedStride, _packedStride, buffer, inputDim);

            var dstOffset = row * _simdPackedStride;
            for (var byteIdx = 0; byteIdx < _simdPackedStride; byteIdx++)
            {
                var baseSlot = byteIdx * 4;
                byte b = 0;
                for (var slot = 0; slot < 4; slot++)
                {
                    var wi = baseSlot + slot;
                    if (wi >= inputDim)
                    {
                        break;
                    }

                    var t = buffer[wi];
                    var code = t == 0 ? 0 : (t > 0 ? 1 : 3);
                    b |= (byte)(code << (slot * 2));
                }

                _simdPackedWeights[dstOffset + byteIdx] = b;
            }
        }
    }

    private static float ComputeAbsMean(float[,] weights)
    {
        if (weights.Length == 0)
        {
            return 0f;
        }

        var sum = 0f;
        foreach (var weight in weights)
        {
            sum += MathF.Abs(weight);
        }

        return sum / weights.Length;
    }

    private static (sbyte[] quantized, float[] rowScales) QuantizeActivations(float[,] input)
    {
        var rows = input.GetLength(0);
        var cols = input.GetLength(1);
        var quantized = new sbyte[rows * cols];
        var rowScales = new float[rows];

        for (var row = 0; row < rows; row++)
        {
            var maxAbs = 0f;
            for (var column = 0; column < cols; column++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(input[row, column]));
            }

            if (maxAbs <= 0f)
            {
                rowScales[row] = 1f;
                continue;
            }

            var scale = maxAbs / ActivationQuantizationMaxMagnitude;
            rowScales[row] = scale;

            var offset = row * cols;
            for (var column = 0; column < cols; column++)
            {
                var q = (int)MathF.Round(input[row, column] / scale, MidpointRounding.AwayFromZero);
                quantized[offset + column] = (sbyte)Math.Clamp(q, -ActivationQuantizationMaxMagnitude, ActivationQuantizationMaxMagnitude);
            }
        }

        return (quantized, rowScales);
    }
}
