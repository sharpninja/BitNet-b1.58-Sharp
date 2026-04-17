// File format confirmed by inspecting data/WikiText2/wikitext-2-valid-tokens.bin on 2026-04-16:
//
//   Size: 1,174,980 bytes (293,745 int32 tokens, no header).
//   First 16 bytes (hex, little-endian): 01 00 00 00 af 74 00 00 0d 00 00 00 61 01 00 00
//     -> int32[0] = 1, int32[1] = 29871, int32[2] = 13, int32[3] = 353
//
// The file is a raw little-endian int32 token-id stream. There is no magic / version /
// length prefix; total token count is (file size / 4). All ids observed in the first 32 KB
// fall within the 32,000-token BitNet default vocab range, matching the WikiText-2
// pre-tokenization produced by the `tools/prepare_wikitext.py` pipeline.

using System.Buffers.Binary;

namespace BitNetSharp.Core.Training;

/// <summary>
/// Loads the pre-tokenized WikiText-2 validation split for perplexity evaluation.
/// The on-disk format is a raw little-endian <c>int32</c> token-id stream with no header.
/// </summary>
public static class WikiTextValidationLoader
{
    private const string DefaultRelativePath = "data/WikiText2/wikitext-2-valid-tokens.bin";

    /// <summary>
    /// Reads the full token-id stream from <paramref name="path"/> (or, when null, the
    /// repo-default location resolved by walking up from <see cref="AppContext.BaseDirectory"/>).
    /// </summary>
    public static int[] LoadValidationTokens(string? path = null)
    {
        string resolved;
        if (path is null)
        {
            if (!TryResolveDefaultPath(out resolved!))
            {
                throw new FileNotFoundException(
                    $"Could not locate '{DefaultRelativePath}' by walking up from '{AppContext.BaseDirectory}'. " +
                    "Pass an explicit path to LoadValidationTokens.");
            }
        }
        else
        {
            resolved = path;
        }

        if (!File.Exists(resolved))
        {
            throw new FileNotFoundException($"WikiText-2 validation token file not found at '{resolved}'.", resolved);
        }

        var bytes = File.ReadAllBytes(resolved);
        if ((bytes.Length % sizeof(int)) != 0)
        {
            throw new InvalidDataException(
                $"Expected file length to be a multiple of {sizeof(int)} bytes for an int32 token stream, " +
                $"but got {bytes.Length} bytes at '{resolved}'.");
        }

        var tokenCount = bytes.Length / sizeof(int);
        var tokens = new int[tokenCount];
        var source = bytes.AsSpan();
        for (var i = 0; i < tokenCount; i++)
        {
            tokens[i] = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(i * sizeof(int), sizeof(int)));
        }

        return tokens;
    }

    /// <summary>
    /// Yields non-overlapping, fixed-length chunks of <paramref name="tokens"/>. Any
    /// remainder shorter than <paramref name="seqLen"/> is dropped — the perplexity loop
    /// needs at least two adjacent positions to score, and the benchmark layer already
    /// covers partial tails elsewhere.
    /// </summary>
    public static IEnumerable<int[]> ChunkIntoSequences(int[] tokens, int seqLen)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);

        for (var offset = 0; offset + seqLen <= tokens.Length; offset += seqLen)
        {
            var chunk = new int[seqLen];
            Array.Copy(tokens, offset, chunk, 0, seqLen);
            yield return chunk;
        }
    }

    /// <summary>
    /// Attempts to locate the default WikiText-2 validation token binary by walking up the
    /// directory tree from <see cref="AppContext.BaseDirectory"/> until a directory
    /// containing <c>data/WikiText2/wikitext-2-valid-tokens.bin</c> is found.
    /// </summary>
    /// <returns><c>true</c> if the file was found; <c>false</c> otherwise.</returns>
    public static bool TryResolveDefaultPath(out string path)
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            var candidate = Path.Combine(current.FullName, DefaultRelativePath);
            if (File.Exists(candidate))
            {
                path = candidate;
                return true;
            }

            current = current.Parent;
        }

        path = string.Empty;
        return false;
    }
}
