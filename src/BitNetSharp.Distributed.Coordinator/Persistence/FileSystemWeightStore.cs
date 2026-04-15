using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace BitNetSharp.Distributed.Coordinator.Persistence;

/// <summary>
/// Filesystem-backed versioned blob store for the coordinator's
/// global training weights. The work queue stores
/// <c>weight_version</c> integers only; actual ternary-packed model
/// bytes live on disk under a flat directory so workers can stream
/// them via <c>GET /weights/{version}</c> with range support.
///
/// <para>
/// Layout under <c>{rootDirectory}</c>:
/// <code>
///   v0000000001.bin
///   v0000000001.sha256
///   v0000000002.bin
///   v0000000002.sha256
///   ...
/// </code>
/// Each <c>.bin</c> file is the immutable blob payload; the matching
/// <c>.sha256</c> file holds the hex digest the coordinator computed
/// at save time so later readers can verify integrity without
/// rehashing.
/// </para>
/// </summary>
public sealed class FileSystemWeightStore
{
    private readonly string _rootDirectory;
    private readonly object _writeGate = new();

    public FileSystemWeightStore(string rootDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(rootDirectory);
        _rootDirectory = rootDirectory;
        Directory.CreateDirectory(_rootDirectory);
    }

    /// <summary>
    /// Persists the given blob as the contents of the specified
    /// weight version. Overwriting an existing version is disallowed
    /// — the whole point of versioning is that each
    /// <c>weight_version</c> is immutable.
    /// </summary>
    public WeightVersionManifest SaveVersion(long version, ReadOnlySpan<byte> payload)
    {
        if (version <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(version), "Weight version must be positive.");
        }

        var binPath = PathForVersion(version);
        var shaPath = binPath + ".sha256";

        lock (_writeGate)
        {
            if (File.Exists(binPath))
            {
                throw new InvalidOperationException(
                    $"Weight version {version} already exists at {binPath}. Coordinator weight versions are immutable.");
            }

            // Write payload atomically via a staging file then rename
            // so a crash mid-write cannot leave a half-written version
            // visible to /weights/{version} readers.
            var stagingPath = binPath + ".tmp";
            try
            {
                using (var stream = File.Create(stagingPath))
                {
                    stream.Write(payload);
                }

                File.Move(stagingPath, binPath);

                var hash = Sha256Hex(payload);
                File.WriteAllText(shaPath, hash, Encoding.ASCII);
                return new WeightVersionManifest(version, binPath, payload.Length, hash);
            }
            catch
            {
                if (File.Exists(stagingPath))
                {
                    File.Delete(stagingPath);
                }

                throw;
            }
        }
    }

    /// <summary>
    /// Returns the manifest (path, size, hash) for the requested
    /// version if it exists on disk, or <c>null</c> otherwise.
    /// </summary>
    public WeightVersionManifest? TryGetManifest(long version)
    {
        if (version <= 0)
        {
            return null;
        }

        var binPath = PathForVersion(version);
        var shaPath = binPath + ".sha256";
        if (!File.Exists(binPath) || !File.Exists(shaPath))
        {
            return null;
        }

        var length = new FileInfo(binPath).Length;
        var hash = File.ReadAllText(shaPath, Encoding.ASCII).Trim();
        return new WeightVersionManifest(version, binPath, length, hash);
    }

    /// <summary>
    /// Opens a read-only stream over the requested version's blob,
    /// or <c>null</c> if the version does not exist on disk. Caller
    /// owns the returned stream and must dispose it.
    /// </summary>
    public Stream? TryOpenReadStream(long version)
    {
        var manifest = TryGetManifest(version);
        if (manifest is null)
        {
            return null;
        }

        return new FileStream(
            manifest.PhysicalPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 64 * 1024,
            useAsync: true);
    }

    /// <summary>
    /// Enumerates every weight version currently on disk in ascending
    /// numeric order. Handy for the /status dashboard.
    /// </summary>
    public IReadOnlyList<long> ListVersions()
    {
        return Directory
            .EnumerateFiles(_rootDirectory, "v*.bin")
            .Select(path => Path.GetFileNameWithoutExtension(path))
            .Select(name => long.TryParse(name.TrimStart('v'), out var parsed) ? parsed : 0L)
            .Where(v => v > 0)
            .OrderBy(v => v)
            .ToList();
    }

    /// <summary>
    /// Returns the highest-numbered version currently on disk, or
    /// <c>null</c> if the store is empty.
    /// </summary>
    public long? GetLatestVersion()
    {
        var versions = ListVersions();
        return versions.Count == 0 ? null : versions[^1];
    }

    /// <summary>
    /// Computes the filesystem path for the given version using a
    /// 10-digit zero-padded name so alphabetical enumeration matches
    /// numerical order for the first ten billion versions.
    /// </summary>
    private string PathForVersion(long version) =>
        Path.Combine(_rootDirectory, $"v{version:D10}.bin");

    private static string Sha256Hex(ReadOnlySpan<byte> payload)
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(payload, hash);
        var sb = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
        {
            sb.Append(b.ToString("x2", System.Globalization.CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }
}

/// <summary>
/// Summary of a weight version on disk. Returned by
/// <see cref="FileSystemWeightStore.SaveVersion"/> and
/// <see cref="FileSystemWeightStore.TryGetManifest"/>.
/// </summary>
public sealed record WeightVersionManifest(
    long Version,
    string PhysicalPath,
    long SizeBytes,
    string Sha256Hex);
