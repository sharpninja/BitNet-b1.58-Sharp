#if NET10_0_OR_GREATER
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests for <see cref="FileSystemWeightStore"/>. Each
/// test gets its own temp directory so parallel test runs never share
/// state.
/// </summary>
public sealed class FileSystemWeightStoreTests : IDisposable
{
    private readonly string _root;
    private readonly FileSystemWeightStore _store;

    public FileSystemWeightStoreTests()
    {
        _root = Path.Combine(Path.GetTempPath(), $"bitnet-weights-{Guid.NewGuid():N}");
        _store = new FileSystemWeightStore(_root);
    }

    public void Dispose()
    {
        if (Directory.Exists(_root))
        {
            try { Directory.Delete(_root, recursive: true); } catch { /* best-effort */ }
        }
    }

    private static byte[] SamplePayload(int seed, int length)
    {
        var rng = new Random(seed);
        var buffer = new byte[length];
        rng.NextBytes(buffer);
        return buffer;
    }

    private static string Sha256Hex(byte[] payload)
    {
        var hash = SHA256.HashData(payload);
        var sb = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
        {
            sb.Append(b.ToString("x2", System.Globalization.CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }

    [Fact]
    public void Empty_store_has_no_versions()
    {
        Assert.Empty(_store.ListVersions());
        Assert.Null(_store.GetLatestVersion());
        Assert.Null(_store.TryGetManifest(1));
        Assert.Null(_store.TryOpenReadStream(1));
    }

    [Fact]
    public void SaveVersion_writes_blob_and_hash_sidecar()
    {
        var payload = SamplePayload(42, 1024);

        var manifest = _store.SaveVersion(1, payload);

        Assert.Equal(1, manifest.Version);
        Assert.Equal(payload.Length, manifest.SizeBytes);
        Assert.Equal(Sha256Hex(payload), manifest.Sha256Hex);
        Assert.True(File.Exists(manifest.PhysicalPath));
        Assert.True(File.Exists(manifest.PhysicalPath + ".sha256"));
    }

    [Fact]
    public void SaveVersion_rejects_non_positive_version()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => _store.SaveVersion(0, new byte[1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => _store.SaveVersion(-1, new byte[1]));
    }

    [Fact]
    public void SaveVersion_refuses_to_overwrite_an_existing_version()
    {
        _store.SaveVersion(7, SamplePayload(1, 32));
        Assert.Throws<InvalidOperationException>(
            () => _store.SaveVersion(7, SamplePayload(2, 32)));
    }

    [Fact]
    public void TryGetManifest_round_trips_the_saved_version()
    {
        var payload = SamplePayload(100, 2048);
        _store.SaveVersion(5, payload);

        var manifest = _store.TryGetManifest(5);

        Assert.NotNull(manifest);
        Assert.Equal(5, manifest!.Version);
        Assert.Equal(payload.Length, manifest.SizeBytes);
        Assert.Equal(Sha256Hex(payload), manifest.Sha256Hex);
    }

    [Fact]
    public void TryOpenReadStream_exposes_exact_bytes()
    {
        var payload = SamplePayload(200, 4096);
        _store.SaveVersion(12, payload);

        using var stream = _store.TryOpenReadStream(12);
        Assert.NotNull(stream);
        using var memory = new MemoryStream();
        stream!.CopyTo(memory);
        Assert.Equal(payload, memory.ToArray());
    }

    [Fact]
    public void ListVersions_returns_versions_in_ascending_numeric_order()
    {
        _store.SaveVersion(3, SamplePayload(3, 16));
        _store.SaveVersion(1, SamplePayload(1, 16));
        _store.SaveVersion(2, SamplePayload(2, 16));

        var versions = _store.ListVersions();

        Assert.Equal(new long[] { 1, 2, 3 }, versions);
    }

    [Fact]
    public void GetLatestVersion_returns_highest_numeric_version()
    {
        _store.SaveVersion(1, SamplePayload(1, 16));
        _store.SaveVersion(42, SamplePayload(42, 16));
        _store.SaveVersion(7, SamplePayload(7, 16));

        Assert.Equal(42, _store.GetLatestVersion());
    }

    [Fact]
    public void Constructor_creates_root_directory_if_missing()
    {
        var fresh = Path.Combine(_root, "sub", "fresh");
        Assert.False(Directory.Exists(fresh));

        var store = new FileSystemWeightStore(fresh);

        Assert.True(Directory.Exists(fresh));
        store.SaveVersion(1, new byte[] { 1, 2, 3 });
        Assert.Single(store.ListVersions());
    }
}
#endif
