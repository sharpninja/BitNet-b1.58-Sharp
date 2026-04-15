#if NET10_0_OR_GREATER
using System;
using System.IO;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests for <see cref="SqliteClientRevocationStore"/>.
/// Revocation is the hook that makes the "immediate API-key rotation"
/// story actually instantaneous, so the exact semantics of the
/// iat-vs-revoked comparison need to be locked by test.
/// </summary>
public sealed class SqliteClientRevocationStoreTests : IDisposable
{
    private readonly string _databasePath;
    private readonly FakeTimeProvider _time;
    private readonly SqliteClientRevocationStore _store;

    public SqliteClientRevocationStoreTests()
    {
        _databasePath = Path.Combine(Path.GetTempPath(), $"bitnet-rev-{Guid.NewGuid():N}.db");
        _time = new FakeTimeProvider(new DateTimeOffset(2026, 4, 15, 19, 0, 0, TimeSpan.Zero));
        _store = new SqliteClientRevocationStore($"Data Source={_databasePath}", _time);
    }

    public void Dispose()
    {
        _store.Dispose();
        TryDelete(_databasePath);
        TryDelete(_databasePath + "-wal");
        TryDelete(_databasePath + "-shm");
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) { File.Delete(path); } } catch { /* best-effort */ }
    }

    [Fact]
    public void GetRevokedAt_returns_null_when_client_has_never_been_revoked()
    {
        Assert.Null(_store.GetRevokedAt("never-seen"));
    }

    [Fact]
    public void Revoke_stamps_current_time_and_round_trips()
    {
        var stamp = _store.Revoke("client-alpha");
        var readback = _store.GetRevokedAt("client-alpha");

        Assert.NotNull(readback);
        Assert.Equal(stamp.ToUnixTimeSeconds(), readback!.Value.ToUnixTimeSeconds());
    }

    [Fact]
    public void Revoke_is_idempotent_and_bumps_timestamp_forward()
    {
        var first = _store.Revoke("client-beta");

        _time.Advance(TimeSpan.FromMinutes(5));
        var second = _store.Revoke("client-beta");

        Assert.True(second > first);
        var readback = _store.GetRevokedAt("client-beta");
        Assert.Equal(second.ToUnixTimeSeconds(), readback!.Value.ToUnixTimeSeconds());
    }

    [Fact]
    public void IsIssuedBeforeRevocation_is_false_when_client_has_no_revocation()
    {
        var anyTime = _time.GetUtcNow();
        Assert.False(_store.IsIssuedBeforeRevocation("untracked-client", anyTime));
    }

    [Fact]
    public void IsIssuedBeforeRevocation_rejects_tokens_issued_before_revoke_call()
    {
        var jwtIssuedAt = _time.GetUtcNow();

        // Later revocation invalidates the earlier token.
        _time.Advance(TimeSpan.FromMinutes(1));
        _store.Revoke("client-gamma");

        Assert.True(_store.IsIssuedBeforeRevocation("client-gamma", jwtIssuedAt));
    }

    [Fact]
    public void IsIssuedBeforeRevocation_accepts_tokens_issued_after_revoke_call()
    {
        _store.Revoke("client-delta");

        // Token issued a minute after the revocation passes.
        _time.Advance(TimeSpan.FromMinutes(1));
        var jwtIssuedAt = _time.GetUtcNow();

        Assert.False(_store.IsIssuedBeforeRevocation("client-delta", jwtIssuedAt));
    }

    [Fact]
    public void IsIssuedBeforeRevocation_rejects_tokens_issued_in_the_same_second()
    {
        // Same-second race window: token issued exactly at the
        // revocation timestamp is rejected to close the gap where an
        // operator rotates and a worker's in-flight /connect/token
        // call returns a token with the same iat.
        _store.Revoke("client-race");
        var jwtIssuedAt = _time.GetUtcNow();

        Assert.True(_store.IsIssuedBeforeRevocation("client-race", jwtIssuedAt));
    }

    [Fact]
    public void ListAll_returns_every_revoked_client_ordered_by_id()
    {
        _store.Revoke("zulu");
        _store.Revoke("alpha");
        _store.Revoke("mike");

        var all = _store.ListAll();
        Assert.Equal(3, all.Count);
        Assert.Equal("alpha", all[0].ClientId);
        Assert.Equal("mike",  all[1].ClientId);
        Assert.Equal("zulu",  all[2].ClientId);
    }
}
#endif
