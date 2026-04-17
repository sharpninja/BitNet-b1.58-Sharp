using System.IO;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// Resolves shard IDs to on-disk file paths in the coordinator's
/// corpus directory. Centralises the rule so the <c>/corpus/{shardId}</c>
/// route, the task browser page, and any diagnostic tooling all agree on
/// where shards live.
/// <para>
/// Resolution order for a given <c>shardId</c>:
/// <list type="number">
///   <item>
///     <description><c>&lt;dbDir&gt;/corpus/{shardId}.bin</c> — raw
///     little-endian <c>int32</c> token stream the worker's
///     <c>CorpusClient</c> expects (post Phase-A). Preferred.</description>
///   </item>
///   <item>
///     <description><c>&lt;dbDir&gt;/corpus/{shardId}.txt</c> — legacy
///     plain-text shard retained for backward compatibility with the
///     synthetic-gradient runs that predate the tokenized pipeline.
///     </description>
///   </item>
///   <item>
///     <description><c>&lt;dbDir&gt;/corpus/{shardId}</c> — shard id
///     that already has an extension baked in.</description>
///   </item>
/// </list>
/// Returns <c>null</c> if none of those paths exist. All paths are
/// absolute — callers can display them directly in admin UIs.
/// </para>
/// </summary>
public static class CorpusShardLocator
{
    public static string GetCorpusDirectory(string databasePath)
    {
        var dbDir = Path.GetDirectoryName(Path.GetFullPath(databasePath)) ?? ".";
        return Path.Combine(dbDir, "corpus");
    }

    /// <summary>
    /// Returns the absolute path to the shard file, or <c>null</c> if
    /// no matching file is found. Tries <c>.bin</c> first (tokenized),
    /// then <c>.txt</c> (legacy), then the bare id as-is.
    /// </summary>
    public static string? TryResolve(string databasePath, string shardId)
    {
        var dir = GetCorpusDirectory(databasePath);
        var tokenizedDir = Path.Combine(dir, "tokenized");

        // Prefer tokenized/.bin so the post Phase-A int32 stream wins
        // over the legacy text shard with the same id.
        var tokenizedBin = Path.Combine(tokenizedDir, shardId + ".bin");
        if (File.Exists(tokenizedBin)) return tokenizedBin;

        var binPath = Path.Combine(dir, shardId + ".bin");
        if (File.Exists(binPath)) return binPath;

        var txtPath = Path.Combine(dir, shardId + ".txt");
        if (File.Exists(txtPath)) return txtPath;

        var rawPath = Path.Combine(dir, shardId);
        if (File.Exists(rawPath)) return rawPath;

        return null;
    }

    /// <summary>
    /// Returns the expected-but-maybe-missing absolute path. Used by
    /// admin UIs that want to display where a shard <em>should</em> be
    /// even when nothing has been staged yet.
    /// </summary>
    public static string GetExpectedBinPath(string databasePath, string shardId)
    {
        return Path.Combine(GetCorpusDirectory(databasePath), shardId + ".bin");
    }
}
