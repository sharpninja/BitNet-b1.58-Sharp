using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// Generates multi-turn training examples by concatenating N
/// independently-drawn single-turn <c>[USER]...[INTENT]...</c> pairs
/// per output line. This is "Option Z" from the post-P4 plan: reuse
/// the existing <c>[USER]</c> / <c>[INTENT]</c> markers as turn
/// separators so the 5,174-token vocab pin is preserved — no new
/// tokens, no retokenization, no weight-shape reset.
///
/// <para>
/// Each line looks like:
/// <code>[USER] u1 [INTENT] {...} [USER] u2 [INTENT] {...} ...</code>
/// Turns are drawn independently from the <see cref="CorpusPoolVersion.V2"/>
/// pool so the distribution matches the single-turn v2 corpus. That
/// means multi-turn consumers see the same city / intent / slot
/// vocabulary they already know — only the turn-stacking behavior
/// is new to the model.
/// </para>
/// </summary>
public static class MultiTurnCorpusGenerator
{
    public const string DefaultManifestName = "multiturn-v1";

    /// <summary>
    /// Generates <paramref name="count"/> multi-turn examples and
    /// shards them to <paramref name="outputDirectory"/>. Returns a
    /// manifest listing each shard's example count and byte size.
    /// Deterministic for a given <paramref name="seed"/> —
    /// byte-for-byte stable across reruns.
    /// </summary>
    public static CorpusManifest Generate(
        string outputDirectory,
        int count = 50_000,
        int examplesPerShard = 5_000,
        int seed = 42,
        int turnsPerExample = 2)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (examplesPerShard <= 0) throw new ArgumentOutOfRangeException(nameof(examplesPerShard));
        if (turnsPerExample < 2)
        {
            throw new ArgumentOutOfRangeException(
                nameof(turnsPerExample),
                "Multi-turn corpus needs at least 2 turns per line.");
        }

        Directory.CreateDirectory(outputDirectory);

        var rng = new Random(seed);
        var shards = new List<CorpusShardInfo>();
        var shardIndex = 0;
        var remaining = count;

        while (remaining > 0)
        {
            var batchSize = Math.Min(remaining, examplesPerShard);
            var shardId = $"{DefaultManifestName}-shard-{shardIndex:D4}";
            var shardPath = Path.Combine(outputDirectory, $"{shardId}.txt");

            using (var writer = new StreamWriter(shardPath, false, Encoding.UTF8))
            {
                var buffer = new StringBuilder(512);
                for (var i = 0; i < batchSize; i++)
                {
                    buffer.Clear();
                    for (var t = 0; t < turnsPerExample; t++)
                    {
                        if (t > 0) buffer.Append(' ');
                        buffer.Append(TruckMateCorpusGenerator.GenerateExample(rng, CorpusPoolVersion.V2));
                    }
                    writer.WriteLine(buffer.ToString());
                }
            }

            var fileInfo = new FileInfo(shardPath);
            shards.Add(new CorpusShardInfo(shardId, shardPath, batchSize, fileInfo.Length));
            shardIndex++;
            remaining -= batchSize;
        }

        var manifest = new CorpusManifest(
            Name:          DefaultManifestName,
            TotalExamples: count,
            Seed:          seed,
            PoolVersion:   CorpusPoolVersion.V2.ToString(),
            Shards:        shards);

        var manifestPath = Path.Combine(outputDirectory, $"manifest.{DefaultManifestName}.json");
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(manifestPath, json, Encoding.UTF8);

        return manifest;
    }
}
