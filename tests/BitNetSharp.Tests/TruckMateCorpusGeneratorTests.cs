#if NET10_0_OR_GREATER
using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Services;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Locks byte parity of the TruckMate synthetic corpus generator
/// across pool versions so refactors don't silently re-roll v1
/// shards. Also gates the v2 expansion against the 5174 vocab cap
/// that keeps previously-serialized weights shape-compatible.
/// </summary>
public sealed class TruckMateCorpusGeneratorTests : IDisposable
{
    private readonly string _root;

    public TruckMateCorpusGeneratorTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "tmcorpus-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best-effort */ }
    }

    private string NewTmpDir()
    {
        var dir = Path.Combine(_root, Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void Generate_seed42_V1_is_byte_deterministic()
    {
        var a = NewTmpDir();
        var b = NewTmpDir();

        TruckMateCorpusGenerator.Generate(a, count: 1000, examplesPerShard: 500, seed: 42,
            poolVersion: CorpusPoolVersion.V1, manifestName: "truckmate-v1");
        TruckMateCorpusGenerator.Generate(b, count: 1000, examplesPerShard: 500, seed: 42,
            poolVersion: CorpusPoolVersion.V1, manifestName: "truckmate-v1");

        foreach (var f in Directory.GetFiles(a, "truckmate-v1-*.txt"))
        {
            var other = Path.Combine(b, Path.GetFileName(f));
            Assert.True(File.Exists(other), $"missing sibling shard {other}");
            Assert.Equal(File.ReadAllBytes(f), File.ReadAllBytes(other));
        }
    }

    [Fact]
    public void Generate_seed42_V2_is_byte_deterministic()
    {
        var a = NewTmpDir();
        var b = NewTmpDir();

        TruckMateCorpusGenerator.Generate(a, count: 1000, examplesPerShard: 500, seed: 42,
            poolVersion: CorpusPoolVersion.V2, manifestName: "truckmate-v2");
        TruckMateCorpusGenerator.Generate(b, count: 1000, examplesPerShard: 500, seed: 42,
            poolVersion: CorpusPoolVersion.V2, manifestName: "truckmate-v2");

        foreach (var f in Directory.GetFiles(a, "truckmate-v2-*.txt"))
        {
            var other = Path.Combine(b, Path.GetFileName(f));
            Assert.True(File.Exists(other), $"missing sibling shard {other}");
            Assert.Equal(File.ReadAllBytes(f), File.ReadAllBytes(other));
        }
    }

    [Fact]
    public void Generate_V1_and_V2_produce_different_byte_streams()
    {
        var a = NewTmpDir();
        var b = NewTmpDir();

        TruckMateCorpusGenerator.Generate(a, 1000, 500, 42, CorpusPoolVersion.V1, "truckmate-v1");
        TruckMateCorpusGenerator.Generate(b, 1000, 500, 42, CorpusPoolVersion.V2, "truckmate-v2");

        var v1 = File.ReadAllLines(Path.Combine(a, "truckmate-v1-shard-0000.txt"));
        var v2 = File.ReadAllLines(Path.Combine(b, "truckmate-v2-shard-0000.txt"));
        Assert.NotEqual(v1, v2);
    }

    [Fact]
    public void Generate_V2_utterance_collisions_under_threshold()
    {
        var dir = NewTmpDir();
        TruckMateCorpusGenerator.Generate(dir, count: 10_000, examplesPerShard: 10_000, seed: 42,
            poolVersion: CorpusPoolVersion.V2, manifestName: "truckmate-v2");

        var lines = File.ReadAllLines(Path.Combine(dir, "truckmate-v2-shard-0000.txt"));
        var utterances = lines.Select(l =>
        {
            var i = l.IndexOf("[INTENT]", StringComparison.Ordinal);
            return i > 0 ? l.Substring(0, i).Trim() : l;
        }).ToArray();

        var uniqueRatio = utterances.Distinct(StringComparer.Ordinal).Count() / (double)utterances.Length;
        // V2 templates are finite — duplicate utterances are expected
        // at 10K draws. Anchor at 0.30 to catch pool regressions (V1
        // is ~0.26 at 10K; V2 superset should be meaningfully higher).
        Assert.True(uniqueRatio > 0.30, $"v2 unique-utterance ratio {uniqueRatio:F3} <= 0.30 (pool regression?)");
    }

    [Fact]
    public void Generate_V2_intent_distribution_roughly_matches_V1()
    {
        string[] Intents(CorpusPoolVersion v, string name)
        {
            var dir = NewTmpDir();
            TruckMateCorpusGenerator.Generate(dir, 10_000, 10_000, 42, v, name);
            var lines = File.ReadAllLines(Path.Combine(dir, $"{name}-shard-0000.txt"));
            var rx = new Regex("\"intent\":\"(\\w+)\"", RegexOptions.Compiled);
            return lines.Select(l =>
            {
                var m = rx.Match(l);
                return m.Success ? m.Groups[1].Value : "unknown";
            }).ToArray();
        }

        var v1 = Intents(CorpusPoolVersion.V1, "truckmate-v1");
        var v2 = Intents(CorpusPoolVersion.V2, "truckmate-v2");

        // Top-level intent switch is still rng.Next(10) so family
        // counts should be comparable. Tolerate 30% drift because
        // v2 variant branches (multi-stop trip, weather reroute)
        // intentionally redistribute within a family.
        var v1Families = v1.GroupBy(x => x).ToDictionary(g => g.Key, g => g.Count());
        var v2Families = v2.GroupBy(x => x).ToDictionary(g => g.Key, g => g.Count());

        foreach (var fam in v1Families.Keys.Intersect(v2Families.Keys))
        {
            var c1 = v1Families[fam];
            var c2 = v2Families[fam];
            if (c1 < 50) continue; // skip rare intents where rounding dominates
            var delta = Math.Abs(c1 - c2) / (double)c1;
            Assert.True(delta < 0.30, $"{fam}: v1={c1} v2={c2} drift={delta:F2} >= 30%");
        }
    }

    [Fact]
    public void Tokenize_V2_sample_stays_under_5174_vocab_cap()
    {
        // Direct WordLevelTokenizer invocation mirrors what the
        // tokenize-corpus CLI does; avoids spawning a child process
        // and keeps the test hermetic.
        var dir = NewTmpDir();
        TruckMateCorpusGenerator.Generate(dir, count: 5_000, examplesPerShard: 5_000, seed: 42,
            poolVersion: CorpusPoolVersion.V2, manifestName: "truckmate-v2");

        var shard = Path.Combine(dir, "truckmate-v2-shard-0000.txt");
        Assert.True(File.Exists(shard));

        var lines = File.ReadAllLines(shard).Where(l => !string.IsNullOrWhiteSpace(l));
        var tokenizer = WordLevelTokenizer.TrainFromCorpus(lines, maxVocab: 5174);

        Assert.True(tokenizer.VocabSize <= 5174,
            $"v2 tokenizer vocab {tokenizer.VocabSize} > 5174 cap — weight compatibility at risk.");
    }

    [Fact]
    public void Generate_writes_versioned_manifest_file()
    {
        var dir = NewTmpDir();
        TruckMateCorpusGenerator.Generate(dir, 500, 500, 42,
            CorpusPoolVersion.V2, "truckmate-v2");

        Assert.True(File.Exists(Path.Combine(dir, "manifest.truckmate-v2.json")));
    }
}
#endif
