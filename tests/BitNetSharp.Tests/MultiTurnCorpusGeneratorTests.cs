#if NET10_0_OR_GREATER
using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using BitNetSharp.Distributed.Coordinator.Services;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Tests for <see cref="MultiTurnCorpusGenerator"/> — the Option Z
/// multi-turn corpus that concatenates N single-turn [USER]/[INTENT]
/// pairs per line. Locks seed-42 byte determinism and the invariant
/// that every emitted line parses as a valid N-turn sequence.
/// </summary>
public sealed class MultiTurnCorpusGeneratorTests : IDisposable
{
    private readonly string _root;

    public MultiTurnCorpusGeneratorTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "mtcorpus-" + Guid.NewGuid().ToString("N"));
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
    public void Generate_seed42_is_byte_deterministic()
    {
        var a = NewTmpDir();
        var b = NewTmpDir();

        MultiTurnCorpusGenerator.Generate(a, count: 400, examplesPerShard: 200, seed: 42);
        MultiTurnCorpusGenerator.Generate(b, count: 400, examplesPerShard: 200, seed: 42);

        foreach (var f in Directory.GetFiles(a, "multiturn-v1-*.txt"))
        {
            var other = Path.Combine(b, Path.GetFileName(f));
            Assert.True(File.Exists(other), $"missing sibling shard {other}");
            Assert.Equal(File.ReadAllBytes(f), File.ReadAllBytes(other));
        }
    }

    [Fact]
    public void Generate_different_seeds_produce_different_bytes()
    {
        var a = NewTmpDir();
        var b = NewTmpDir();

        MultiTurnCorpusGenerator.Generate(a, count: 100, examplesPerShard: 100, seed: 1);
        MultiTurnCorpusGenerator.Generate(b, count: 100, examplesPerShard: 100, seed: 2);

        var aBytes = File.ReadAllBytes(Path.Combine(a, "multiturn-v1-shard-0000.txt"));
        var bBytes = File.ReadAllBytes(Path.Combine(b, "multiturn-v1-shard-0000.txt"));
        Assert.NotEqual(aBytes, bBytes);
    }

    [Fact]
    public void Every_line_has_at_least_two_user_intent_pairs()
    {
        var dir = NewTmpDir();
        MultiTurnCorpusGenerator.Generate(dir, count: 300, examplesPerShard: 300, seed: 42);

        var lines = File.ReadAllLines(Path.Combine(dir, "multiturn-v1-shard-0000.txt"));
        Assert.NotEmpty(lines);

        foreach (var line in lines)
        {
            var userCount = Regex.Matches(line, Regex.Escape("[USER]")).Count;
            var intentCount = Regex.Matches(line, Regex.Escape("[INTENT]")).Count;
            Assert.True(userCount >= 2, $"expected >= 2 [USER] markers, got {userCount}: {line}");
            Assert.True(intentCount >= 2, $"expected >= 2 [INTENT] markers, got {intentCount}: {line}");
            Assert.Equal(userCount, intentCount);
        }
    }

    [Fact]
    public void Every_line_starts_with_user_and_alternates_with_intent()
    {
        var dir = NewTmpDir();
        MultiTurnCorpusGenerator.Generate(dir, count: 50, examplesPerShard: 50, seed: 42);

        var lines = File.ReadAllLines(Path.Combine(dir, "multiturn-v1-shard-0000.txt"));
        foreach (var line in lines)
        {
            Assert.StartsWith("[USER] ", line);
            // Turn boundaries: [USER] and [INTENT] must strictly alternate
            // with [USER] first. Using a forward scan rather than a regex
            // so the failure message names the offending index.
            var tokens = Regex.Matches(line, @"\[USER\]|\[INTENT\]");
            for (var i = 0; i < tokens.Count; i++)
            {
                var expected = i % 2 == 0 ? "[USER]" : "[INTENT]";
                Assert.Equal(expected, tokens[i].Value);
            }
        }
    }

    [Fact]
    public void Shard_name_prefix_is_multiturn_v1()
    {
        var dir = NewTmpDir();
        MultiTurnCorpusGenerator.Generate(dir, count: 20, examplesPerShard: 10, seed: 42);

        var shards = Directory.GetFiles(dir, "*.txt").Select(Path.GetFileName).OrderBy(n => n).ToArray();
        Assert.Equal(new[] { "multiturn-v1-shard-0000.txt", "multiturn-v1-shard-0001.txt" }, shards);
    }

    [Fact]
    public void Manifest_records_shard_and_total_counts()
    {
        var dir = NewTmpDir();
        var manifest = MultiTurnCorpusGenerator.Generate(dir, count: 30, examplesPerShard: 10, seed: 42);

        Assert.Equal("multiturn-v1", manifest.Name);
        Assert.Equal(30, manifest.TotalExamples);
        Assert.Equal(42, manifest.Seed);
        Assert.Equal(3, manifest.Shards.Count);
        Assert.All(manifest.Shards, s => Assert.Equal(10, s.ExampleCount));

        var manifestPath = Path.Combine(dir, "manifest.multiturn-v1.json");
        Assert.True(File.Exists(manifestPath));
    }

    [Fact]
    public void TurnsPerExample_argument_controls_turn_count()
    {
        var dir = NewTmpDir();
        MultiTurnCorpusGenerator.Generate(
            dir,
            count: 20,
            examplesPerShard: 20,
            seed: 42,
            turnsPerExample: 3);

        var lines = File.ReadAllLines(Path.Combine(dir, "multiturn-v1-shard-0000.txt"));
        foreach (var line in lines)
        {
            Assert.Equal(3, Regex.Matches(line, Regex.Escape("[USER]")).Count);
            Assert.Equal(3, Regex.Matches(line, Regex.Escape("[INTENT]")).Count);
        }
    }
}
#endif
