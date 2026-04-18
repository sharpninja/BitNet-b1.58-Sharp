using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;
using BitNetSharp.Distributed.Coordinator.Configuration;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests for <see cref="AppSettingsWriter"/>. The writer
/// atomically updates the <c>Coordinator:ActiveShardPrefixes</c> array
/// in <c>appsettings.json</c> on disk so
/// <c>IOptionsMonitor&lt;CoordinatorOptions&gt;</c>'s
/// reloadOnChange-backed source picks it up without a restart. The
/// temp-file + <see cref="File.Replace(string,string,string?)"/> path
/// must preserve every other key in the file (ports, auth, etc).
/// </summary>
public sealed class AppSettingsWriterTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _settingsPath;

    public AppSettingsWriterTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"bitnet-appsettings-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
        _settingsPath = Path.Combine(_tempDir, "appsettings.json");
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true); }
        catch { /* best-effort */ }
    }

    private void WriteInitialSettings(string json)
    {
        File.WriteAllText(_settingsPath, json);
    }

    [Fact]
    public void UpdateActiveShardPrefixes_writes_atomic_temp_and_replaces()
    {
        WriteInitialSettings("""
        {
          "Coordinator": {
            "ActiveShardPrefixes": [
              { "Prefix": "old-", "DisplayLabel": "Old" }
            ]
          }
        }
        """);

        var writer = new AppSettingsWriter(_tempDir);
        writer.UpdateActiveShardPrefixes(new[]
        {
            new ShardPrefixConfig { Prefix = "asr-v1-", DisplayLabel = "ASR v1" },
            new ShardPrefixConfig { Prefix = "truckmate-v2-", DisplayLabel = "TruckMate v2" },
        });

        var after = JsonNode.Parse(File.ReadAllText(_settingsPath))!;
        var arr = after["Coordinator"]!["ActiveShardPrefixes"]!.AsArray();
        Assert.Equal(2, arr.Count);
        Assert.Equal("asr-v1-", (string?)arr[0]!["Prefix"]);
        Assert.Equal("ASR v1", (string?)arr[0]!["DisplayLabel"]);
        Assert.Equal("truckmate-v2-", (string?)arr[1]!["Prefix"]);

        // No tmp file lingering.
        Assert.False(File.Exists(_settingsPath + ".tmp"));
    }

    [Fact]
    public void UpdateActiveShardPrefixes_preserves_other_settings_keys()
    {
        WriteInitialSettings("""
        {
          "Logging": { "LogLevel": { "Default": "Information" } },
          "Kestrel": { "Endpoints": { "Http": { "Url": "http://*:5000" } } },
          "Coordinator": {
            "BaseUrl": "http://PAYTON-DESKTOP:5000",
            "MaxStalenessSteps": 5,
            "ActiveShardPrefixes": []
          }
        }
        """);

        var writer = new AppSettingsWriter(_tempDir);
        writer.UpdateActiveShardPrefixes(new[]
        {
            new ShardPrefixConfig { Prefix = "new-", DisplayLabel = "New" },
        });

        var after = JsonNode.Parse(File.ReadAllText(_settingsPath))!;
        Assert.Equal("Information", (string?)after["Logging"]!["LogLevel"]!["Default"]);
        Assert.Equal("http://*:5000", (string?)after["Kestrel"]!["Endpoints"]!["Http"]!["Url"]);
        Assert.Equal("http://PAYTON-DESKTOP:5000", (string?)after["Coordinator"]!["BaseUrl"]);
        Assert.Equal(5, (int?)after["Coordinator"]!["MaxStalenessSteps"]);
        Assert.Single(after["Coordinator"]!["ActiveShardPrefixes"]!.AsArray());
    }

    [Fact]
    public void UpdateActiveShardPrefixes_round_trips_display_labels_with_unicode()
    {
        WriteInitialSettings("""{ "Coordinator": { "ActiveShardPrefixes": [] } }""");

        var writer = new AppSettingsWriter(_tempDir);
        writer.UpdateActiveShardPrefixes(new[]
        {
            new ShardPrefixConfig { Prefix = "日本-", DisplayLabel = "日本語 corpus → 2026" },
        });

        var after = JsonNode.Parse(File.ReadAllText(_settingsPath))!;
        var arr = after["Coordinator"]!["ActiveShardPrefixes"]!.AsArray();
        Assert.Equal("日本-", (string?)arr[0]!["Prefix"]);
        Assert.Equal("日本語 corpus → 2026", (string?)arr[0]!["DisplayLabel"]);
    }

    [Fact]
    public void UpdateActiveShardPrefixes_fails_loud_when_file_missing()
    {
        // No WriteInitialSettings — file does not exist. Writer must
        // NOT silently create it; ops needs to know the path was wrong.
        var writer = new AppSettingsWriter(_tempDir);
        var ex = Assert.Throws<FileNotFoundException>(() =>
            writer.UpdateActiveShardPrefixes(new[]
            {
                new ShardPrefixConfig { Prefix = "x-", DisplayLabel = "X" },
            }));
        Assert.Contains("appsettings.json", ex.Message);
    }

    [Fact]
    public void UpdateActiveShardPrefixes_creates_coordinator_section_when_missing()
    {
        // Legitimate edge: fresh appsettings with only top-level Logging.
        // Writer should graft on a Coordinator.ActiveShardPrefixes key.
        WriteInitialSettings("""
        {
          "Logging": { "LogLevel": { "Default": "Information" } }
        }
        """);

        var writer = new AppSettingsWriter(_tempDir);
        writer.UpdateActiveShardPrefixes(new[]
        {
            new ShardPrefixConfig { Prefix = "seed-", DisplayLabel = "Seed" },
        });

        var after = JsonNode.Parse(File.ReadAllText(_settingsPath))!;
        Assert.Equal("Information", (string?)after["Logging"]!["LogLevel"]!["Default"]);
        var arr = after["Coordinator"]!["ActiveShardPrefixes"]!.AsArray();
        Assert.Single(arr);
        Assert.Equal("seed-", (string?)arr[0]!["Prefix"]);
    }

    [Fact]
    public void UpdateActiveShardPrefixes_rejects_null_list()
    {
        WriteInitialSettings("""{ "Coordinator": { "ActiveShardPrefixes": [] } }""");
        var writer = new AppSettingsWriter(_tempDir);
        Assert.Throws<ArgumentNullException>(() =>
            writer.UpdateActiveShardPrefixes(null!));
    }

    [Fact]
    public void UpdateActiveShardPrefixes_rejects_blank_prefix_entry()
    {
        WriteInitialSettings("""{ "Coordinator": { "ActiveShardPrefixes": [] } }""");
        var writer = new AppSettingsWriter(_tempDir);
        Assert.Throws<ArgumentException>(() =>
            writer.UpdateActiveShardPrefixes(new[]
            {
                new ShardPrefixConfig { Prefix = "", DisplayLabel = "bad" },
            }));
    }
}
