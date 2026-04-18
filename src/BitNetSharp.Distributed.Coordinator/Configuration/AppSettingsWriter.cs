using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace BitNetSharp.Distributed.Coordinator.Configuration;

/// <summary>
/// Atomic writer for <c>appsettings.json</c>. Used by the
/// <c>/admin/config/shard-prefixes</c> page to persist edits to
/// <c>Coordinator:ActiveShardPrefixes</c> so
/// <c>IOptionsMonitor&lt;CoordinatorOptions&gt;</c>'s reload-on-change
/// JSON source picks them up without a service restart.
///
/// <para>
/// Write strategy: parse the existing file with <see cref="JsonNode"/>
/// so every unmentioned key is preserved verbatim, serialise to a
/// sibling <c>.tmp</c> file, then <see cref="File.Replace(string,string,string?)"/>
/// swap in place. This keeps readers never seeing a truncated file.
/// </para>
///
/// <para>Fails loud (throws <see cref="FileNotFoundException"/>) when
/// the path is wrong — silent file creation would leave operators
/// editing a ghost file that <c>CreateBuilder</c> never reads.</para>
/// </summary>
public sealed class AppSettingsWriter
{
    private readonly string _contentRootPath;
    private const string FileName = "appsettings.json";
    private const string SectionName = "Coordinator";
    private const string KeyName = "ActiveShardPrefixes";

    public AppSettingsWriter(string contentRootPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(contentRootPath);
        _contentRootPath = contentRootPath;
    }

    /// <summary>
    /// Replaces the <c>Coordinator:ActiveShardPrefixes</c> array on
    /// disk with <paramref name="prefixes"/>. Every other key in the
    /// file is preserved byte-for-byte by the underlying
    /// <see cref="JsonNode"/> round-trip.
    /// </summary>
    public void UpdateActiveShardPrefixes(IReadOnlyList<ShardPrefixConfig> prefixes)
    {
        ArgumentNullException.ThrowIfNull(prefixes);
        foreach (var p in prefixes)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(p.Prefix);
        }

        var path = Path.Combine(_contentRootPath, FileName);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"appsettings.json not found under '{_contentRootPath}'. " +
                "Pass the app's ContentRootPath so the writer edits the " +
                "same file the configuration builder reads.",
                path);
        }

        var raw = File.ReadAllText(path);
        var root = JsonNode.Parse(raw) as JsonObject
            ?? throw new InvalidOperationException(
                $"appsettings.json at '{path}' is not a JSON object.");

        if (root[SectionName] is not JsonObject section)
        {
            section = new JsonObject();
            root[SectionName] = section;
        }

        var array = new JsonArray();
        foreach (var p in prefixes)
        {
            array.Add(new JsonObject
            {
                ["Prefix"] = p.Prefix,
                ["DisplayLabel"] = p.DisplayLabel ?? string.Empty,
            });
        }
        section[KeyName] = array;

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };
        var serialised = root.ToJsonString(options);

        var tmp = path + ".tmp";
        File.WriteAllText(tmp, serialised, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        try
        {
            File.Replace(tmp, path, destinationBackupFileName: null);
        }
        catch
        {
            if (File.Exists(tmp)) File.Delete(tmp);
            throw;
        }
    }
}
