using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using CommunityToolkit.Mvvm.ComponentModel;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.ViewModels;

/// <summary>
/// MVVM view-model for <c>/admin/config/shard-prefixes</c>. Loads the
/// current <see cref="CoordinatorOptions.ActiveShardPrefixes"/> list
/// into an editable buffer, exposes add / remove / reorder mutations,
/// and flushes the edit back to <c>appsettings.json</c> via
/// <see cref="AppSettingsWriter"/>. The write triggers
/// reloadOnChange-backed <see cref="IOptionsMonitor{T}"/> so every
/// downstream consumer (seeders, dashboards, etc.) picks up the new
/// list within ~250 ms — no service restart.
/// </summary>
public sealed partial class ShardPrefixesPageViewModel : ObservableObject
{
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly AppSettingsWriter _writer;
    private readonly SqliteWorkQueueStore _workQueue;

    public ShardPrefixesPageViewModel(
        IOptionsMonitor<CoordinatorOptions> options,
        AppSettingsWriter writer,
        SqliteWorkQueueStore workQueue)
    {
        _options = options;
        _writer = writer;
        _workQueue = workQueue;
    }

    /// <summary>Editable buffer bound to the page's form rows.</summary>
    public ObservableCollection<EditablePrefix> Rows { get; } = new();

    [ObservableProperty] private string? _lastError;
    [ObservableProperty] private string? _lastSuccess;

    public void LoadFromOptions()
    {
        Rows.Clear();
        foreach (var p in _options.CurrentValue.ActiveShardPrefixes ?? Array.Empty<ShardPrefixConfig>())
        {
            Rows.Add(new EditablePrefix
            {
                Prefix = p.Prefix,
                DisplayLabel = p.DisplayLabel,
                PendingCount = SafeCount(p.Prefix),
            });
        }
    }

    public void AddRow()
    {
        Rows.Add(new EditablePrefix { Prefix = string.Empty, DisplayLabel = string.Empty });
    }

    public void RemoveRow(EditablePrefix row)
    {
        Rows.Remove(row);
    }

    public void MoveUp(EditablePrefix row)
    {
        var idx = Rows.IndexOf(row);
        if (idx > 0)
        {
            Rows.Move(idx, idx - 1);
        }
    }

    public void MoveDown(EditablePrefix row)
    {
        var idx = Rows.IndexOf(row);
        if (idx >= 0 && idx < Rows.Count - 1)
        {
            Rows.Move(idx, idx + 1);
        }
    }

    public void RefreshPreviewCounts()
    {
        foreach (var row in Rows)
        {
            row.PendingCount = SafeCount(row.Prefix);
        }
    }

    /// <summary>
    /// Validates, atomically persists, and waits up to
    /// <paramref name="reloadTimeout"/> (default 1 s) for
    /// <see cref="IOptionsMonitor{T}"/> to observe the new list. Returns
    /// <c>true</c> on success; sets <see cref="LastError"/> otherwise.
    /// </summary>
    public async Task<bool> SaveAsync(TimeSpan? reloadTimeout = null)
    {
        LastError = null;
        LastSuccess = null;

        var trimmed = Rows
            .Select(r => new ShardPrefixConfig
            {
                Prefix = (r.Prefix ?? string.Empty).Trim(),
                DisplayLabel = (r.DisplayLabel ?? string.Empty).Trim(),
            })
            .ToList();

        foreach (var r in trimmed)
        {
            if (string.IsNullOrWhiteSpace(r.Prefix))
            {
                LastError = "Each prefix row must have a non-blank Prefix value.";
                return false;
            }
        }

        var duplicates = trimmed
            .GroupBy(r => r.Prefix, StringComparer.Ordinal)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToArray();
        if (duplicates.Length > 0)
        {
            LastError = $"Duplicate prefix(es): {string.Join(", ", duplicates)}";
            return false;
        }

        try
        {
            _writer.UpdateActiveShardPrefixes(trimmed);
        }
        catch (Exception ex)
        {
            LastError = $"Failed to persist prefixes: {ex.Message}";
            return false;
        }

        // Wait for IOptionsMonitor to observe the change. The JSON
        // source's reloadOnChange watcher fires on FileSystemWatcher
        // notifications, usually <250 ms.
        var deadline = DateTime.UtcNow + (reloadTimeout ?? TimeSpan.FromSeconds(1));
        while (DateTime.UtcNow < deadline)
        {
            var live = _options.CurrentValue.ActiveShardPrefixes ?? Array.Empty<ShardPrefixConfig>();
            if (SameOrder(live, trimmed))
            {
                LastSuccess = $"Saved {trimmed.Count} prefix(es).";
                RefreshPreviewCounts();
                return true;
            }
            await Task.Delay(50).ConfigureAwait(false);
        }

        // File was written atomically but the in-memory snapshot has
        // not caught up yet. Still report success — the reload will
        // land on the next watcher tick.
        LastSuccess = $"Saved {trimmed.Count} prefix(es). Reload may lag a moment.";
        return true;
    }

    private int SafeCount(string? prefix)
    {
        if (string.IsNullOrWhiteSpace(prefix)) return 0;
        try
        {
            return _workQueue.CountByShardPrefixAndState(prefix, WorkTaskState.Pending);
        }
        catch
        {
            return 0;
        }
    }

    private static bool SameOrder(
        IReadOnlyList<ShardPrefixConfig> left,
        IReadOnlyList<ShardPrefixConfig> right)
    {
        if (left.Count != right.Count) return false;
        for (var i = 0; i < left.Count; i++)
        {
            if (!string.Equals(left[i].Prefix, right[i].Prefix, StringComparison.Ordinal)) return false;
            if (!string.Equals(left[i].DisplayLabel, right[i].DisplayLabel, StringComparison.Ordinal)) return false;
        }
        return true;
    }

    public sealed partial class EditablePrefix : ObservableObject
    {
        [ObservableProperty] private string _prefix = string.Empty;
        [ObservableProperty] private string _displayLabel = string.Empty;
        [ObservableProperty] private int _pendingCount;
    }
}
