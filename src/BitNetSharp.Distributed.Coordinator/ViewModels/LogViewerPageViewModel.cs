using System;
using System.Collections.Generic;
using BitNetSharp.Distributed.Coordinator.Persistence;
using CommunityToolkit.Mvvm.ComponentModel;

namespace BitNetSharp.Distributed.Coordinator.ViewModels;

/// <summary>
/// MVVM view-model backing <c>/admin/logs</c>. Queries the
/// <see cref="SqliteLogStore"/> directly (no CQRS query — the log
/// viewer is a read-only diagnostic tool, not a domain operation
/// worth the ceremony). Exposes filter parameters that the page
/// binds to URL query strings via
/// <see cref="Microsoft.AspNetCore.Components.SupplyParameterFromQueryAttribute"/>.
/// </summary>
public sealed partial class LogViewerPageViewModel : ObservableObject
{
    private readonly SqliteLogStore _logStore;

    public LogViewerPageViewModel(SqliteLogStore logStore)
    {
        _logStore = logStore;
    }

    [ObservableProperty]
    private IReadOnlyList<LogEntryRow> _entries = Array.Empty<LogEntryRow>();

    [ObservableProperty]
    private string? _filterWorkerId;

    [ObservableProperty]
    private string? _filterLevel;

    [ObservableProperty]
    private string? _filterSearch;

    [ObservableProperty]
    private int _limit = 200;

    public void Load()
    {
        Entries = _logStore.Query(
            limit: Math.Clamp(Limit, 10, 2000),
            workerId: string.IsNullOrWhiteSpace(FilterWorkerId) ? null : FilterWorkerId,
            minLevel: string.IsNullOrWhiteSpace(FilterLevel) ? null : FilterLevel,
            search: string.IsNullOrWhiteSpace(FilterSearch) ? null : FilterSearch);
    }
}
