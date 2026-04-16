using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
using CommunityToolkit.Mvvm.ComponentModel;
using McpServer.Cqrs;

namespace BitNetSharp.Distributed.Coordinator.ViewModels;

/// <summary>
/// MVVM view-model backing the <c>/admin/dashboard</c> Razor page.
/// Loads a single <see cref="DashboardSnapshot"/> from the CQRS
/// pipeline and exposes it as an observable property the page
/// binds to. The page renders static SSR with a 5-second meta
/// refresh for live-ish updates; a future iteration will upgrade
/// to an interactive-server circuit that streams delta updates.
/// </summary>
public sealed partial class DashboardPageViewModel : ObservableObject
{
    private readonly IDispatcher _dispatcher;

    public DashboardPageViewModel(IDispatcher dispatcher)
    {
        _dispatcher = dispatcher;
    }

    [ObservableProperty]
    private DashboardSnapshot? _snapshot;

    [ObservableProperty]
    private string? _lastError;

    public async Task LoadAsync()
    {
        var result = await _dispatcher
            .QueryAsync<DashboardSnapshot>(new GetDashboardSnapshotQuery())
            .ConfigureAwait(false);

        if (result.IsSuccess)
        {
            Snapshot = result.Value;
            LastError = null;
        }
        else
        {
            Snapshot = null;
            LastError = result.Error;
        }
    }
}
