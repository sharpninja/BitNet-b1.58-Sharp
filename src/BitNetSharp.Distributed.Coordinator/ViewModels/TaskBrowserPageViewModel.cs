using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
using CommunityToolkit.Mvvm.ComponentModel;
using McpServer.Cqrs;

namespace BitNetSharp.Distributed.Coordinator.ViewModels;

/// <summary>
/// MVVM view-model for the <c>/admin/task-browser</c> Razor page. Wraps
/// <see cref="GetTaskBrowserSnapshotQuery"/> so the page can bind
/// against <see cref="Snapshot"/> and trigger refreshes via
/// <see cref="LoadAsync"/>.
/// </summary>
public sealed partial class TaskBrowserPageViewModel : ObservableObject
{
    private readonly IDispatcher _dispatcher;

    public TaskBrowserPageViewModel(IDispatcher dispatcher)
    {
        _dispatcher = dispatcher;
    }

    [ObservableProperty]
    private TaskBrowserSnapshot? _snapshot;

    [ObservableProperty]
    private string? _lastError;

    [ObservableProperty]
    private int _limit = 200;

    public async Task LoadAsync()
    {
        var result = await _dispatcher
            .QueryAsync<TaskBrowserSnapshot>(new GetTaskBrowserSnapshotQuery(Limit))
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
