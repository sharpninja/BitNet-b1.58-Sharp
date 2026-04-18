using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
using CommunityToolkit.Mvvm.ComponentModel;
using McpServer.Cqrs;

namespace BitNetSharp.Distributed.Coordinator.ViewModels;

/// <summary>
/// MVVM view-model for <c>/admin/training-status</c>. Polls the
/// <see cref="GetTrainingStatusQuery"/> on a 2-second cadence driven
/// by the page's InteractiveServer timer, and swaps the current
/// snapshot atomically so partial renders never display.
/// </summary>
public sealed partial class TrainingStatusPageViewModel : ObservableObject
{
    private readonly IDispatcher _dispatcher;

    public TrainingStatusPageViewModel(IDispatcher dispatcher)
    {
        _dispatcher = dispatcher;
    }

    [ObservableProperty]
    private TrainingStatusSnapshot? _snapshot;

    [ObservableProperty]
    private string? _lastError;

    public async Task LoadAsync()
    {
        var result = await _dispatcher
            .QueryAsync<TrainingStatusSnapshot>(new GetTrainingStatusQuery())
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
