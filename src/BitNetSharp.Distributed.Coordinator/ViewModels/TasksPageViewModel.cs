using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Cqrs.Commands;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
using CommunityToolkit.Mvvm.ComponentModel;
using McpServer.Cqrs;

namespace BitNetSharp.Distributed.Coordinator.ViewModels;

/// <summary>
/// MVVM view-model backing the <c>/admin/tasks</c> Razor page.
/// Reads the task queue snapshot via
/// <see cref="GetTaskQueueSnapshotQuery"/> and dispatches
/// <see cref="EnqueueTasksCommand"/> when the operator submits the
/// seed form.
/// </summary>
public sealed partial class TasksPageViewModel : ObservableObject
{
    private readonly IDispatcher _dispatcher;

    public TasksPageViewModel(IDispatcher dispatcher)
    {
        _dispatcher = dispatcher;
    }

    [ObservableProperty]
    private TaskQueueSnapshot _snapshot = new(0, 0, 0, 0);

    [ObservableProperty]
    private string? _lastError;

    [ObservableProperty]
    private string? _lastEnqueueSummary;

    /// <summary>
    /// Query current queue counts for the page header.
    /// </summary>
    public async Task LoadAsync()
    {
        var result = await _dispatcher
            .QueryAsync<TaskQueueSnapshot>(new GetTaskQueueSnapshotQuery())
            .ConfigureAwait(false);

        if (result.IsSuccess)
        {
            Snapshot = result.Value ?? new TaskQueueSnapshot(0, 0, 0, 0);
            LastError = null;
        }
        else
        {
            Snapshot = new TaskQueueSnapshot(0, 0, 0, 0);
            LastError = result.Error;
        }
    }

    /// <summary>
    /// Fan-out N pending tasks from the supplied command payload.
    /// Surfaced as a ViewModel method so the page can call it from
    /// a form POST handler without touching IDispatcher directly.
    /// </summary>
    public async Task EnqueueAsync(EnqueueTasksCommand command)
    {
        var result = await _dispatcher
            .SendAsync<EnqueueTasksResult>(command)
            .ConfigureAwait(false);

        if (result.IsSuccess)
        {
            LastEnqueueSummary = $"Enqueued {result.Value!.Inserted} tasks — first={result.Value.FirstTaskId}, last={result.Value.LastTaskId}.";
            LastError = null;
        }
        else
        {
            LastEnqueueSummary = null;
            LastError = result.Error;
        }

        await LoadAsync().ConfigureAwait(false);
    }
}
