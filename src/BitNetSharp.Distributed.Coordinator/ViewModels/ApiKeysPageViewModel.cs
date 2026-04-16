using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Cqrs.Commands;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
using CommunityToolkit.Mvvm.ComponentModel;
using McpServer.Cqrs;

namespace BitNetSharp.Distributed.Coordinator.ViewModels;

/// <summary>
/// MVVM view-model backing the <c>/admin/api-keys</c> Razor page.
/// All admin-page state lives here so the component's code-behind
/// can shrink to a one-line <c>OnInitialized</c> that forwards to
/// <see cref="LoadAsync"/>. Read-side state (client list) comes from
/// a CQRS <see cref="GetWorkerClientsQuery"/>; the rotate action is
/// a <see cref="RotateClientSecretCommand"/> dispatched through the
/// shared <see cref="IDispatcher"/>.
///
/// <para>
/// Inheriting from CommunityToolkit.Mvvm's
/// <see cref="ObservableObject"/> lets future interactive pages
/// observe property changes without rewriting the VM — today the
/// admin page renders static SSR so the observable plumbing is
/// unused, but keeping it here means an @rendermode InteractiveServer
/// upgrade would not require touching the VM.
/// </para>
/// </summary>
public sealed partial class ApiKeysPageViewModel : ObservableObject
{
    private readonly IDispatcher _dispatcher;

    public ApiKeysPageViewModel(IDispatcher dispatcher)
    {
        _dispatcher = dispatcher;
    }

    /// <summary>The list of worker clients currently displayed.</summary>
    [ObservableProperty]
    private IReadOnlyList<WorkerClientView> _clients = Array.Empty<WorkerClientView>();

    /// <summary>
    /// When set, the page shows a one-shot confirmation banner saying
    /// this client was just rotated. Populated from the
    /// <c>?rotated=</c> query string that the rotate endpoint adds
    /// when it bounces the browser back to the admin page.
    /// </summary>
    [ObservableProperty]
    private string? _rotatedClientId;

    /// <summary>
    /// When set, the page shows a one-shot confirmation banner saying
    /// this client was just added. Populated from the <c>?added=</c>
    /// query string that <c>POST /admin/clients</c> redirects with.
    /// </summary>
    [ObservableProperty]
    private string? _addedClientId;

    /// <summary>
    /// Error message surfaced on the page when the last dispatch
    /// failed. <c>null</c> means everything is green.
    /// </summary>
    [ObservableProperty]
    private string? _lastError;

    /// <summary>
    /// Dispatches a <see cref="GetWorkerClientsQuery"/> through the
    /// CQRS pipeline and stores the result in <see cref="Clients"/>.
    /// Invoked from the page's <c>OnInitializedAsync</c> so each
    /// render picks up the latest registry state.
    /// </summary>
    public async Task LoadAsync()
    {
        var result = await _dispatcher
            .QueryAsync<IReadOnlyList<WorkerClientView>>(new GetWorkerClientsQuery())
            .ConfigureAwait(false);

        if (result.IsSuccess)
        {
            Clients = result.Value ?? Array.Empty<WorkerClientView>();
            LastError = null;
        }
        else
        {
            Clients = Array.Empty<WorkerClientView>();
            LastError = result.Error;
        }
    }

    /// <summary>
    /// Dispatches the <see cref="RotateClientSecretCommand"/> and
    /// returns the result so the minimal API endpoint can decide
    /// whether to bounce the caller back to the page or respond
    /// with a JSON body.
    /// </summary>
    public Task<Result<RotationResult>> RotateAsync(string clientId) =>
        _dispatcher.SendAsync<RotationResult>(new RotateClientSecretCommand(clientId));
}
