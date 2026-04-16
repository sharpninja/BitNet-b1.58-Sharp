using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
using BitNetSharp.Distributed.Coordinator.Identity;
using CommunityToolkit.Mvvm.ComponentModel;
using McpServer.Cqrs;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.ViewModels;

/// <summary>
/// MVVM view-model backing the <c>/admin/install</c> Razor page.
/// Materializes a bash install script and a PowerShell install
/// script for every currently-configured worker client by
/// dispatching <see cref="GetWorkerClientsQuery"/> then one
/// <see cref="GetWorkerInstallScriptQuery"/> per client per shell.
/// </summary>
public sealed partial class InstallPageViewModel : ObservableObject
{
    private readonly IDispatcher _dispatcher;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly WorkerClientRegistry _registry;

    public InstallPageViewModel(
        IDispatcher dispatcher,
        IOptionsMonitor<CoordinatorOptions> options,
        WorkerClientRegistry registry)
    {
        _dispatcher = dispatcher;
        _options = options;
        _registry = registry;
    }

    [ObservableProperty]
    private IReadOnlyList<WorkerInstallBundle> _bundles = Array.Empty<WorkerInstallBundle>();

    [ObservableProperty]
    private string? _lastError;

    /// <summary>
    /// Public base URL of the coordinator, trimmed. Used in the
    /// one-liner install snippets so operators can paste a
    /// curl/iwr command into a remote machine.
    /// </summary>
    public string BaseUrl => string.IsNullOrWhiteSpace(_options.CurrentValue.BaseUrl)
        ? "http://localhost:5000"
        : _options.CurrentValue.BaseUrl.TrimEnd('/');

    public async Task LoadAsync()
    {
        var clientsResult = await _dispatcher
            .QueryAsync<IReadOnlyList<WorkerClientView>>(new GetWorkerClientsQuery())
            .ConfigureAwait(false);

        if (!clientsResult.IsSuccess)
        {
            Bundles = Array.Empty<WorkerInstallBundle>();
            LastError = clientsResult.Error;
            return;
        }

        var bundles = new List<WorkerInstallBundle>();
        foreach (var client in clientsResult.Value ?? Array.Empty<WorkerClientView>())
        {
            var bash = await _dispatcher
                .QueryAsync<InstallScriptResult>(new GetWorkerInstallScriptQuery(client.ClientId, InstallShell.Bash))
                .ConfigureAwait(false);
            var ps1 = await _dispatcher
                .QueryAsync<InstallScriptResult>(new GetWorkerInstallScriptQuery(client.ClientId, InstallShell.PowerShell))
                .ConfigureAwait(false);

            var entry = _registry.Find(client.ClientId);
            bundles.Add(new WorkerInstallBundle(
                ClientId: client.ClientId,
                DisplayName: client.DisplayName,
                ClientSecret: entry?.PlainTextSecret ?? string.Empty,
                BashScript: bash.IsSuccess ? bash.Value!.Content : $"[error] {bash.Error}",
                PowerShellScript: ps1.IsSuccess ? ps1.Value!.Content : $"[error] {ps1.Error}"));
        }

        Bundles = bundles;
        LastError = null;
    }
}

/// <summary>
/// Render-ready view model for one worker client on the install
/// page. The bash and powershell scripts are already fully
/// rendered strings with the plaintext client secret embedded.
/// </summary>
public sealed record WorkerInstallBundle(
    string ClientId,
    string DisplayName,
    string ClientSecret,
    string BashScript,
    string PowerShellScript);
