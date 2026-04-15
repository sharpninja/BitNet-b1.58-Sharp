using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
using CommunityToolkit.Mvvm.ComponentModel;
using McpServer.Cqrs;

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

    public InstallPageViewModel(IDispatcher dispatcher)
    {
        _dispatcher = dispatcher;
    }

    [ObservableProperty]
    private IReadOnlyList<WorkerInstallBundle> _bundles = Array.Empty<WorkerInstallBundle>();

    [ObservableProperty]
    private string? _lastError;

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

            bundles.Add(new WorkerInstallBundle(
                ClientId: client.ClientId,
                DisplayName: client.DisplayName,
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
    string BashScript,
    string PowerShellScript);
