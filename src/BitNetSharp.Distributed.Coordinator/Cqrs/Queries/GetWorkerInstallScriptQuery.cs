using System;
using System.Text;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Identity;
using McpServer.Cqrs;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Queries;

/// <summary>
/// Shell the requested install script should target.
/// </summary>
public enum InstallShell
{
    Bash,
    PowerShell
}

/// <summary>
/// Query that renders a turnkey install script for a single
/// configured worker client. The admin /admin/install page calls
/// this once per client per shell and displays the result so the
/// operator can copy-paste it into the target machine.
///
/// The rendered script embeds the plaintext client secret, so the
/// admin page MUST be gated by <c>AdminPolicy</c> — anyone who can
/// see the script can impersonate the worker.
/// </summary>
public sealed record GetWorkerInstallScriptQuery(
    string ClientId,
    InstallShell Shell) : IQuery<InstallScriptResult>;

/// <summary>
/// Rendered install script plus a suggested download filename.
/// </summary>
public sealed record InstallScriptResult(
    string Filename,
    string ContentType,
    string Content);

public sealed class GetWorkerInstallScriptQueryHandler
    : IQueryHandler<GetWorkerInstallScriptQuery, InstallScriptResult>
{
    private readonly WorkerClientRegistry _registry;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;

    public GetWorkerInstallScriptQueryHandler(
        WorkerClientRegistry registry,
        IOptionsMonitor<CoordinatorOptions> options)
    {
        _registry = registry;
        _options = options;
    }

    public Task<Result<InstallScriptResult>> HandleAsync(
        GetWorkerInstallScriptQuery query,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(query);
        if (string.IsNullOrWhiteSpace(query.ClientId))
        {
            return Task.FromResult(Result<InstallScriptResult>.Failure("ClientId must not be empty."));
        }

        var entry = _registry.Find(query.ClientId);
        if (entry is null)
        {
            return Task.FromResult(Result<InstallScriptResult>.Failure($"Unknown worker client '{query.ClientId}'."));
        }

        var opts = _options.CurrentValue;
        var baseUrl = string.IsNullOrWhiteSpace(opts.BaseUrl)
            ? "http://localhost:5000"
            : opts.BaseUrl.TrimEnd('/');

        var script = query.Shell switch
        {
            InstallShell.Bash => BuildBashScript(entry.ClientId, entry.PlainTextSecret, entry.DisplayName, baseUrl, opts.HeartbeatIntervalSeconds),
            InstallShell.PowerShell => BuildPowerShellScript(entry.ClientId, entry.PlainTextSecret, entry.DisplayName, baseUrl, opts.HeartbeatIntervalSeconds),
            _ => throw new ArgumentOutOfRangeException(nameof(query.Shell), query.Shell, "Unsupported shell.")
        };

        var filename = query.Shell switch
        {
            InstallShell.Bash => $"bitnet-worker-{entry.ClientId}.sh",
            InstallShell.PowerShell => $"bitnet-worker-{entry.ClientId}.ps1",
            _ => "bitnet-worker.txt"
        };

        var contentType = query.Shell switch
        {
            InstallShell.Bash => "text/x-shellscript; charset=utf-8",
            InstallShell.PowerShell => "text/x-powershell; charset=utf-8",
            _ => "text/plain; charset=utf-8"
        };

        return Task.FromResult(Result<InstallScriptResult>.Success(
            new InstallScriptResult(filename, contentType, script)));
    }

    private static string BuildBashScript(
        string clientId,
        string clientSecret,
        string displayName,
        string baseUrl,
        int heartbeatIntervalSeconds)
    {
        var sb = new StringBuilder();
        sb.AppendLine("#!/usr/bin/env bash");
        sb.AppendLine("# BitNetSharp distributed training worker installer");
        sb.AppendLine($"# Generated for client '{clientId}' ({displayName})");
        sb.AppendLine($"# Coordinator: {baseUrl}");
        sb.AppendLine("#");
        sb.AppendLine("# USAGE:");
        sb.AppendLine("#   Save to a file on the worker machine and run:");
        sb.AppendLine("#     chmod +x bitnet-worker.sh && ./bitnet-worker.sh");
        sb.AppendLine("#");
        sb.AppendLine("# The script tries Docker first (fastest); if docker is not");
        sb.AppendLine("# available, it falls back to building + running the worker");
        sb.AppendLine("# from source via the .NET 10 SDK.");
        sb.AppendLine();
        sb.AppendLine("set -euo pipefail");
        sb.AppendLine();
        sb.AppendLine($"export BITNET_COORDINATOR_URL=\"{baseUrl}/\"");
        sb.AppendLine($"export BITNET_CLIENT_ID=\"{clientId}\"");
        sb.AppendLine($"export BITNET_CLIENT_SECRET=\"{clientSecret}\"");
        sb.AppendLine("export BITNET_WORKER_NAME=\"${BITNET_WORKER_NAME:-$(hostname)}\"");
        sb.AppendLine($"export BITNET_HEARTBEAT_SECONDS=\"{heartbeatIntervalSeconds}\"");
        sb.AppendLine("export BITNET_HEALTH_BEACON=\"${BITNET_HEALTH_BEACON:-/tmp/bitnet-worker-$BITNET_CLIENT_ID.alive}\"");
        sb.AppendLine();
        sb.AppendLine("if command -v docker >/dev/null 2>&1; then");
        sb.AppendLine("    echo '[installer] Using Docker worker image.'");
        sb.AppendLine("    exec docker run --rm \\");
        sb.AppendLine("        --pull=always \\");
        sb.AppendLine("        --name \"bitnet-worker-$BITNET_CLIENT_ID\" \\");
        sb.AppendLine("        --read-only --tmpfs /tmp:size=64m,mode=1777 \\");
        sb.AppendLine("        --cap-drop ALL --security-opt no-new-privileges \\");
        sb.AppendLine("        -e BITNET_COORDINATOR_URL \\");
        sb.AppendLine("        -e BITNET_CLIENT_ID \\");
        sb.AppendLine("        -e BITNET_CLIENT_SECRET \\");
        sb.AppendLine("        -e BITNET_WORKER_NAME \\");
        sb.AppendLine("        -e BITNET_HEARTBEAT_SECONDS \\");
        sb.AppendLine("        ghcr.io/sharpninja/bitnetsharp-worker:latest");
        sb.AppendLine("fi");
        sb.AppendLine();
        sb.AppendLine("echo '[installer] Docker not found — building from source.'");
        sb.AppendLine("for bin in git dotnet; do");
        sb.AppendLine("    if ! command -v \"$bin\" >/dev/null 2>&1; then");
        sb.AppendLine("        echo \"[installer] Missing prerequisite: $bin\" >&2");
        sb.AppendLine("        exit 2");
        sb.AppendLine("    fi");
        sb.AppendLine("done");
        sb.AppendLine();
        sb.AppendLine("REPO_DIR=\"${BITNET_REPO_DIR:-$HOME/src/BitNet-b1.58-Sharp}\"");
        sb.AppendLine("if [ ! -d \"$REPO_DIR/.git\" ]; then");
        sb.AppendLine("    mkdir -p \"$(dirname \"$REPO_DIR\")\"");
        sb.AppendLine("    git clone https://github.com/sharpninja/BitNet-b1.58-Sharp.git \"$REPO_DIR\"");
        sb.AppendLine("fi");
        sb.AppendLine("cd \"$REPO_DIR\"");
        sb.AppendLine("git fetch origin");
        sb.AppendLine("git reset --hard origin/main");
        sb.AppendLine();
        sb.AppendLine("dotnet build src/BitNetSharp.Distributed.Worker/BitNetSharp.Distributed.Worker.csproj -c Release");
        sb.AppendLine("exec dotnet \"$REPO_DIR/src/BitNetSharp.Distributed.Worker/bin/Release/net10.0/BitNetSharp.Distributed.Worker.dll\"");
        return sb.ToString();
    }

    private static string BuildPowerShellScript(
        string clientId,
        string clientSecret,
        string displayName,
        string baseUrl,
        int heartbeatIntervalSeconds)
    {
        var sb = new StringBuilder();
        sb.AppendLine("#Requires -Version 7");
        sb.AppendLine("# BitNetSharp distributed training worker installer");
        sb.AppendLine($"# Generated for client '{clientId}' ({displayName})");
        sb.AppendLine($"# Coordinator: {baseUrl}");
        sb.AppendLine("#");
        sb.AppendLine("# USAGE:");
        sb.AppendLine("#   Save to a .ps1 on the worker machine and run it via:");
        sb.AppendLine("#     pwsh -NoProfile -ExecutionPolicy Bypass -File .\\bitnet-worker.ps1");
        sb.AppendLine();
        sb.AppendLine("$ErrorActionPreference = 'Stop'");
        sb.AppendLine();
        sb.AppendLine($"$env:BITNET_COORDINATOR_URL   = '{baseUrl}/'");
        sb.AppendLine($"$env:BITNET_CLIENT_ID         = '{clientId}'");
        sb.AppendLine($"$env:BITNET_CLIENT_SECRET     = '{clientSecret}'");
        sb.AppendLine("if (-not $env:BITNET_WORKER_NAME) { $env:BITNET_WORKER_NAME = $env:COMPUTERNAME }");
        sb.AppendLine($"$env:BITNET_HEARTBEAT_SECONDS = '{heartbeatIntervalSeconds}'");
        sb.AppendLine("if (-not $env:BITNET_HEALTH_BEACON) {");
        sb.AppendLine("    $env:BITNET_HEALTH_BEACON = Join-Path $env:TEMP \"bitnet-worker-$($env:BITNET_CLIENT_ID).alive\"");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("$docker = Get-Command docker -ErrorAction SilentlyContinue");
        sb.AppendLine("if ($docker) {");
        sb.AppendLine("    Write-Host '[installer] Using Docker worker image.'");
        sb.AppendLine("    & docker run --rm `");
        sb.AppendLine("        --pull=always `");
        sb.AppendLine("        --name \"bitnet-worker-$($env:BITNET_CLIENT_ID)\" `");
        sb.AppendLine("        --read-only --tmpfs /tmp:size=64m,mode=1777 `");
        sb.AppendLine("        --cap-drop ALL --security-opt no-new-privileges `");
        sb.AppendLine("        -e BITNET_COORDINATOR_URL `");
        sb.AppendLine("        -e BITNET_CLIENT_ID `");
        sb.AppendLine("        -e BITNET_CLIENT_SECRET `");
        sb.AppendLine("        -e BITNET_WORKER_NAME `");
        sb.AppendLine("        -e BITNET_HEARTBEAT_SECONDS `");
        sb.AppendLine("        ghcr.io/sharpninja/bitnetsharp-worker:latest");
        sb.AppendLine("    exit $LASTEXITCODE");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("Write-Host '[installer] Docker not found - building from source.'");
        sb.AppendLine("foreach ($bin in @('git','dotnet')) {");
        sb.AppendLine("    if (-not (Get-Command $bin -ErrorAction SilentlyContinue)) {");
        sb.AppendLine("        Write-Error \"Missing prerequisite: $bin\"");
        sb.AppendLine("        exit 2");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("$repoDir = if ($env:BITNET_REPO_DIR) { $env:BITNET_REPO_DIR } else { Join-Path $HOME 'src\\BitNet-b1.58-Sharp' }");
        sb.AppendLine("if (-not (Test-Path (Join-Path $repoDir '.git'))) {");
        sb.AppendLine("    New-Item -ItemType Directory -Path (Split-Path -Parent $repoDir) -Force | Out-Null");
        sb.AppendLine("    git clone https://github.com/sharpninja/BitNet-b1.58-Sharp.git $repoDir");
        sb.AppendLine("}");
        sb.AppendLine("Set-Location $repoDir");
        sb.AppendLine("git fetch origin");
        sb.AppendLine("git reset --hard origin/main");
        sb.AppendLine();
        sb.AppendLine("dotnet build src/BitNetSharp.Distributed.Worker/BitNetSharp.Distributed.Worker.csproj -c Release");
        sb.AppendLine("dotnet \"$repoDir\\src\\BitNetSharp.Distributed.Worker\\bin\\Release\\net10.0\\BitNetSharp.Distributed.Worker.dll\"");
        return sb.ToString();
    }
}
