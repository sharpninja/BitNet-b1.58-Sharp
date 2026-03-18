using BitNetSharp.Core;
using System.Diagnostics;

namespace BitNetSharp.App;

public sealed class LocalCommandHostedAgentModel(LocalCommandModelConfig config, VerbosityLevel verbosity) : IHostedAgentModel
{
    public LocalCommandModelConfig Config { get; } = config ?? throw new ArgumentNullException(nameof(config));

    public string AgentName => Config.ModelId;

    public string ModelId => Config.ModelId;

    public string DisplayName => Config.DisplayName;

    public string PrimaryLanguage => Config.PrimaryLanguage;

    public VerbosityLevel Verbosity { get; } = verbosity;

    public string SystemPrompt => $"Respond in clear American English using the locally hosted model '{DisplayName}'.";

    public IReadOnlyList<string> DescribeModel() =>
    [
        DisplayName,
        $"Model ID: {ModelId}",
        $"Executable: {Config.ExecutablePath}",
        $"Prompt transport: {Config.PromptTransport}",
        $"Working directory: {Config.WorkingDirectory ?? Environment.CurrentDirectory}"
    ];

    public async Task<HostedAgentModelResponse> GetResponseAsync(
        string prompt,
        int? maxOutputTokens = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var startInfo = new ProcessStartInfo
        {
            FileName = Config.ExecutablePath,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = Config.PromptTransport == LocalCommandPromptTransport.StandardInput,
            UseShellExecute = false,
            WorkingDirectory = Config.WorkingDirectory ?? Environment.CurrentDirectory
        };

        foreach (var argument in Config.Arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        if (Config.PromptTransport == LocalCommandPromptTransport.FinalArgument)
        {
            startInfo.ArgumentList.Add(prompt);
        }

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException($"Unable to start the local model process '{Config.ExecutablePath}'.");

        if (Config.PromptTransport == LocalCommandPromptTransport.StandardInput)
        {
            await process.StandardInput.WriteAsync(prompt.AsMemory(), cancellationToken);
            await process.StandardInput.FlushAsync(cancellationToken);
            process.StandardInput.Close();
        }

        var outputTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
        var errorTask = process.StandardError.ReadToEndAsync(cancellationToken);

        await process.WaitForExitAsync(cancellationToken);

        var output = (await outputTask).Trim();
        var error = (await errorTask).Trim();

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"The local model process '{Config.ExecutablePath}' exited with code {process.ExitCode}. Error output: {error}");
        }

        var diagnostics = Verbosity == VerbosityLevel.Quiet
            ? Array.Empty<string>()
            : new[]
            {
                $"Model: {ModelId}",
                "Architecture: local command model",
                $"Primary language: {PrimaryLanguage}"
            };

        return new HostedAgentModelResponse(
            string.IsNullOrWhiteSpace(output) ? $"{DisplayName} produced no response text." : output,
            diagnostics);
    }

    public void Dispose()
    {
    }
}
