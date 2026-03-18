using System.Text.Json;
using BitNetSharp.Core;

namespace BitNetSharp.App;

public sealed record HostedAgentBenchmarkOptions(
    IReadOnlyList<string> ModelSpecifiers,
    string Prompt,
    int? MaxOutputTokens,
    VerbosityLevel Verbosity)
{
    public const string EnvironmentVariableName = "BITNETSHARP_BENCHMARK_OPTIONS";

    public static HostedAgentBenchmarkOptions Parse(string[] args, VerbosityLevel verbosity)
    {
        var models = new List<string>();
        var primaryModel = GetOption(args, "--model=");
        if (!string.IsNullOrWhiteSpace(primaryModel))
        {
            models.Add(primaryModel);
        }

        models.AddRange(GetOptions(args, "--compare-model="));

        if (models.Count == 0)
        {
            models.Add(HostedAgentModelFactory.DefaultModelId);
        }

        return new HostedAgentBenchmarkOptions(
            models.Distinct(StringComparer.Ordinal).ToArray(),
            GetOption(args, "--prompt=") ?? "how are you hosted",
            ParseNullableInt(GetOption(args, "--max-tokens=")),
            verbosity);
    }

    public void StoreInEnvironment() =>
        Environment.SetEnvironmentVariable(EnvironmentVariableName, JsonSerializer.Serialize(this));

    public static HostedAgentBenchmarkOptions LoadFromEnvironment()
    {
        var json = Environment.GetEnvironmentVariable(EnvironmentVariableName);
        if (string.IsNullOrWhiteSpace(json))
        {
            return new HostedAgentBenchmarkOptions(
                [HostedAgentModelFactory.DefaultModelId],
                "how are you hosted",
                null,
                VerbosityLevel.Normal);
        }

        return JsonSerializer.Deserialize<HostedAgentBenchmarkOptions>(json)
            ?? throw new InvalidOperationException("The benchmark options stored in the environment could not be parsed.");
    }

    private static string? GetOption(IEnumerable<string> args, string prefix) =>
        args.FirstOrDefault(argument => argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            ?.Split('=', 2)
            .LastOrDefault();

    private static IEnumerable<string> GetOptions(IEnumerable<string> args, string prefix) =>
        args.Where(argument => argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            .Select(argument => argument.Split('=', 2).Last())
            .Where(value => !string.IsNullOrWhiteSpace(value));

    private static int? ParseNullableInt(string? value) =>
        int.TryParse(value, out var parsed) ? parsed : null;
}
