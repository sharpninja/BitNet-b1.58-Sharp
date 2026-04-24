using BenchmarkDotNet.Running;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// Entry point for the BitNet inference-latency benchmark suites.
///
/// Run a single suite:
///   dotnet run -c Release --project benchmarks/BitNetSharp.Benchmarks -- --filter '*BitLinear*'
/// Run everything:
///   dotnet run -c Release --project benchmarks/BitNetSharp.Benchmarks -- --filter '*'
/// </summary>
internal static class Program
{
    public static int Main(string[] args)
    {
        var switcher = BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly);
        var summaries = switcher.Run(args);
        return summaries is null ? 0 : 0;
    }
}
