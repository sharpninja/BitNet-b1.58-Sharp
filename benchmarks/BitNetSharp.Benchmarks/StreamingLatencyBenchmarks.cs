using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BitNetSharp.Core;
using BitNetSharp.Core.Training;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// Phase 5: measures time-to-first-token (streaming) vs total generation
/// time (blocking). The streaming path runs the producer on a worker task
/// and yields from a channel; time-to-first-token benefits AnythingLLM-style
/// clients that time out waiting for the full response.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 1, iterationCount: 3)]
public class StreamingLatencyBenchmarks
{
    [Params(8)]
    public int MaxTokens;

    private BitNetPaperModel _model = null!;
    private string _prompt = "benchmark prompt";

    [GlobalSetup]
    public void Setup()
    {
        var examples = new List<TrainingExample>
        {
            new("hello", "world"),
            new("benchmark", "stream"),
            new("quick", "brown fox")
        };
        _model = BitNetPaperModel.CreateForTrainingCorpus(
            examples,
            VerbosityLevel.Quiet,
            enableChainBuckets: false,
            enableSequenceCompression: false,
            NullLoggerFactory.Instance);
    }

    [Benchmark(Baseline = true)]
    public string Blocking_FullResponse()
    {
        var result = _model.GenerateResponse(_prompt, MaxTokens);
        return result.ResponseText;
    }

    [Benchmark]
    public async Task<string> Streaming_TimeToFirstToken()
    {
        await foreach (var token in _model.StreamGenerateAsync(_prompt, MaxTokens))
        {
            return token.TokenText;
        }
        return string.Empty;
    }

    [Benchmark]
    public async Task<int> Streaming_FullResponse()
    {
        var count = 0;
        await foreach (var _ in _model.StreamGenerateAsync(_prompt, MaxTokens))
        {
            count++;
        }
        return count;
    }
}
