using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// Queries Anthropic Claude Sonnet to synthesize ASR-noisy training
/// lines in the same <c>[USER] &lt;utterance&gt; [INTENT] {json}</c>
/// format as <see cref="TruckMateCorpusGenerator"/>. Output slots into
/// the existing corpus pipeline under the <c>asr-v1-</c> shard prefix.
///
/// <para>Unlike the deterministic generators, Sonnet output depends on
/// model sampling, so runs are not byte-reproducible. Few-shot examples
/// drawn from <see cref="TruckMateCorpusGenerator.GenerateExample"/>
/// at fixed seeds anchor the vocab distribution to the existing 5,174-
/// token vocab so the tokenizer downstream does not need retraining.
/// </para>
///
/// <para>Cost guard: after each batch the <c>usage</c> block from the
/// Messages API is priced (Sonnet 4.6 input/output rates) and added to
/// a running total. Once the total exceeds
/// <see cref="CoordinatorOptions.AsrCostCapUsd"/> the generator stops
/// and returns a partial manifest.</para>
/// </summary>
public sealed class SonnetAsrCorpusGenerator
{
    /// <summary>USD per million input tokens for Claude Sonnet 4.6.</summary>
    public const decimal InputUsdPerMillion = 3.00m;

    /// <summary>USD per million output tokens for Claude Sonnet 4.6.</summary>
    public const decimal OutputUsdPerMillion = 15.00m;

    private const string ManifestName = "asr-v1";
    private const string AnthropicVersion = "2023-06-01";
    private const string MessagesEndpoint = "v1/messages";

    private readonly HttpClient _http;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly ILogger<SonnetAsrCorpusGenerator> _logger;

    public SonnetAsrCorpusGenerator(
        HttpClient http,
        IOptionsMonitor<CoordinatorOptions> options,
        ILogger<SonnetAsrCorpusGenerator> logger)
    {
        _http = http;
        _options = options;
        _logger = logger;
        if (_http.BaseAddress is null)
        {
            _http.BaseAddress = new Uri("https://api.anthropic.com/");
        }
    }

    public async Task<CorpusManifest> GenerateAsync(
        string outputDirectory,
        int count,
        int examplesPerShard,
        int seed,
        int batchSize = 20,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (examplesPerShard <= 0) throw new ArgumentOutOfRangeException(nameof(examplesPerShard));
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));

        var opts = _options.CurrentValue;
        if (string.IsNullOrWhiteSpace(opts.AnthropicApiKey))
        {
            throw new InvalidOperationException(
                "Coordinator__AnthropicApiKey is not set; cannot query Sonnet for ASR corpus.");
        }

        Directory.CreateDirectory(outputDirectory);

        var shardPrefix = string.IsNullOrWhiteSpace(opts.AsrShardPrefix) ? "asr-v1-" : opts.AsrShardPrefix;
        var systemPrompt = BuildSystemPrompt(seed);
        var userPrompt = BuildUserPrompt(batchSize);

        var shards = new List<CorpusShardInfo>();
        var emitted = 0;
        var shardIndex = 0;
        decimal cumulativeUsd = 0m;
        var capped = false;

        while (emitted < count && !capped)
        {
            var shardTarget = Math.Min(count - emitted, examplesPerShard);
            var shardId = $"{shardPrefix}shard-{shardIndex:D4}";
            var shardPath = Path.Combine(outputDirectory, $"{shardId}.txt");
            var shardLines = new List<string>(shardTarget);

            while (shardLines.Count < shardTarget && !capped)
            {
                var wanted = Math.Min(batchSize, shardTarget - shardLines.Count);
                var (lines, usage) = await RequestBatchAsync(
                    systemPrompt,
                    BuildUserPrompt(wanted),
                    opts,
                    ct).ConfigureAwait(false);

                foreach (var line in lines)
                {
                    if (shardLines.Count >= shardTarget) break;
                    if (IsWellFormed(line)) shardLines.Add(line);
                }

                cumulativeUsd += PriceUsage(usage);
                if (cumulativeUsd >= opts.AsrCostCapUsd)
                {
                    _logger.LogWarning(
                        "ASR generator hit cost cap {Cap:C} (spent {Spent:C}); returning partial manifest.",
                        opts.AsrCostCapUsd,
                        cumulativeUsd);
                    capped = true;
                }
            }

            if (shardLines.Count == 0) break;

            await File.WriteAllLinesAsync(shardPath, shardLines, Encoding.UTF8, ct).ConfigureAwait(false);
            var size = new FileInfo(shardPath).Length;
            shards.Add(new CorpusShardInfo(shardId, shardPath, shardLines.Count, size));
            emitted += shardLines.Count;
            shardIndex++;
        }

        var manifest = new CorpusManifest(
            Name: ManifestName,
            TotalExamples: emitted,
            Seed: seed,
            PoolVersion: "Sonnet-asr-v1",
            Shards: shards);

        var manifestPath = Path.Combine(outputDirectory, $"manifest.{ManifestName}.json");
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(manifestPath, json, Encoding.UTF8, ct).ConfigureAwait(false);

        _logger.LogInformation(
            "Sonnet ASR run complete: emitted={Emitted} shards={Shards} spent={Spent:C}.",
            emitted, shards.Count, cumulativeUsd);

        return manifest;
    }

    public decimal EstimateCostUsd(int count, int batchSize)
    {
        if (count <= 0 || batchSize <= 0) return 0m;
        var batches = (count + batchSize - 1) / batchSize;
        // Rough per-batch estimates: ~800 prompt tokens (system + few-shot) + ~40 tokens per requested line.
        var inputTokens = (long)batches * (800L + 40L * batchSize);
        var outputTokens = (long)count * 60L;
        return ((decimal)inputTokens / 1_000_000m) * InputUsdPerMillion
             + ((decimal)outputTokens / 1_000_000m) * OutputUsdPerMillion;
    }

    private static decimal PriceUsage(AnthropicUsage usage) =>
        ((decimal)usage.InputTokens / 1_000_000m) * InputUsdPerMillion
      + ((decimal)usage.OutputTokens / 1_000_000m) * OutputUsdPerMillion;

    private static bool IsWellFormed(string line)
    {
        if (string.IsNullOrWhiteSpace(line)) return false;
        if (!line.StartsWith("[USER] ", StringComparison.Ordinal)) return false;
        var marker = line.IndexOf(" [INTENT] ", StringComparison.Ordinal);
        if (marker < 0) return false;
        var json = line.Substring(marker + " [INTENT] ".Length);
        if (json.Length == 0 || json[0] != '{') return false;
        try
        {
            using var doc = JsonDocument.Parse(json);
            return doc.RootElement.TryGetProperty("intent", out _);
        }
        catch (JsonException) { return false; }
    }

    private static string BuildSystemPrompt(int seed)
    {
        var rng = new Random(seed);
        var fewShot = new StringBuilder();
        for (var i = 0; i < 6; i++)
        {
            fewShot.AppendLine(TruckMateCorpusGenerator.GenerateExample(rng, CorpusPoolVersion.V2));
        }
        return
            "You are a training-data synthesizer for a trucking voice-assistant SLM. " +
            "Emit ONLY lines in the exact format:\n" +
            "[USER] <noisy ASR transcript> [INTENT] {\"intent\":\"<name>\",\"slots\":{...}}\n" +
            "Rules:\n" +
            "- One example per line, no blank lines, no numbering, no prose, no markdown fences.\n" +
            "- The utterance simulates an over-the-road trucker speaking to a hands-free assistant. " +
            "Include realistic ASR noise: disfluencies (uh, um, like), mis-hearings, and CB-radio shorthand.\n" +
            "- The intent JSON must be valid; slot keys match the examples below.\n" +
            "- Intents: start_trip, stop_trip, navigate, find_poi, route_preference, hos_status, " +
            "hos_break_check, hos_drive_remaining, add_todo, add_expense, eta_query, next_stop_query, " +
            "trip_status, reroute, update_load.\n" +
            "- Draw cities, truck-stop chains, and interstates from US freight-hub vocabulary.\n\n" +
            "Examples:\n" +
            fewShot.ToString();
    }

    private static string BuildUserPrompt(int batchSize) =>
        $"Generate {batchSize} diverse training examples now. " +
        "Output nothing but the lines themselves.";

    private async Task<(List<string> Lines, AnthropicUsage Usage)> RequestBatchAsync(
        string systemPrompt,
        string userPrompt,
        CoordinatorOptions opts,
        CancellationToken ct)
    {
        var body = JsonSerializer.Serialize(new
        {
            model = opts.AnthropicModel,
            max_tokens = 4096,
            system = systemPrompt,
            messages = new[]
            {
                new { role = "user", content = userPrompt },
            },
        });

        const int maxAttempts = 3;
        for (var attempt = 1; attempt <= maxAttempts; attempt++)
        {
            ct.ThrowIfCancellationRequested();
            using var req = new HttpRequestMessage(HttpMethod.Post, MessagesEndpoint);
            req.Headers.Add("x-api-key", opts.AnthropicApiKey);
            req.Headers.Add("anthropic-version", AnthropicVersion);
            req.Content = new StringContent(body, Encoding.UTF8, "application/json");

            HttpResponseMessage resp;
            try
            {
                resp = await _http.SendAsync(req, ct).ConfigureAwait(false);
            }
            catch (HttpRequestException ex) when (attempt < maxAttempts)
            {
                _logger.LogWarning(ex, "Anthropic call failed (attempt {Attempt}); retrying.", attempt);
                await BackoffAsync(attempt, ct).ConfigureAwait(false);
                continue;
            }

            if ((int)resp.StatusCode == 429 ||
                (int)resp.StatusCode == 529 ||
                (int)resp.StatusCode >= 500)
            {
                resp.Dispose();
                if (attempt == maxAttempts)
                {
                    throw new HttpRequestException($"Anthropic API returned {(int)resp.StatusCode} after {maxAttempts} attempts.");
                }
                _logger.LogWarning("Anthropic returned {Status}; retrying (attempt {Attempt}).", (int)resp.StatusCode, attempt);
                await BackoffAsync(attempt, ct).ConfigureAwait(false);
                continue;
            }

            resp.EnsureSuccessStatusCode();
            var payload = await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            resp.Dispose();
            return ParseResponse(payload);
        }

        throw new InvalidOperationException("Unreachable retry loop exit.");
    }

    private static async Task BackoffAsync(int attempt, CancellationToken ct)
    {
        var baseMs = 250 * (1 << (attempt - 1));
        var jitter = Random.Shared.Next(0, 100);
        await Task.Delay(TimeSpan.FromMilliseconds(baseMs + jitter), ct).ConfigureAwait(false);
    }

    private static (List<string> Lines, AnthropicUsage Usage) ParseResponse(string payload)
    {
        using var doc = JsonDocument.Parse(payload);
        var root = doc.RootElement;

        var text = string.Empty;
        if (root.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.Array)
        {
            var sb = new StringBuilder();
            foreach (var block in content.EnumerateArray())
            {
                if (block.TryGetProperty("type", out var typeEl) &&
                    typeEl.GetString() == "text" &&
                    block.TryGetProperty("text", out var textEl))
                {
                    sb.Append(textEl.GetString());
                }
            }
            text = sb.ToString();
        }

        var usage = new AnthropicUsage(0, 0);
        if (root.TryGetProperty("usage", out var usageEl))
        {
            var inputTokens = usageEl.TryGetProperty("input_tokens", out var it) ? it.GetInt64() : 0;
            var outputTokens = usageEl.TryGetProperty("output_tokens", out var ot) ? ot.GetInt64() : 0;
            usage = new AnthropicUsage(inputTokens, outputTokens);
        }

        var lines = text.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(static l => l.Trim())
            .Where(static l => l.Length > 0)
            .ToList();

        return (lines, usage);
    }

    private readonly record struct AnthropicUsage(long InputTokens, long OutputTokens);
}
