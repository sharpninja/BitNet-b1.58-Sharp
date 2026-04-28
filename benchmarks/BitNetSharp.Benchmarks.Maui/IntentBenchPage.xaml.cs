using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using BitNetSharp.Core;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Benchmarks.Maui;

public partial class IntentBenchPage : ContentPage
{
    private const string ModelAssetName = "truckmate-small.tmv1";

    /// <summary>
    /// Fixed bank of TruckMate prompts that exercise every intent family
    /// in the v1 corpus generator. Each tuple is (utterance,
    /// expected_intent_name). The bench records both raw decode
    /// latency and substring-accuracy.
    /// </summary>
    private static readonly (string Utterance, string ExpectedIntent)[] Prompts =
    {
        ("[USER] start my trip to dallas [INTENT]",                 "start_trip"),
        ("[USER] begin trip houston [INTENT]",                      "start_trip"),
        ("[USER] stop trip [INTENT]",                               "stop_trip"),
        ("[USER] take me to memphis [INTENT]",                      "navigate"),
        ("[USER] navigate to atlanta [INTENT]",                     "navigate"),
        ("[USER] find me a truck stop near nashville [INTENT]",     "find_poi"),
        ("[USER] where's the nearest fuel [INTENT]",                "find_poi"),
        ("[USER] avoid tolls [INTENT]",                             "route_preference"),
        ("[USER] no mountain passes [INTENT]",                      "route_preference"),
        ("[USER] check my hours [INTENT]",                          "hos_status"),
        ("[USER] when do i need to stop for my break [INTENT]",     "hos_break_check"),
        ("[USER] am i good to keep driving [INTENT]",               "hos_drive_remaining"),
        ("[USER] add todo check tire pressure [INTENT]",            "add_todo"),
        ("[USER] add expense 45.50 dollars for fuel [INTENT]",      "add_expense"),
        ("[USER] what's my eta [INTENT]",                           "eta_query"),
        ("[USER] how far to denver [INTENT]",                       "eta_query"),
        ("[USER] what's my next stop [INTENT]",                     "next_stop_query"),
        ("[USER] reroute around traffic [INTENT]",                  "reroute"),
        ("[USER] there's construction on i-40 reroute me [INTENT]", "reroute"),
        ("[USER] mark load LD-12345 as delivered [INTENT]",         "update_load"),
    };

    private readonly StringBuilder _output = new();

    public IntentBenchPage()
    {
        InitializeComponent();
    }

    private void Append(string line)
    {
        _output.AppendLine(line);
        Console.WriteLine($"BENCH_INTENT: {line}");
        Dispatcher.Dispatch(() => OutputEditor.Text = _output.ToString());
    }

    private void SetProgress(double fraction, string label)
    {
        Dispatcher.Dispatch(() =>
        {
            LoadProgress.IsVisible = true;
            LoadProgress.Progress = Math.Clamp(fraction, 0d, 1d);
            ProgressLabel.Text = label;
        });
    }

    private void HideProgress()
    {
        Dispatcher.Dispatch(() =>
        {
            LoadProgress.IsVisible = false;
            ProgressLabel.Text = "";
        });
    }

    private async void OnRunClicked(object? sender, EventArgs e)
    {
        RunBtn.IsEnabled = false;
        StatusLabel.Text = "Loading + benching...";
        _output.Clear();
        OutputEditor.Text = "";
        SetProgress(0, "Starting...");

        var pickedKv = KvPicker.SelectedIndex == 1 ? KvCacheQuantization.Int8 : KvCacheQuantization.Fp32;
        if (!int.TryParse(MaxTokensEntry.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var maxTokens) || maxTokens <= 0)
        {
            maxTokens = 24;
        }

        await Task.Run(async () =>
        {
            try
            {
                Environment.SetEnvironmentVariable(
                    "BITNETSHARP_KV_CACHE_QUANTIZATION",
                    pickedKv.ToString());

                var modelPath = await EnsureModelExtractedAsync();
                Append($"Loading TMV1 from {modelPath} ({new FileInfo(modelPath).Length:N0} bytes)");

                SetProgress(0.1, "Reading TMV1 header + flat-vector...");
                var loadSw = Stopwatch.StartNew();
                var (header, flat) = TruckMateModelStore.Load(modelPath);
                Append($"  vocab={header.VocabSize} dim={header.Dimension} layers={header.LayerCount} heads={header.HeadCount} kvHeads={header.KvHeadCount} headDim={header.Dimension / header.HeadCount}");

                var cfg = new BitNetConfig(
                    vocabSize: header.VocabSize,
                    dimension: header.Dimension,
                    hiddenDimension: header.HiddenDimension,
                    layerCount: header.LayerCount,
                    headCount: header.HeadCount,
                    maxSequenceLength: header.MaxSequenceLength,
                    kvHeadCount: header.KvHeadCount,
                    kvCacheQuantization: pickedKv);

                SetProgress(0.4, "Constructing transformer (skipRandomInit)...");
                var transformer = new BitNetTransformer(
                    cfg,
                    NullLogger<BitNetTransformer>.Instance,
                    seed: 42,
                    skipRandomInit: true);

                SetProgress(0.85, "Unpacking weights...");
                FlatParameterPack.Unpack(transformer, flat);

                SetProgress(0.92, "Building tokenizer...");
                var tokenizer = BuildTokenizerFromHeader(header);

                var model = new WordLevelInferenceModel(transformer, tokenizer)
                {
                    MaxResponseTokens = maxTokens,
                    SuppressEosAndUnk = true, // smoke build argmaxes to EOS too aggressively otherwise
                };
                loadSw.Stop();
                Append($"Model ready in {loadSw.Elapsed.TotalMilliseconds:F0} ms (KV={pickedKv})");
                Append($"AdvSimd={System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported} Vector<float>.HW={System.Numerics.Vector.IsHardwareAccelerated}");
                Append("");
                Append($"step | ttft_ms | tokens | per_dec_ms | hit | utterance -> response");
                Append(new string('-', 80));

                long sumTtft = 0;
                long sumTotal = 0;
                int sumTokens = 0;
                int hits = 0;
                var perPromptDecodeMs = new List<double>(Prompts.Length);

                for (var i = 0; i < Prompts.Length; i++)
                {
                    var (utterance, expectedIntent) = Prompts[i];
                    SetProgress(0.85 + 0.15 * i / (double)Prompts.Length,
                        $"Bench {i + 1}/{Prompts.Length}");

                    var promptSw = Stopwatch.StartNew();
                    long ttftMs = 0;
                    var first = true;
                    var sumDecode = 0d;
                    var step = 0;
                    var responseText = new StringBuilder();
                    await foreach (var token in model.StreamGenerateAsync(utterance, maxTokens))
                    {
                        if (first)
                        {
                            ttftMs = promptSw.ElapsedMilliseconds;
                            first = false;
                        }
                        sumDecode += token.DecodeMs;
                        responseText.Append(token.TokenText);
                        step++;
                    }
                    promptSw.Stop();

                    var totalMs = promptSw.ElapsedMilliseconds;
                    var perDecMs = step > 1 ? sumDecode / (step - 1) : 0d;
                    var responseStr = responseText.ToString();
                    var hit = responseStr.Contains(expectedIntent, StringComparison.OrdinalIgnoreCase);
                    if (hit) hits++;
                    sumTtft += ttftMs;
                    sumTotal += totalMs;
                    sumTokens += step;
                    perPromptDecodeMs.Add(perDecMs);

                    var snippet = responseStr.Length > 60 ? responseStr[..60] + "…" : responseStr;
                    Append($"{i + 1,4} | {ttftMs,7} | {step,6} | {perDecMs,10:F1} | {(hit ? " Y " : " . ")} | {expectedIntent,-22} -> {snippet.Replace('\n', ' ')}");
                }

                Append(new string('-', 80));
                Append($"prompts={Prompts.Length} hits={hits} accuracy={(double)hits / Prompts.Length * 100:F1}%");
                Append($"avg_ttft_ms={sumTtft / (double)Prompts.Length:F1}");
                Append($"avg_total_ms={sumTotal / (double)Prompts.Length:F1}");
                Append($"avg_tokens_per_prompt={sumTokens / (double)Prompts.Length:F1}");
                if (perPromptDecodeMs.Count > 0)
                {
                    var sorted = perPromptDecodeMs.OrderBy(x => x).ToList();
                    var p50 = sorted[sorted.Count / 2];
                    var p95 = sorted[Math.Min(sorted.Count - 1, (int)(sorted.Count * 0.95))];
                    Append($"per_decode_ms p50={p50:F1} p95={p95:F1} max={sorted[^1]:F1}");
                    Append($"throughput_tok_per_sec p50={(p50 > 0 ? 1000d / p50 : 0d):F2} p95={(p95 > 0 ? 1000d / p95 : 0d):F2}");
                }
                Append($"KV={pickedKv}");
            }
            catch (Exception ex)
            {
                Append($"BENCH FAILED: {ex.GetType().Name}: {ex.Message}");
                Append(ex.StackTrace ?? "");
            }
        });

        StatusLabel.Text = "Done.";
        HideProgress();
        RunBtn.IsEnabled = true;
    }

    private static WordLevelTokenizer BuildTokenizerFromHeader(TruckMateModelStore.Header header)
    {
        var tempPath = Path.Combine(FileSystem.CacheDirectory, $"tmvocab-{Guid.NewGuid():N}.json");
        try
        {
            var arr = header.Vocabulary is string[] direct ? direct : header.Vocabulary.ToArray();
            File.WriteAllText(tempPath, JsonSerializer.Serialize(arr), Encoding.UTF8);
            return WordLevelTokenizer.LoadFromFile(tempPath);
        }
        finally
        {
            try { File.Delete(tempPath); } catch { /* best-effort */ }
        }
    }

    /// <summary>
    /// Same extract path as ChatPage. Re-uses the cached file if Chat
    /// already extracted it. Two-tab cache hit on second tap.
    /// </summary>
    private async Task<string> EnsureModelExtractedAsync()
    {
        var dst = Path.Combine(FileSystem.AppDataDirectory, ModelAssetName);
        var sentinel = dst + ".complete";
        if (File.Exists(sentinel) && File.Exists(dst))
        {
            return dst;
        }

        Append($"Extracting {ModelAssetName} from APK to {dst}...");
        SetProgress(0, "Extracting from APK...");
        var sw = Stopwatch.StartNew();
        const int bufferSize = 1 << 22; // 4 MiB
        var buffer = new byte[bufferSize];
        long copied = 0;
        long? totalBytes = null;
        await using (var src = await FileSystem.OpenAppPackageFileAsync(ModelAssetName))
        {
            try { totalBytes = src.Length; } catch (NotSupportedException) { /* unknown */ }

            await using var fs = new FileStream(dst, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize);
            int read;
            while ((read = await src.ReadAsync(buffer)) > 0)
            {
                await fs.WriteAsync(buffer.AsMemory(0, read));
                copied += read;
                if (totalBytes is { } total && total > 0)
                {
                    SetProgress((double)copied / total,
                        $"Extracting from APK: {copied / 1_048_576} / {total / 1_048_576} MiB");
                }
            }
        }
        sw.Stop();
        File.WriteAllText(sentinel, "ok");
        Append($"  extracted {new FileInfo(dst).Length:N0} bytes in {sw.Elapsed.TotalSeconds:F1} s");
        return dst;
    }

    private async void OnCopyClicked(object? sender, EventArgs e)
    {
        if (string.IsNullOrEmpty(OutputEditor.Text))
        {
            return;
        }
        await Clipboard.SetTextAsync(OutputEditor.Text);
        StatusLabel.Text = "Copied.";
    }
}
