using System.Diagnostics;
using System.Globalization;
using System.Text;
using BitNetSharp.Core;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Benchmarks.Maui;

public partial class ChatPage : ContentPage
{
    private BitNetPaperModel? _model;
    private KvCacheQuantization _modelKv;
    private readonly StringBuilder _output = new();

    public ChatPage()
    {
        InitializeComponent();
    }

    private void Append(string line)
    {
        _output.AppendLine(line);
        Console.WriteLine($"BENCH_CHAT: {line}");
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

    private async void OnLoadClicked(object? sender, EventArgs e)
    {
        LoadBtn.IsEnabled = false;
        StatusLabel.Text = "Loading model...";
        _output.Clear();
        OutputEditor.Text = "";
        SetProgress(0, "Starting...");

        var pickedKv = KvPicker.SelectedIndex == 1 ? KvCacheQuantization.Int8 : KvCacheQuantization.Fp32;
        await Task.Run(async () =>
        {
            try
            {
                Environment.SetEnvironmentVariable(
                    "BITNETSHARP_KV_CACHE_QUANTIZATION",
                    pickedKv.ToString());

                var ggufPath = await EnsureGgufExtractedAsync();
                Append($"Loading GGUF from {ggufPath}");
                Append($"  size on disk: {new FileInfo(ggufPath).Length:N0} bytes");

                SetProgress(0, "Parsing GGUF header + tensors...");
                var loadProgress = new Progress<double>(p =>
                    SetProgress(p, $"Loading tensors ({p * 100:F0}%)"));

                var sw = Stopwatch.StartNew();
                _model = BitNetPaperGguf.Load(
                    ggufPath,
                    Microsoft.Extensions.Logging.Abstractions.NullLogger<BitNetPaperModel>.Instance,
                    Microsoft.Extensions.Logging.Abstractions.NullLoggerFactory.Instance,
                    VerbosityLevel.Quiet,
                    progress: loadProgress);
                sw.Stop();
                _modelKv = _model.Config.KvCacheQuantization;
                Append($"Model loaded in {sw.Elapsed.TotalMilliseconds:F0} ms");
                Append($"  KV={_modelKv} (requested={pickedKv})");
                Append($"  vocab={_model.Config.VocabSize} dim={_model.Config.Dimension} layers={_model.Config.LayerCount} heads={_model.Config.HeadCount} kvHeads={_model.Config.KvHeadCount}");
                Append($"  AdvSimd.IsSupported={System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported} Vector<float>.IsHardwareAccelerated={System.Numerics.Vector.IsHardwareAccelerated}");
                Append($"  Working set bytes: GC.GetTotalMemory={GC.GetTotalMemory(false):N0}");
                Append("");
            }
            catch (Exception ex)
            {
                Append($"LOAD FAILED: {ex.GetType().Name}: {ex.Message}");
                Append(ex.StackTrace ?? "");
            }
        });

        StatusLabel.Text = _model is null ? "Load failed." : "Ready. Enter prompt and Send.";
        HideProgress();
        LoadBtn.IsEnabled = true;
        SendBtn.IsEnabled = _model is not null;
    }

    private const string GgufAssetName = "bonsai.bitnetsharp.gguf";

    /// <summary>
    /// Copies the MauiAsset gguf to the app's private files directory on
    /// first launch. Uses a sentinel marker file ('.complete') instead of a
    /// size check so we never read the asset stream twice or copy on every
    /// load tap. 4 MiB CopyToAsync buffer matches phone NAND page sizes.
    /// Subsequent launches: instant cache hit.
    /// </summary>
    private async Task<string> EnsureGgufExtractedAsync()
    {
        var dst = Path.Combine(FileSystem.AppDataDirectory, GgufAssetName);
        var sentinel = dst + ".complete";
        if (File.Exists(sentinel) && File.Exists(dst))
        {
            Append($"GGUF cache hit at {dst}");
            SetProgress(0, "GGUF cached; preparing reader...");
            return dst;
        }

        Append($"Extracting {GgufAssetName} from APK to {dst}...");
        SetProgress(0, "Extracting from APK...");
        var sw = Stopwatch.StartNew();

        // Buffered manual copy with byte-count progress reporting. Update UI
        // every 16 MiB to avoid Dispatcher flooding.
        const int bufferSize = 1 << 22; // 4 MiB
        const long reportEvery = 16L * 1024 * 1024;
        var buffer = new byte[bufferSize];
        long copied = 0;
        long lastReport = 0;
        long? totalBytes = null;
        await using (var src = await FileSystem.OpenAppPackageFileAsync(GgufAssetName))
        {
            try { totalBytes = src.Length; } catch (NotSupportedException) { /* unknown */ }

            await using var fs = new FileStream(dst, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize);
            int read;
            while ((read = await src.ReadAsync(buffer)) > 0)
            {
                await fs.WriteAsync(buffer.AsMemory(0, read));
                copied += read;
                if (copied - lastReport >= reportEvery)
                {
                    lastReport = copied;
                    if (totalBytes is { } total && total > 0)
                    {
                        SetProgress((double)copied / total,
                            $"Extracting from APK: {copied / 1_048_576} / {total / 1_048_576} MiB");
                    }
                    else
                    {
                        SetProgress(0,
                            $"Extracting from APK: {copied / 1_048_576} MiB");
                    }
                }
            }
        }
        sw.Stop();
        File.WriteAllText(sentinel, "ok");
        Append($"  extracted {new FileInfo(dst).Length:N0} bytes in {sw.Elapsed.TotalSeconds:F1} s");
        SetProgress(1, "Extract complete.");
        return dst;
    }

    private static BitNetPaperModel BuildBootstrapInt8()
    {
        // Mirror BitNetPaperModel.CreateDefault but flip KvCacheQuantization=Int8.
        // Default vocab pulled from BitNetBootstrap defaults (small word list).
        var defaultModel = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Quiet);
        var srcConfig = defaultModel.Config;
        var int8Config = new BitNetConfig(
            vocabSize: srcConfig.VocabSize,
            dimension: srcConfig.Dimension,
            hiddenDimension: srcConfig.HiddenDimension,
            layerCount: srcConfig.LayerCount,
            headCount: srcConfig.HeadCount,
            maxSequenceLength: srcConfig.MaxSequenceLength,
            rmsNormEpsilon: srcConfig.RmsNormEpsilon,
            kvHeadCount: srcConfig.KvHeadCount,
            ropeTheta: srcConfig.RopeTheta,
            kvCacheQuantization: KvCacheQuantization.Int8);
        return new BitNetPaperModel(
            defaultModel.Options,
            NullLogger<BitNetPaperModel>.Instance,
            NullLoggerFactory.Instance,
            int8Config,
            seed: 42);
    }

    private async void OnSendClicked(object? sender, EventArgs e)
    {
        if (_model is null)
        {
            return;
        }
        var prompt = (PromptEntry.Text ?? "").Trim();
        if (string.IsNullOrEmpty(prompt))
        {
            StatusLabel.Text = "Empty prompt.";
            return;
        }

        if (!int.TryParse(MaxTokensEntry.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var maxTokens) || maxTokens <= 0)
        {
            maxTokens = 16;
        }

        SendBtn.IsEnabled = false;
        StatusLabel.Text = "Generating...";

        await Task.Run(async () =>
        {
            try
            {
                Append($"Prompt: {prompt}");
                Append($"max_tokens={maxTokens}");
                Append("");

                var totalSw = Stopwatch.StartNew();
                long ttftMs = 0;
                var first = true;
                var step = 0;
                var sumForward = 0d;
                var sumSelect = 0d;
                var sumDecode = 0d;
                var responseText = new StringBuilder();

                await foreach (var token in _model.StreamGenerateAsync(prompt, maxTokens))
                {
                    if (first)
                    {
                        ttftMs = totalSw.ElapsedMilliseconds;
                        first = false;
                    }
                    sumForward += token.ForwardMs;
                    sumSelect += token.SelectMs;
                    sumDecode += token.DecodeMs;
                    responseText.Append(token.TokenText);

                    Append($"step={token.Step,2} fwd={token.ForwardMs,7:F1} sel={token.SelectMs,5:F2} dec={token.DecodeMs,7:F1} tok={token.TokenId,5} text=\"{token.TokenText}\"");
                    step++;
                }
                totalSw.Stop();

                Append("");
                Append($"Response: \"{responseText}\"");
                Append($"total_ms={totalSw.ElapsedMilliseconds} TTFT_ms={ttftMs} steps={step}");
                if (step > 0)
                {
                    Append($"avg_forward_ms={sumForward / step:F2} avg_select_ms={sumSelect / step:F2} avg_decode_ms={sumDecode / step:F2}");
                    var perTok = step > 1 ? sumDecode / (step - 1) : 0;
                    Append($"per_decode_token_ms={perTok:F2} (decode_dur / (step-1))");
                }
                Append($"KV={_modelKv}");
            }
            catch (Exception ex)
            {
                Append($"GEN FAILED: {ex.GetType().Name}: {ex.Message}");
                Append(ex.StackTrace ?? "");
            }
        });

        StatusLabel.Text = "Done.";
        SendBtn.IsEnabled = true;
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
