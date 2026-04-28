using System.Diagnostics;
using System.Globalization;
using System.Text;
using BitNetSharp.Core;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
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
        await Task.Run(() =>
        {
            try
            {
                Environment.SetEnvironmentVariable(
                    "BITNETSHARP_KV_CACHE_QUANTIZATION",
                    pickedKv.ToString());

                var modelPath = EnsureModelExtractedAsync().GetAwaiter().GetResult();
                Append($"Loading TMV1 from {modelPath}");
                Append($"  size on disk: {new FileInfo(modelPath).Length:N0} bytes");

                SetProgress(0, "Reading TMV1 header + flat-vector...");
                var sw = Stopwatch.StartNew();
                var (header, flat) = TruckMateModelStore.Load(modelPath);
                var ioMs = sw.Elapsed.TotalMilliseconds;
                Append($"  header_io_ms={ioMs:F0} vocab={header.VocabSize} flat_floats={flat.Length:N0}");

                SetProgress(0.4, "Building config...");
                var cfg = new BitNetConfig(
                    vocabSize: header.VocabSize,
                    dimension: header.Dimension,
                    hiddenDimension: header.HiddenDimension,
                    layerCount: header.LayerCount,
                    headCount: header.HeadCount,
                    maxSequenceLength: header.MaxSequenceLength,
                    kvHeadCount: header.KvHeadCount,
                    kvCacheQuantization: pickedKv);

                SetProgress(0.5, "Constructing transformer (skipRandomInit)...");
                // skipRandomInit=true: BitLinear ctors allocate zero-filled
                // ternary buffers; FlatParameterPack.Unpack overwrites all
                // weights immediately after. No multi-GB random pass.
                var ctorProgress = new Progress<double>(p =>
                    SetProgress(0.5 + 0.3 * p, $"Constructing transformer ({p * 100:F0}%)"));
                _model = new BitNetPaperModel(
                    new BitNetOptions(header.Vocabulary, VerbosityLevel.Quiet, MaxResponseTokens: 32),
                    NullLogger<BitNetPaperModel>.Instance,
                    NullLoggerFactory.Instance,
                    config: cfg,
                    seed: 42,
                    constructionProgress: ctorProgress,
                    skipRandomInit: true);

                SetProgress(0.85, "Unpacking weights...");
                FlatParameterPack.Unpack(_model.Transformer, flat);
                sw.Stop();

                _modelKv = _model.Config.KvCacheQuantization;
                Append($"Model loaded in {sw.Elapsed.TotalMilliseconds:F0} ms");
                Append($"  KV={_modelKv} (requested={pickedKv})");
                Append($"  vocab={_model.Config.VocabSize} dim={_model.Config.Dimension} layers={_model.Config.LayerCount} heads={_model.Config.HeadCount} kvHeads={_model.Config.KvHeadCount} headDim={_model.Config.HeadDimension}");
                Append($"  AdvSimd.IsSupported={System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported} Vector<float>.IsHardwareAccelerated={System.Numerics.Vector.IsHardwareAccelerated}");
                Append($"  resident_bytes={_model.EstimateResidentParameterBytes():N0}");
                Append($"  GC.GetTotalMemory={GC.GetTotalMemory(false):N0}");
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

    private const string ModelAssetName = "truckmate-small.tmv1";

    /// <summary>
    /// Copies the MauiAsset truckmate-small.tmv1 to the app's private files
    /// directory on first launch. Tens of MiB, fast (one-shot, sentinel
    /// guard). Subsequent launches are instant cache hits.
    /// </summary>
    private async Task<string> EnsureModelExtractedAsync()
    {
        var dst = Path.Combine(FileSystem.AppDataDirectory, ModelAssetName);
        var sentinel = dst + ".complete";
        if (File.Exists(sentinel) && File.Exists(dst))
        {
            Append($"TMV1 cache hit at {dst}");
            SetProgress(0, "TMV1 cached; preparing reader...");
            return dst;
        }

        Append($"Extracting {ModelAssetName} from APK to {dst}...");
        SetProgress(0, "Extracting from APK...");
        var sw = Stopwatch.StartNew();

        const int bufferSize = 1 << 22; // 4 MiB
        const long reportEvery = 4L * 1024 * 1024;
        var buffer = new byte[bufferSize];
        long copied = 0;
        long lastReport = 0;
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
            maxTokens = 32;
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
