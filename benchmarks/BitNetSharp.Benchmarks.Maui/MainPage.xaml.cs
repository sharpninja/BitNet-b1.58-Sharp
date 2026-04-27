using System.Text;

namespace BitNetSharp.Benchmarks.Maui;

public partial class MainPage : ContentPage
{
    public MainPage()
    {
        InitializeComponent();
        // Auto-run benchmarks 500 ms after first appearance so adb logcat
        // capture from a remote-launched APK gets results without a tap.
        this.Loaded += async (_, _) =>
        {
            await Task.Delay(500);
            OnRunClicked(this, EventArgs.Empty);
        };
    }

    private async void OnRunClicked(object? sender, EventArgs e)
    {
        RunBtn.IsEnabled = false;
        StatusLabel.Text = "Running...";
        OutputEditor.Text = "";

        var sb = new StringBuilder();
        try
        {
            await Task.Run(() =>
            {
                KvCacheBenchmark.Run(line =>
                {
                    sb.AppendLine(line);
                    Dispatcher.Dispatch(() => OutputEditor.Text = sb.ToString());
                });
            });
            StatusLabel.Text = "Done.";
        }
        catch (Exception ex)
        {
            sb.AppendLine();
            sb.AppendLine($"FAILED: {ex.GetType().Name}: {ex.Message}");
            sb.AppendLine(ex.StackTrace);
            OutputEditor.Text = sb.ToString();
            StatusLabel.Text = "Failed.";
        }
        finally
        {
            RunBtn.IsEnabled = true;
        }
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
