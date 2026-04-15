using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Updates the mtime of a small sentinel file on every heartbeat cycle so
/// Docker (and systemd) health checks can tell whether the worker process
/// is actually making progress, not just "still running". The file is kept
/// under <c>/tmp</c> by default so it is discarded on container restart.
/// </summary>
internal sealed class HealthBeacon
{
    private readonly string _path;

    public HealthBeacon(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _path = path;
        var directory = Path.GetDirectoryName(_path);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // Write the initial beacon so health checks during the brief startup
        // calibration window do not flag the container as unhealthy.
        Touch();
    }

    /// <summary>
    /// Bumps the mtime of the beacon file to "now". Errors are swallowed
    /// because a transient filesystem hiccup must never kill the worker —
    /// the Docker healthcheck will mark it unhealthy on its own if the
    /// problem persists.
    /// </summary>
    public void Touch()
    {
        try
        {
            if (File.Exists(_path))
            {
                File.SetLastWriteTimeUtc(_path, DateTime.UtcNow);
            }
            else
            {
                File.WriteAllText(_path, DateTime.UtcNow.ToString("O"));
            }
        }
        catch
        {
            // Intentionally swallowed.
        }
    }

    /// <summary>
    /// Runs an await-cooperative loop that bumps the beacon on the supplied
    /// interval until <paramref name="cancellationToken"/> fires. Returned
    /// task completes cleanly on cancellation.
    /// </summary>
    public async Task RunAsync(TimeSpan interval, CancellationToken cancellationToken)
    {
        if (interval <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(interval), "Heartbeat interval must be positive.");
        }

        using var timer = new PeriodicTimer(interval);
        try
        {
            while (await timer.WaitForNextTickAsync(cancellationToken).ConfigureAwait(false))
            {
                Touch();
            }
        }
        catch (OperationCanceledException)
        {
            // Expected on graceful shutdown.
        }
    }
}
