using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Thin HTTP client wrapper that talks to the BitNet coordinator's
/// REST surface on behalf of the worker. Owns a long-lived
/// <see cref="HttpClient"/> pre-loaded with the shared
/// <c>X-Api-Key</c> and <c>X-Worker-Id</c> request headers, and
/// presents convenience methods for the worker lifecycle: register,
/// heartbeat, try-claim-work, and submit-gradient.
///
/// <para>
/// Authentication is a single shared API key set by the operator on
/// the coordinator via the <c>Coordinator__WorkerApiKey</c> env var.
/// Every worker sends the same key — there is no token exchange,
/// no refresh, no rotation protocol. When the operator rotates the
/// key on the coordinator and restarts it, every worker with the
/// old key starts getting 401s and must be redeployed.
/// </para>
///
/// <para>
/// The class is deliberately dependency-free beyond
/// <see cref="HttpClient"/> / <see cref="HttpMessageHandler"/> so
/// tests can inject a fake handler without spinning up a TestServer.
/// </para>
/// </summary>
internal sealed class CoordinatorClient : IDisposable
{
    /// <summary>Header the coordinator's ApiKey handler reads.</summary>
    public const string ApiKeyHeader = "X-Api-Key";

    /// <summary>Header the coordinator stamps as the <c>client_id</c> claim.</summary>
    public const string WorkerIdHeader = "X-Worker-Id";

    private readonly HttpClient _http;
    private readonly WorkerConfig _config;
    private readonly bool _ownsHttpClient;

    public CoordinatorClient(WorkerConfig config)
        : this(config, httpClient: CreateDefaultHttpClient(config), ownsHttpClient: true)
    {
    }

    internal CoordinatorClient(WorkerConfig config, HttpClient httpClient, bool ownsHttpClient)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(httpClient);
        _config = config;
        _http = httpClient;
        _ownsHttpClient = ownsHttpClient;

        // Both worker auth headers are constant for the lifetime of
        // the process — install them once on the default headers so
        // every request picks them up. Clear any pre-seeded value so
        // test harnesses that re-use a handler do not double up.
        _http.DefaultRequestHeaders.Remove(ApiKeyHeader);
        _http.DefaultRequestHeaders.Remove(WorkerIdHeader);
        _http.DefaultRequestHeaders.Add(ApiKeyHeader, config.ApiKey);
        _http.DefaultRequestHeaders.Add(WorkerIdHeader, config.WorkerId);
    }

    private static HttpClient CreateDefaultHttpClient(WorkerConfig config)
    {
        var client = new HttpClient
        {
            BaseAddress = config.CoordinatorUrl,
            Timeout = TimeSpan.FromSeconds(30)
        };
        client.DefaultRequestHeaders.UserAgent.ParseAdd($"BitNetSharpWorker/1.0 ({config.WorkerName})");
        return client;
    }

    /// <summary>
    /// POSTs the initial <see cref="WorkerRegistrationRequest"/> to
    /// <c>/register</c>. Auth headers are applied automatically.
    /// Throws on non-2xx responses.
    /// </summary>
    public async Task<WorkerRegistrationResponse> RegisterAsync(
        WorkerRegistrationRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        using var response = await _http
            .PostAsJsonAsync("register", request, cancellationToken)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var payload = await response.Content
            .ReadFromJsonAsync<WorkerRegistrationResponse>(cancellationToken: cancellationToken)
            .ConfigureAwait(false);
        if (payload is null)
        {
            throw new InvalidOperationException("Coordinator returned an empty /register response.");
        }

        return payload;
    }

    /// <summary>
    /// POSTs a heartbeat to <c>/heartbeat</c>. Returns the parsed
    /// response on success; null on 410 Gone which signals the
    /// worker must re-register.
    /// </summary>
    public async Task<HeartbeatResponse?> SendHeartbeatAsync(
        HeartbeatRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        using var response = await _http
            .PostAsJsonAsync("heartbeat", request, cancellationToken)
            .ConfigureAwait(false);

        if (response.StatusCode == System.Net.HttpStatusCode.Gone)
        {
            return null;
        }

        response.EnsureSuccessStatusCode();
        return await response.Content
            .ReadFromJsonAsync<HeartbeatResponse>(cancellationToken: cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// GETs <c>/work</c> and returns the parsed
    /// <see cref="WorkTaskAssignment"/> on a 200 response or
    /// <c>null</c> when the coordinator returns 204 No Content
    /// (empty queue).
    /// </summary>
    public async Task<WorkTaskAssignment?> TryClaimWorkAsync(CancellationToken cancellationToken = default)
    {
        using var response = await _http
            .GetAsync("work", cancellationToken)
            .ConfigureAwait(false);

        if (response.StatusCode == System.Net.HttpStatusCode.NoContent)
        {
            return null;
        }

        response.EnsureSuccessStatusCode();
        return await response.Content
            .ReadFromJsonAsync<WorkTaskAssignment>(cancellationToken: cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// POSTs the completed <see cref="GradientSubmission"/> to
    /// <c>/gradient</c>. Returns <c>true</c> when the coordinator
    /// accepted the submission (200), <c>false</c> when it rejected
    /// the submission due to ownership mismatch (403) or because the
    /// task was already recycled (409). Throws on any other non-2xx
    /// status so transient failures bubble up to the caller's
    /// retry/backoff logic; the response body is included in the
    /// exception message so operators can diagnose 400s without
    /// needing to attach a network sniffer to the worker container.
    /// </summary>
    public async Task<bool> SubmitGradientAsync(
        GradientSubmission submission,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(submission);

        using var response = await _http
            .PostAsJsonAsync("gradient", submission, cancellationToken)
            .ConfigureAwait(false);

        if (response.IsSuccessStatusCode)
        {
            return true;
        }

        if (response.StatusCode == System.Net.HttpStatusCode.Forbidden
            || response.StatusCode == System.Net.HttpStatusCode.Conflict)
        {
            return false;
        }

        var body = string.Empty;
        try
        {
            body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            // Best effort — we still want to throw with status code.
        }

        throw new HttpRequestException(
            $"Gradient submission rejected by coordinator: {(int)response.StatusCode} {response.ReasonPhrase}. Body: {body}");
    }

    /// <summary>
    /// GETs a weight blob from the absolute URL provided on a
    /// <see cref="WorkTaskAssignment"/>. Returns the raw bytes so the
    /// caller can hand them to
    /// <see cref="WeightBlobCodec.TryDecode(System.ReadOnlySpan{byte}, out long, out float[], out string?)"/>.
    /// Throws on non-2xx responses.
    /// </summary>
    public async Task<byte[]> DownloadWeightsAsync(
        string weightUrl,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(weightUrl);

        using var response = await _http
            .GetAsync(weightUrl, cancellationToken)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content
            .ReadAsByteArrayAsync(cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Low-level send for the log sink that needs to attach its
    /// own body. Auth headers ride on the client's DefaultRequestHeaders
    /// so callers do not need to add them explicitly. Exposes the raw
    /// <see cref="HttpResponseMessage"/> so the caller can handle
    /// status codes however it pleases.
    /// </summary>
    internal async Task<HttpResponseMessage> SendRawAsync(
        HttpRequestMessage request,
        CancellationToken cancellationToken = default)
    {
        return await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _http.Dispose();
        }
    }
}
