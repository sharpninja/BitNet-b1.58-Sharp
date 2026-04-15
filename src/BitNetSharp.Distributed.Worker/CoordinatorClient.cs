using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Thin HTTP client wrapper that talks to the BitNet coordinator's
/// REST surface on behalf of the worker. Owns a long-lived
/// <see cref="HttpClient"/>, fetches JWT access tokens via OAuth 2.0
/// client credentials against the coordinator's Duende
/// IdentityServer, and presents convenience methods for the worker
/// lifecycle: register, heartbeat, try-claim-work, and
/// submit-gradient.
///
/// <para>
/// The class is deliberately dependency-free beyond
/// <see cref="HttpClient"/> / <see cref="HttpMessageHandler"/> so
/// tests can inject a fake handler without spinning up a TestServer.
/// </para>
/// </summary>
internal sealed class CoordinatorClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly WorkerConfig _config;
    private readonly SemaphoreSlim _tokenLock = new(1, 1);
    private readonly TimeProvider _time;
    private AccessToken? _currentToken;
    private readonly bool _ownsHttpClient;

    /// <summary>Scope requested on every token call.</summary>
    public const string WorkerScope = "bitnet-worker";

    public CoordinatorClient(WorkerConfig config, TimeProvider? time = null)
        : this(config, httpClient: CreateDefaultHttpClient(config), ownsHttpClient: true, time)
    {
    }

    internal CoordinatorClient(WorkerConfig config, HttpClient httpClient, bool ownsHttpClient, TimeProvider? time = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(httpClient);
        _config = config;
        _http = httpClient;
        _ownsHttpClient = ownsHttpClient;
        _time = time ?? TimeProvider.System;
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
    /// Ensures a non-expired JWT is held in memory and returns it.
    /// Fetches a fresh one via <c>POST /connect/token</c> with the
    /// <see cref="WorkerConfig.ClientId"/> /
    /// <see cref="WorkerConfig.ClientSecret"/> credentials on a miss
    /// or when the cached token is within 30 seconds of expiry.
    /// </summary>
    public async Task<string> GetAccessTokenAsync(CancellationToken cancellationToken = default)
    {
        await _tokenLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var now = _time.GetUtcNow();
            if (_currentToken is not null && _currentToken.ExpiresAtUtc - now > TimeSpan.FromSeconds(30))
            {
                return _currentToken.Token;
            }

            var form = new List<KeyValuePair<string, string>>
            {
                new("grant_type",    "client_credentials"),
                new("client_id",     _config.ClientId),
                new("client_secret", _config.ClientSecret),
                new("scope",         WorkerScope)
            };

            using var request = new HttpRequestMessage(HttpMethod.Post, "connect/token")
            {
                Content = new FormUrlEncodedContent(form)
            };

            using var response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var payload = await response.Content.ReadFromJsonAsync<TokenResponse>(cancellationToken: cancellationToken).ConfigureAwait(false);
            if (payload is null || string.IsNullOrWhiteSpace(payload.AccessToken))
            {
                throw new InvalidOperationException("Coordinator returned an empty token response.");
            }

            var expiresIn = payload.ExpiresIn > 0 ? payload.ExpiresIn : 3600;
            _currentToken = new AccessToken(
                Token: payload.AccessToken!,
                ExpiresAtUtc: _time.GetUtcNow().AddSeconds(expiresIn));

            return _currentToken.Token;
        }
        finally
        {
            _tokenLock.Release();
        }
    }

    /// <summary>
    /// Forces the cached access token to be discarded so the next
    /// <see cref="GetAccessTokenAsync"/> call re-authenticates.
    /// Called by the caller when a protected endpoint returns 401,
    /// which typically means the operator rotated the client
    /// secret on the coordinator admin page.
    /// </summary>
    public void InvalidateAccessToken()
    {
        _currentToken = null;
    }

    /// <summary>
    /// POSTs the initial <see cref="WorkerRegistrationRequest"/> to
    /// <c>/register</c>, attaching the JWT access token as the
    /// bearer header. Throws on non-2xx responses.
    /// </summary>
    public async Task<WorkerRegistrationResponse> RegisterAsync(
        WorkerRegistrationRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        var token = await GetAccessTokenAsync(cancellationToken).ConfigureAwait(false);

        using var message = new HttpRequestMessage(HttpMethod.Post, "register")
        {
            Content = JsonContent.Create(request)
        };
        message.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);

        using var response = await _http.SendAsync(message, cancellationToken).ConfigureAwait(false);
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

        var token = await GetAccessTokenAsync(cancellationToken).ConfigureAwait(false);
        using var message = new HttpRequestMessage(HttpMethod.Post, "heartbeat")
        {
            Content = JsonContent.Create(request)
        };
        message.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);

        using var response = await _http.SendAsync(message, cancellationToken).ConfigureAwait(false);
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
        var token = await GetAccessTokenAsync(cancellationToken).ConfigureAwait(false);
        using var message = new HttpRequestMessage(HttpMethod.Get, "work");
        message.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);

        using var response = await _http.SendAsync(message, cancellationToken).ConfigureAwait(false);
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
    /// retry/backoff logic.
    /// </summary>
    public async Task<bool> SubmitGradientAsync(
        GradientSubmission submission,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(submission);

        var token = await GetAccessTokenAsync(cancellationToken).ConfigureAwait(false);
        using var message = new HttpRequestMessage(HttpMethod.Post, "gradient")
        {
            Content = JsonContent.Create(submission)
        };
        message.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);

        using var response = await _http.SendAsync(message, cancellationToken).ConfigureAwait(false);
        if (response.IsSuccessStatusCode)
        {
            return true;
        }

        if (response.StatusCode == System.Net.HttpStatusCode.Forbidden
            || response.StatusCode == System.Net.HttpStatusCode.Conflict)
        {
            return false;
        }

        response.EnsureSuccessStatusCode();
        return false; // Unreachable; EnsureSuccessStatusCode throws.
    }

    public void Dispose()
    {
        _tokenLock.Dispose();
        if (_ownsHttpClient)
        {
            _http.Dispose();
        }
    }

    /// <summary>
    /// Internal holder for a currently-cached JWT plus its computed
    /// expiry. Never serialized.
    /// </summary>
    private sealed record AccessToken(string Token, DateTimeOffset ExpiresAtUtc);

    /// <summary>
    /// Minimal wire-format view of Duende IdentityServer's
    /// <c>/connect/token</c> response. Mapped via snake_case property
    /// names because the OAuth 2.0 spec uses snake_case.
    /// </summary>
    private sealed record TokenResponse
    {
        [System.Text.Json.Serialization.JsonPropertyName("access_token")]
        public string? AccessToken { get; init; }

        [System.Text.Json.Serialization.JsonPropertyName("expires_in")]
        public int ExpiresIn { get; init; }

        [System.Text.Json.Serialization.JsonPropertyName("token_type")]
        public string? TokenType { get; init; }

        [System.Text.Json.Serialization.JsonPropertyName("scope")]
        public string? Scope { get; init; }
    }
}
