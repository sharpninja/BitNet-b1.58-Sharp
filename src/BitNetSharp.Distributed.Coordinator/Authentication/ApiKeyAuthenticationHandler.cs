using System;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using Microsoft.AspNetCore.Authentication;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Authentication;

/// <summary>
/// Authentication handler that validates requests to the worker
/// endpoints by constant-time-comparing the <c>X-Api-Key</c> request
/// header against the single shared API key configured on the
/// coordinator via <see cref="CoordinatorOptions.WorkerApiKey"/>.
///
/// <para>
/// Every worker presents the same key. The operator controls the key
/// exclusively — rotating it means editing the env var and restarting
/// the coordinator, which immediately locks out every worker with the
/// old key.
/// </para>
///
/// <para>
/// On success the handler stamps a <c>client_id</c> claim taken from
/// the optional <c>X-Worker-Id</c> header so downstream endpoints can
/// identify the calling worker. Workers that omit the header land
/// with an empty <c>client_id</c> which the register/heartbeat
/// handlers treat as a validation failure.
/// </para>
/// </summary>
public sealed class ApiKeyAuthenticationHandler : AuthenticationHandler<ApiKeyAuthenticationOptions>
{
    public const string SchemeName = "ApiKey";
    public const string HeaderName = "X-Api-Key";
    public const string WorkerIdHeaderName = "X-Worker-Id";

    public ApiKeyAuthenticationHandler(
        IOptionsMonitor<ApiKeyAuthenticationOptions> options,
        ILoggerFactory logger,
        UrlEncoder encoder)
        : base(options, logger, encoder)
    {
    }

    protected override Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        if (!Request.Headers.TryGetValue(HeaderName, out var provided) ||
            provided.Count == 0 ||
            string.IsNullOrWhiteSpace(provided[0]))
        {
            return Task.FromResult(AuthenticateResult.NoResult());
        }

        var expected = Options.ExpectedKey?.Invoke() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(expected))
        {
            Logger.LogWarning(
                "X-Api-Key request arrived but Coordinator:WorkerApiKey is not configured. Rejecting.");
            return Task.FromResult(AuthenticateResult.Fail("WorkerApiKey not configured on the coordinator."));
        }

        var expectedBytes = Encoding.UTF8.GetBytes(expected);
        var actualBytes = Encoding.UTF8.GetBytes(provided[0]!);
        if (expectedBytes.Length != actualBytes.Length ||
            !CryptographicOperations.FixedTimeEquals(expectedBytes, actualBytes))
        {
            return Task.FromResult(AuthenticateResult.Fail("Invalid API key."));
        }

        var workerId = Request.Headers.TryGetValue(WorkerIdHeaderName, out var widValues) &&
                       widValues.Count > 0 && !string.IsNullOrWhiteSpace(widValues[0])
            ? widValues[0]!
            : string.Empty;

        var claims = new[]
        {
            new Claim("client_id", workerId),
            new Claim(ClaimTypes.NameIdentifier, workerId),
            new Claim(ClaimTypes.Name, workerId)
        };
        var identity = new ClaimsIdentity(claims, SchemeName);
        var principal = new ClaimsPrincipal(identity);
        var ticket = new AuthenticationTicket(principal, SchemeName);
        return Task.FromResult(AuthenticateResult.Success(ticket));
    }

    protected override Task HandleChallengeAsync(AuthenticationProperties properties)
    {
        Response.StatusCode = 401;
        Response.Headers["WWW-Authenticate"] = $"{SchemeName} realm=\"bitnet-worker\"";
        return Task.CompletedTask;
    }
}

/// <summary>
/// Options for <see cref="ApiKeyAuthenticationHandler"/>. The
/// <see cref="ExpectedKey"/> factory is resolved on every request
/// against the current <see cref="CoordinatorOptions"/> snapshot so an
/// env-var change picked up by <c>IOptionsMonitor</c> takes effect
/// without a full app rebuild.
/// </summary>
public sealed class ApiKeyAuthenticationOptions : AuthenticationSchemeOptions
{
    /// <summary>
    /// Factory returning the currently-configured worker API key. The
    /// handler calls this on every request so operators can re-set the
    /// env var without a process restart when running on a host that
    /// reloads configuration.
    /// </summary>
    public Func<string>? ExpectedKey { get; set; }
}
