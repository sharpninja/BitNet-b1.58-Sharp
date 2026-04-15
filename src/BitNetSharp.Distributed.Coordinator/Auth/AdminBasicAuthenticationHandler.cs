using System;
using System.Security.Claims;
using System.Text;
using System.Text.Encodings.Web;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using Microsoft.AspNetCore.Authentication;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Auth;

/// <summary>
/// Minimalist HTTP Basic authentication handler that protects the
/// coordinator's <c>/admin/*</c> pages. Credentials are read from
/// <see cref="AdminOptions"/> which itself is bound from environment
/// variables at startup (see
/// <see cref="CoordinatorOptions"/> for naming).
///
/// <para>
/// Why custom instead of a NuGet package: HTTP Basic has only a dozen
/// lines of logic and pulling a third-party auth package for it would
/// add another surface for version skew. The handler is deliberately
/// self-contained so it is easy to audit.
/// </para>
/// </summary>
public sealed class AdminBasicAuthenticationHandler : AuthenticationHandler<AdminBasicAuthenticationOptions>
{
    private readonly IOptionsMonitor<CoordinatorOptions> _coordinatorOptions;

    public AdminBasicAuthenticationHandler(
        IOptionsMonitor<AdminBasicAuthenticationOptions> options,
        ILoggerFactory logger,
        UrlEncoder encoder,
        IOptionsMonitor<CoordinatorOptions> coordinatorOptions)
        : base(options, logger, encoder)
    {
        _coordinatorOptions = coordinatorOptions;
    }

    protected override Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        var coord = _coordinatorOptions.CurrentValue;
        var admin = coord.Admin;

        if (string.IsNullOrWhiteSpace(admin.Password))
        {
            return Task.FromResult(AuthenticateResult.Fail(
                "Coordinator admin password is not configured. Set Coordinator__Admin__Password."));
        }

        if (!Request.Headers.TryGetValue("Authorization", out var authHeader))
        {
            return Task.FromResult(AuthenticateResult.NoResult());
        }

        var headerValue = authHeader.ToString();
        const string prefix = "Basic ";
        if (!headerValue.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(AuthenticateResult.NoResult());
        }

        string decoded;
        try
        {
            var base64 = headerValue[prefix.Length..].Trim();
            var bytes = Convert.FromBase64String(base64);
            decoded = Encoding.UTF8.GetString(bytes);
        }
        catch (FormatException)
        {
            return Task.FromResult(AuthenticateResult.Fail("Malformed Basic credentials."));
        }

        var separator = decoded.IndexOf(':');
        if (separator < 0)
        {
            return Task.FromResult(AuthenticateResult.Fail("Malformed Basic credentials."));
        }

        var username = decoded[..separator];
        var password = decoded[(separator + 1)..];

        var expectedUser = admin.Username;
        var userOk = CryptographicTimingSafeEquals(username, expectedUser);
        var passOk = CryptographicTimingSafeEquals(password, admin.Password);
        if (!userOk || !passOk)
        {
            return Task.FromResult(AuthenticateResult.Fail("Invalid admin credentials."));
        }

        var identity = new ClaimsIdentity(
            [
                new Claim(ClaimTypes.Name, username),
                new Claim(ClaimTypes.Role, "admin")
            ],
            Scheme.Name);
        var principal = new ClaimsPrincipal(identity);
        var ticket = new AuthenticationTicket(principal, Scheme.Name);
        return Task.FromResult(AuthenticateResult.Success(ticket));
    }

    protected override Task HandleChallengeAsync(AuthenticationProperties properties)
    {
        Response.StatusCode = 401;
        Response.Headers["WWW-Authenticate"] = "Basic realm=\"BitNetCoordinator\", charset=\"UTF-8\"";
        return Task.CompletedTask;
    }

    /// <summary>
    /// Length-independent constant-time string comparison so the
    /// handler does not leak username/password prefixes through
    /// timing side channels. .NET's built-in CryptographicOperations
    /// is byte-oriented so we wrap it here.
    /// </summary>
    private static bool CryptographicTimingSafeEquals(string a, string b)
    {
        var aBytes = Encoding.UTF8.GetBytes(a);
        var bBytes = Encoding.UTF8.GetBytes(b);
        // Pad to the same length so the equality check itself runs in
        // constant time regardless of the input lengths.
        var length = Math.Max(aBytes.Length, bBytes.Length);
        var padA = new byte[length];
        var padB = new byte[length];
        Array.Copy(aBytes, padA, aBytes.Length);
        Array.Copy(bBytes, padB, bBytes.Length);
        var equal = System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(padA, padB);
        return equal && aBytes.Length == bBytes.Length;
    }
}

/// <summary>
/// Scheme options for <see cref="AdminBasicAuthenticationHandler"/>.
/// Currently empty — the handler reads everything it needs from
/// <see cref="CoordinatorOptions.Admin"/> — but the class is required
/// by the <see cref="AuthenticationHandler{TOptions}"/> generic
/// constraint.
/// </summary>
public sealed class AdminBasicAuthenticationOptions : AuthenticationSchemeOptions;
