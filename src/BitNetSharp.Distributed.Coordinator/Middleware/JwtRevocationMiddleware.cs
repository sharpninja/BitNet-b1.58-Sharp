using System;
using System.Globalization;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Middleware;

/// <summary>
/// Custom ASP.NET Core middleware that runs AFTER
/// <see cref="Microsoft.AspNetCore.Authentication.AuthenticationMiddleware"/>
/// validates the JWT and rejects any token whose <c>iat</c> (issued-at)
/// claim is older than the revocation timestamp stored in
/// <see cref="SqliteClientRevocationStore"/> for the token's
/// <c>client_id</c>.
///
/// <para>
/// This is the "immediate expiration" story for API-key rotation:
/// bump the revocation timestamp and every token already in the wild
/// for that client fails its next request. No natural JWT-expiry wait,
/// no fleet-wide restart.
/// </para>
///
/// <para>
/// The middleware is a no-op for unauthenticated requests — it trusts
/// the earlier <c>[Authorize]</c> attribute or endpoint metadata to
/// require authentication on the protected routes. Its job is ONLY
/// the revocation check.
/// </para>
/// </summary>
public sealed class JwtRevocationMiddleware
{
    private readonly RequestDelegate _next;
    private readonly SqliteClientRevocationStore _revocations;
    private readonly ILogger<JwtRevocationMiddleware> _logger;

    public JwtRevocationMiddleware(
        RequestDelegate next,
        SqliteClientRevocationStore revocations,
        ILogger<JwtRevocationMiddleware> logger)
    {
        _next = next;
        _revocations = revocations;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        var user = context.User;
        if (user.Identity?.IsAuthenticated == true)
        {
            var clientId = user.FindFirst("client_id")?.Value;
            var iatClaim = user.FindFirst("iat")?.Value;

            if (!string.IsNullOrWhiteSpace(clientId) && !string.IsNullOrWhiteSpace(iatClaim))
            {
                if (long.TryParse(iatClaim, NumberStyles.Integer, CultureInfo.InvariantCulture, out var iatUnix))
                {
                    var issuedAt = DateTimeOffset.FromUnixTimeSeconds(iatUnix);
                    if (_revocations.IsIssuedBeforeRevocation(clientId, issuedAt))
                    {
                        _logger.LogWarning(
                            "Rejecting JWT for client {ClientId} — issued {IssuedAt} which predates the current revocation timestamp.",
                            clientId,
                            issuedAt);

                        context.Response.StatusCode = StatusCodes.Status401Unauthorized;
                        context.Response.Headers["WWW-Authenticate"] =
                            "Bearer error=\"invalid_token\", error_description=\"token revoked\"";
                        await context.Response.WriteAsync("{\"code\":\"token_revoked\",\"message\":\"Client credential was rotated. Re-authenticate with the current secret.\"}");
                        return;
                    }
                }
            }
        }

        await _next(context);
    }
}
