using System.Collections.Generic;
using Duende.IdentityServer;
using Duende.IdentityServer.Models;

namespace BitNetSharp.Distributed.Coordinator.Identity;

/// <summary>
/// Static OAuth 2.0 resource definitions the coordinator's Duende
/// IdentityServer configuration exposes. We only ship one API scope —
/// <c>bitnet-worker</c> — because the v1 topology has exactly one type
/// of caller.
/// </summary>
public static class IdentityServerResources
{
    /// <summary>
    /// OAuth scope name that every worker JWT must carry to hit
    /// <c>/register</c>, <c>/work</c>, <c>/heartbeat</c>,
    /// <c>/gradient</c>, and <c>/weights</c>.
    /// </summary>
    public const string WorkerScopeName = "bitnet-worker";

    /// <summary>
    /// Authorization policy name applied to protected worker
    /// endpoints via <c>[Authorize(Policy = WorkerPolicyName)]</c>.
    /// </summary>
    public const string WorkerPolicyName = "BitNetWorkerPolicy";

    /// <summary>
    /// The API scope definition Duende uses when issuing tokens.
    /// </summary>
    public static IEnumerable<ApiScope> ApiScopes => new[]
    {
        new ApiScope(WorkerScopeName, "BitNet distributed training worker API")
    };

    /// <summary>
    /// API resource wrapping the worker scope. Without an explicit
    /// resource the Duende token endpoint still works in v7 but the
    /// <c>aud</c> claim is omitted, which JwtBearer validation would
    /// then have to disable audience checks for. Keeping the resource
    /// lets us enforce a proper audience in the JWT validation layer.
    /// </summary>
    public static IEnumerable<ApiResource> ApiResources => new[]
    {
        new ApiResource(WorkerScopeName, "BitNet distributed training worker API")
        {
            Scopes = { WorkerScopeName }
        }
    };

    /// <summary>
    /// Interactive identity resources exposed to the admin Blazor UI
    /// OIDC client. Only the minimum — openid + profile — so the
    /// admin cookie just needs the sub and name claims.
    /// </summary>
    public static IEnumerable<IdentityResource> IdentityResources => new IdentityResource[]
    {
        new IdentityResources.OpenId(),
        new IdentityResources.Profile()
    };

    /// <summary>
    /// OAuth client id for the coordinator's own Blazor admin UI.
    /// The admin panel authenticates against the local IS using this
    /// client via OIDC authorization-code + PKCE.
    /// </summary>
    public const string AdminUiClientId = "bitnet-coordinator-admin-ui";

    /// <summary>
    /// Builds the Duende <see cref="Client"/> the OpenIdConnect
    /// middleware on the admin Blazor UI authenticates against. The
    /// redirect URI must match the SignInScheme's callback path the
    /// OIDC middleware advertises (the standard <c>/signin-oidc</c>).
    /// </summary>
    public static Client BuildAdminUiClient(string coordinatorBaseUrl)
    {
        var baseUrl = coordinatorBaseUrl.TrimEnd('/');
        return new Client
        {
            ClientId = AdminUiClientId,
            ClientName = "BitNet Coordinator Admin UI",
            AllowedGrantTypes = GrantTypes.Code,
            RequireClientSecret = false,
            RequirePkce = true,
            AllowedScopes =
            {
                IdentityServerConstants.StandardScopes.OpenId,
                IdentityServerConstants.StandardScopes.Profile
            },
            RedirectUris = { $"{baseUrl}/signin-oidc" },
            PostLogoutRedirectUris = { $"{baseUrl}/signout-callback-oidc" },
            FrontChannelLogoutUri = $"{baseUrl}/signout-oidc",
            AllowOfflineAccess = false,
            AllowAccessTokensViaBrowser = false
        };
    }
}
