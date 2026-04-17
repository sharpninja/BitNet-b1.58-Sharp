using System.Collections.Generic;
using Duende.IdentityServer;
using Duende.IdentityServer.Models;

namespace BitNetSharp.Distributed.Coordinator.Identity;

/// <summary>
/// OAuth/OIDC resource definitions the coordinator's Duende
/// IdentityServer configuration exposes. Only the admin Blazor UI
/// authenticates through IS — worker-to-coordinator auth uses a
/// shared <c>X-Api-Key</c> and never touches IS.
/// </summary>
public static class IdentityServerResources
{
    /// <summary>
    /// OpenID identity resources exposed to the admin Blazor UI
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
