using System.Collections.Generic;
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
}
