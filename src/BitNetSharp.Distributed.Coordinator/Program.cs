using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Claims;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Components;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Identity;
using BitNetSharp.Distributed.Coordinator.Middleware;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Duende.IdentityServer;
using Duende.IdentityServer.Models;
using Duende.IdentityServer.Test;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication.OpenIdConnect;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Protocols.OpenIdConnect;

// ────────────────────────────────────────────────────────────────────────
//  BitNetSharp.Distributed.Coordinator — Phase D-1 host
// ────────────────────────────────────────────────────────────────────────
//  Single-process host serving three concerns:
//
//  1. Duende IdentityServer                                 (OAuth/OIDC provider)
//     ├─ Worker machine-login (client_credentials grant)
//     └─ Admin interactive login (authorization_code + PKCE)
//
//  2. Coordinator worker API                                (JWT bearer guarded)
//     ├─ POST /register        — worker-on-startup capability handshake
//     └─ /work /heartbeat /gradient /weights (future steps)
//
//  3. Blazor admin UI                                       (cookie + OIDC guarded)
//     ├─ GET  /admin/api-keys  — list + rotate worker secrets
//     ├─ GET  /Account/Login   — login form presented by IS
//     └─ POST /Account/Login/submit  — credential validator
//
//  Auth schemes stacked in this file:
//      "Cookies"  — admin session cookie set after OIDC code exchange
//      "oidc"     — OpenIdConnect challenge pointing at local IS
//      "Bearer"   — JWT validator for worker endpoints
//      "idsrv"    — Duende's own default cookie scheme for the IS UI
// ────────────────────────────────────────────────────────────────────────

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddEnvironmentVariables();

builder.Services.Configure<CoordinatorOptions>(
    builder.Configuration.GetSection(CoordinatorOptions.SectionName));
builder.Services.AddSingleton(sp =>
    sp.GetRequiredService<IOptions<CoordinatorOptions>>().Value);

// ── Persistence stores ────────────────────────────────────────────
builder.Services.AddSingleton<TimeProvider>(TimeProvider.System);

builder.Services.AddSingleton(sp =>
{
    var coord = sp.GetRequiredService<CoordinatorOptions>();
    var time  = sp.GetRequiredService<TimeProvider>();
    return new SqliteWorkQueueStore(BuildConnectionString(coord), time);
});
builder.Services.AddSingleton(sp =>
{
    var coord = sp.GetRequiredService<CoordinatorOptions>();
    var time  = sp.GetRequiredService<TimeProvider>();
    return new SqliteWorkerRegistryStore(BuildConnectionString(coord), time);
});
builder.Services.AddSingleton(sp =>
{
    var coord = sp.GetRequiredService<CoordinatorOptions>();
    var time  = sp.GetRequiredService<TimeProvider>();
    return new SqliteClientRevocationStore(BuildConnectionString(coord), time);
});
builder.Services.AddSingleton(sp =>
{
    var coord = sp.GetRequiredService<CoordinatorOptions>();
    var weightsDir = Path.Combine(
        Path.GetDirectoryName(Path.GetFullPath(coord.DatabasePath)) ?? ".",
        "weights");
    return new FileSystemWeightStore(weightsDir);
});

// ── Worker client registry + Duende IdentityServer ────────────────
var workerRegistry = new WorkerClientRegistry();
workerRegistry.Seed(builder.Configuration
    .GetSection($"{CoordinatorOptions.SectionName}:WorkerClients")
    .Get<List<WorkerClientOptions>>() ?? new List<WorkerClientOptions>());
builder.Services.AddSingleton(workerRegistry);

var coordinatorSnapshot = builder.Configuration
    .GetSection(CoordinatorOptions.SectionName)
    .Get<CoordinatorOptions>() ?? new CoordinatorOptions();
var accessTokenLifetimeSeconds = coordinatorSnapshot.AccessTokenLifetimeSeconds;
var coordinatorBaseUrl = coordinatorSnapshot.BaseUrl.TrimEnd('/');

// Duende TestUsers — seeded with the single admin account read from
// CoordinatorOptions.Admin. The cookie-based IS login flow validates
// credentials against this list via the default ResourceOwnerPasswordValidator.
var adminTestUsers = new List<TestUser>();
if (!string.IsNullOrWhiteSpace(coordinatorSnapshot.Admin.Username) &&
    !string.IsNullOrWhiteSpace(coordinatorSnapshot.Admin.Password))
{
    adminTestUsers.Add(new TestUser
    {
        SubjectId = "admin",
        Username = coordinatorSnapshot.Admin.Username,
        Password = coordinatorSnapshot.Admin.Password,
        Claims =
        {
            new Claim("name", coordinatorSnapshot.Admin.Username),
            new Claim("role", "admin")
        }
    });
}

// Merge worker clients + admin UI client into the Duende client list.
var duendeClients = new List<Client>();
duendeClients.AddRange(workerRegistry.ToDuendeClients(accessTokenLifetimeSeconds));
duendeClients.Add(IdentityServerResources.BuildAdminUiClient(coordinatorBaseUrl));

builder.Services.AddIdentityServer(options =>
    {
        options.Events.RaiseErrorEvents = true;
        options.Events.RaiseFailureEvents = true;
        options.Events.RaiseSuccessEvents = true;
        options.EmitStaticAudienceClaim = true;
        options.LicenseKey = null; // community / dev mode

        // Interactive login page the IS redirects to when an
        // unauthenticated user hits /connect/authorize.
        options.UserInteraction.LoginUrl = "/Account/Login";
        options.UserInteraction.LoginReturnUrlParameter = "returnUrl";
    })
    .AddInMemoryIdentityResources(IdentityServerResources.IdentityResources)
    .AddInMemoryApiScopes(IdentityServerResources.ApiScopes)
    .AddInMemoryApiResources(IdentityServerResources.ApiResources)
    .AddInMemoryClients(duendeClients)
    .AddTestUsers(adminTestUsers)
    .AddDeveloperSigningCredential(persistKey: false);

// ── Authentication schemes ────────────────────────────────────────
builder.Services
    .AddAuthentication(options =>
    {
        // The default scheme used by admin Blazor pages is the cookie
        // the OIDC middleware drops after a successful code exchange.
        // Worker endpoints opt into JWT validation explicitly via their
        // authorization policy so the two planes do not interfere.
        options.DefaultScheme = CookieAuthenticationDefaults.AuthenticationScheme;
        options.DefaultChallengeScheme = OpenIdConnectDefaults.AuthenticationScheme;
    })
    .AddCookie(CookieAuthenticationDefaults.AuthenticationScheme, options =>
    {
        options.Cookie.Name = "bitnet-coord-admin";
        options.Cookie.SameSite = SameSiteMode.Lax;
        options.LoginPath = "/Account/Login";
        options.AccessDeniedPath = "/Account/Login";
    })
    .AddOpenIdConnect(OpenIdConnectDefaults.AuthenticationScheme, options =>
    {
        options.Authority = coordinatorBaseUrl;
        options.ClientId = IdentityServerResources.AdminUiClientId;
        options.ResponseType = OpenIdConnectResponseType.Code;
        options.UsePkce = true;
        options.RequireHttpsMetadata = false; // localhost dev / ngrok TLS
        options.SaveTokens = true;
        options.GetClaimsFromUserInfoEndpoint = true;
        options.Scope.Clear();
        options.Scope.Add("openid");
        options.Scope.Add("profile");
        options.TokenValidationParameters.NameClaimType = "name";
        options.TokenValidationParameters.RoleClaimType = "role";
    })
    .AddJwtBearer("Bearer", options =>
    {
        options.Authority = coordinatorBaseUrl;
        options.Audience = IdentityServerResources.WorkerScopeName;
        options.RequireHttpsMetadata = false;
        options.MapInboundClaims = false;
    });

builder.Services.AddAuthorization(options =>
{
    options.AddPolicy(IdentityServerResources.WorkerPolicyName, policy =>
    {
        policy.AddAuthenticationSchemes("Bearer");
        policy.RequireAuthenticatedUser();
        policy.RequireClaim("scope", IdentityServerResources.WorkerScopeName);
    });

    options.AddPolicy("AdminPolicy", policy =>
    {
        policy.AddAuthenticationSchemes(CookieAuthenticationDefaults.AuthenticationScheme);
        policy.RequireAuthenticatedUser();
        policy.RequireRole("admin");
    });
});

builder.Services.AddRazorComponents();

var app = builder.Build();

// Ensure all stores create their schema / directories on startup.
_ = app.Services.GetRequiredService<SqliteWorkQueueStore>();
_ = app.Services.GetRequiredService<SqliteWorkerRegistryStore>();
_ = app.Services.GetRequiredService<SqliteClientRevocationStore>();
_ = app.Services.GetRequiredService<FileSystemWeightStore>();

app.UseAuthentication();
app.UseAuthorization();
app.UseMiddleware<JwtRevocationMiddleware>();
app.UseIdentityServer();

// ── Unauthenticated endpoints ─────────────────────────────────────
app.MapGet("/health", () => Results.Ok(new
{
    status = "ok",
    time   = DateTimeOffset.UtcNow,
    phase  = "D-1"
})).AllowAnonymous();

app.MapGet("/status", (
    SqliteWorkQueueStore workQueue,
    SqliteWorkerRegistryStore workerRegistryStore,
    WorkerClientRegistry clientRegistry) =>
{
    return Results.Ok(new
    {
        tasks = new
        {
            pending  = workQueue.CountByState(WorkTaskState.Pending),
            assigned = workQueue.CountByState(WorkTaskState.Assigned),
            done     = workQueue.CountByState(WorkTaskState.Done),
            failed   = workQueue.CountByState(WorkTaskState.Failed)
        },
        workers = new
        {
            configured = clientRegistry.Count,
            active     = workerRegistryStore.CountByState(WorkerState.Active),
            draining   = workerRegistryStore.CountByState(WorkerState.Draining),
            gone       = workerRegistryStore.CountByState(WorkerState.Gone)
        },
        time = DateTimeOffset.UtcNow
    });
}).AllowAnonymous();

// ── Worker endpoints (JWT-protected) ──────────────────────────────
app.MapPost("/register", (
    [FromBody] WorkerRegistrationRequest request,
    HttpContext http,
    SqliteWorkerRegistryStore workerStore,
    CoordinatorOptions options,
    TimeProvider time) =>
{
    if (request is null)
    {
        return Results.Json(
            new ErrorResponse("invalid_request", "Request body is missing."),
            statusCode: StatusCodes.Status400BadRequest);
    }

    var clientId = http.User.FindFirst("client_id")?.Value;
    if (string.IsNullOrWhiteSpace(clientId))
    {
        return Results.Json(
            new ErrorResponse("unknown_client", "JWT did not carry a client_id claim."),
            statusCode: StatusCodes.Status401Unauthorized);
    }

    var recommendedTokens = TaskSizingCalculator.RecommendedTokensPerTask(
        request.Capability.TokensPerSecond,
        TimeSpan.FromSeconds(options.TargetTaskDurationSeconds),
        options.FullStepEfficiency);

    var now = time.GetUtcNow();
    workerStore.Upsert(new WorkerRecord(
        WorkerId: clientId,
        Name: string.IsNullOrWhiteSpace(request.WorkerName) ? clientId : request.WorkerName,
        CpuThreads: request.Capability.CpuThreads,
        TokensPerSecond: request.Capability.TokensPerSecond,
        RecommendedTokensPerTask: recommendedTokens,
        ProcessArchitecture: request.ProcessArchitecture,
        OsDescription: request.OsDescription,
        RegisteredAtUtc: now,
        LastHeartbeatUtc: now,
        State: WorkerState.Active));

    var response = new WorkerRegistrationResponse(
        WorkerId: clientId,
        BearerToken: string.Empty,
        InitialWeightVersion: options.InitialWeightVersion,
        RecommendedTokensPerTask: recommendedTokens,
        HeartbeatIntervalSeconds: options.HeartbeatIntervalSeconds,
        ServerTime: now);

    return Results.Ok(response);
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── /work — claim the next pending task for this worker ──────────
app.MapGet("/work", (
    HttpContext http,
    SqliteWorkQueueStore workQueue,
    CoordinatorOptions options) =>
{
    var clientId = http.User.FindFirst("client_id")?.Value;
    if (string.IsNullOrWhiteSpace(clientId))
    {
        return Results.Json(
            new ErrorResponse("unknown_client", "JWT did not carry a client_id claim."),
            statusCode: StatusCodes.Status401Unauthorized);
    }

    var leaseDuration = TimeSpan.FromSeconds(options.TargetTaskDurationSeconds * 2);
    var claimed = workQueue.TryClaimNextPending(clientId, leaseDuration);
    if (claimed is null)
    {
        return Results.StatusCode(StatusCodes.Status204NoContent);
    }

    var baseUrl = options.BaseUrl.TrimEnd('/');
    var assignment = new WorkTaskAssignment(
        TaskId: claimed.TaskId,
        WeightVersion: claimed.WeightVersion,
        WeightUrl: $"{baseUrl}/weights/{claimed.WeightVersion}",
        ShardId: claimed.ShardId,
        ShardOffset: claimed.ShardOffset,
        ShardLength: claimed.ShardLength,
        TokensPerTask: claimed.TokensPerTask,
        KLocalSteps: claimed.KLocalSteps,
        HyperparametersJson: claimed.HyperparametersJson,
        DeadlineUtc: claimed.DeadlineUtc ?? DateTimeOffset.UtcNow.Add(leaseDuration));

    return Results.Ok(assignment);
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── /heartbeat — worker pings the coordinator periodically ───────
app.MapPost("/heartbeat", (
    [FromBody] HeartbeatRequest request,
    HttpContext http,
    SqliteWorkerRegistryStore workerStore,
    CoordinatorOptions options,
    TimeProvider time) =>
{
    if (request is null)
    {
        return Results.Json(
            new ErrorResponse("invalid_request", "Heartbeat body is missing."),
            statusCode: StatusCodes.Status400BadRequest);
    }

    var clientId = http.User.FindFirst("client_id")?.Value;
    if (string.IsNullOrWhiteSpace(clientId))
    {
        return Results.Json(
            new ErrorResponse("unknown_client", "JWT did not carry a client_id claim."),
            statusCode: StatusCodes.Status401Unauthorized);
    }

    var touched = workerStore.TouchHeartbeat(clientId);
    if (!touched)
    {
        return Results.Json(
            new ErrorResponse("unregistered", "Worker must POST /register before heartbeating."),
            statusCode: StatusCodes.Status410Gone);
    }

    return Results.Ok(new HeartbeatResponse(
        ShouldDrain: false,
        RecommendedTokensPerTaskOverride: null,
        ServerTime: time.GetUtcNow()));
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── /gradient — worker reports task completion ──────────────────
// D-1 stub: validates ownership, marks the task Done, does NOT yet
// apply the gradient to the global weights. Phase D-4 introduces the
// gradient decoder + weight updater.
app.MapPost("/gradient", (
    [FromBody] GradientSubmission submission,
    HttpContext http,
    SqliteWorkQueueStore workQueue,
    ILogger<Program> logger) =>
{
    if (submission is null)
    {
        return Results.Json(
            new ErrorResponse("invalid_request", "Gradient body is missing."),
            statusCode: StatusCodes.Status400BadRequest);
    }

    var clientId = http.User.FindFirst("client_id")?.Value;
    if (string.IsNullOrWhiteSpace(clientId) || clientId != submission.WorkerId)
    {
        return Results.Json(
            new ErrorResponse("worker_mismatch", "Gradient workerId must match the JWT client_id."),
            statusCode: StatusCodes.Status403Forbidden);
    }

    var completed = workQueue.MarkCompleted(submission.TaskId, clientId);
    if (!completed)
    {
        return Results.Json(
            new ErrorResponse("task_not_assigned", "Task is not currently assigned to this worker."),
            statusCode: StatusCodes.Status409Conflict);
    }

    logger.LogInformation(
        "Accepted gradient for task {TaskId} from worker {ClientId}: format={Format}, bytes={Size}, tokens={Tokens}, loss={Loss}, staleness={Staleness}",
        submission.TaskId,
        clientId,
        submission.GradientFormat,
        submission.GradientPayload?.Length ?? 0,
        submission.TokensSeen,
        submission.LossAfter,
        0);

    return Results.Ok(new
    {
        accepted = true,
        task_id = submission.TaskId,
        worker_id = clientId
    });
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── /weights/{version} — streams a weight blob to the worker ─────
app.MapGet("/weights/{version:long}", (
    long version,
    FileSystemWeightStore weights) =>
{
    var manifest = weights.TryGetManifest(version);
    if (manifest is null)
    {
        return Results.Json(
            new ErrorResponse("unknown_version", $"Weight version {version} is not available."),
            statusCode: StatusCodes.Status404NotFound);
    }

    var stream = weights.TryOpenReadStream(version);
    if (stream is null)
    {
        return Results.Json(
            new ErrorResponse("unknown_version", $"Weight version {version} is not available."),
            statusCode: StatusCodes.Status404NotFound);
    }

    return Results.File(
        fileStream: stream,
        contentType: "application/octet-stream",
        fileDownloadName: $"bitnet-weights-v{version}.bin",
        enableRangeProcessing: true);
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── Admin Blazor UI (cookie + OIDC) ───────────────────────────────
app.MapRazorComponents<App>();

// ── Account/Login POST handler ────────────────────────────────────
// Receives the login form submission from Components/Pages/LoginPage.razor,
// validates the credentials against Duende's TestUserStore, signs in
// on the "idsrv" cookie so the in-progress /connect/authorize
// request can resume, and redirects the browser back to the caller's
// returnUrl — which is the Duende authorize continuation URL.
app.MapPost("/Account/Login/submit", async (
    [FromForm] string username,
    [FromForm] string password,
    [FromForm] string? returnUrl,
    HttpContext http,
    TestUserStore users) =>
{
    if (!users.ValidateCredentials(username, password))
    {
        var safeReturn = string.IsNullOrWhiteSpace(returnUrl) ? "/admin/api-keys" : returnUrl;
        return Results.Redirect(
            $"/Account/Login?error={Uri.EscapeDataString("Invalid credentials")}&returnUrl={Uri.EscapeDataString(safeReturn)}");
    }

    var user = users.FindByUsername(username);
    var claims = new List<Claim>
    {
        new(ClaimTypes.Name, user.Username),
        new("name", user.Username),
        new("role", "admin"),
        new("sub", user.SubjectId)
    };
    var principal = new ClaimsPrincipal(new ClaimsIdentity(claims, IdentityServerConstants.DefaultCookieAuthenticationScheme));
    await http.SignInAsync(IdentityServerConstants.DefaultCookieAuthenticationScheme, principal);

    var redirect = string.IsNullOrWhiteSpace(returnUrl) ? "/admin/api-keys" : returnUrl;
    return Results.Redirect(redirect);
}).DisableAntiforgery();

// ── Admin rotate endpoint (cookie auth) ───────────────────────────
app.MapPost("/admin/rotate/{clientId}", (
    string clientId,
    [FromQuery] string? redirect,
    WorkerClientRegistry registry,
    SqliteClientRevocationStore revocations,
    ILogger<Program> logger) =>
{
    try
    {
        var freshSecret = registry.Rotate(clientId);
        var revokedAt   = revocations.Revoke(clientId);
        logger.LogWarning(
            "Admin rotated client secret for {ClientId} at {RevokedAt}. Existing JWTs for this client are now invalid.",
            clientId,
            revokedAt);

        if (!string.IsNullOrWhiteSpace(redirect))
        {
            var separator = redirect.Contains('?') ? '&' : '?';
            var target = $"{redirect}{separator}rotated={Uri.EscapeDataString(clientId)}";
            return Results.Redirect(target);
        }

        return Results.Ok(new
        {
            client_id  = clientId,
            new_secret = freshSecret,
            revoked_at = revokedAt
        });
    }
    catch (KeyNotFoundException)
    {
        return Results.Json(
            new ErrorResponse("unknown_client", $"Client '{clientId}' is not registered."),
            statusCode: StatusCodes.Status404NotFound);
    }
}).RequireAuthorization("AdminPolicy").DisableAntiforgery();

app.Run();

static string BuildConnectionString(CoordinatorOptions coord) =>
    $"Data Source={coord.DatabasePath};Cache=Shared";

/// <summary>
/// Empty partial so WebApplicationFactory-based integration tests in
/// <c>BitNetSharp.Tests</c> can bootstrap the coordinator without
/// spawning a separate process. Required by the test SDK to
/// discriminate the assembly entry point.
/// </summary>
public partial class Program;
