using System;
using System.Collections.Generic;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Auth;
using BitNetSharp.Distributed.Coordinator.Components;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Identity;
using BitNetSharp.Distributed.Coordinator.Middleware;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

// ────────────────────────────────────────────────────────────────────────
//  BitNetSharp.Distributed.Coordinator — Phase D-1 host
// ────────────────────────────────────────────────────────────────────────
//  Responsibilities wired up in this file:
//    1. Bind CoordinatorOptions from config + environment. The
//       per-worker OAuth client list and admin credentials both come
//       from environment variables so no secrets live in source.
//    2. Configure Duende IdentityServer with in-memory clients seeded
//       from WorkerClientRegistry, an ApiResource + ApiScope, and a
//       developer signing credential.
//    3. Configure Microsoft.AspNetCore.Authentication.JwtBearer to
//       validate worker tokens against the in-process IS authority.
//    4. Add a custom JwtRevocationMiddleware that rejects any JWT
//       whose `iat` claim predates the SqliteClientRevocationStore
//       entry for the client — this is the immediate-expiration hook
//       for /admin/rotate/{clientId}.
//    5. Register /health and /status.
//    6. Register /register as the worker's on-startup endpoint (JWT
//       required; persists a WorkerRecord).
//    7. Register /admin/api-keys and /admin/rotate/{clientId} behind
//       HTTP Basic auth so the operator can display and rotate the
//       worker credentials from a browser.
// ────────────────────────────────────────────────────────────────────────

var builder = WebApplication.CreateBuilder(args);

builder.Configuration
    .AddEnvironmentVariables();

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

builder.Services.AddIdentityServer(options =>
    {
        options.Events.RaiseErrorEvents = true;
        options.Events.RaiseFailureEvents = true;
        options.Events.RaiseSuccessEvents = true;
        options.EmitStaticAudienceClaim = true;
        options.LicenseKey = null; // community / dev mode — warns but works
    })
    .AddInMemoryApiScopes(IdentityServerResources.ApiScopes)
    .AddInMemoryApiResources(IdentityServerResources.ApiResources)
    .AddInMemoryClients(workerRegistry.ToDuendeClients(accessTokenLifetimeSeconds))
    .AddDeveloperSigningCredential(persistKey: false);

// ── Authentication: JWT bearer for workers + Basic for admin ──────
builder.Services
    .AddAuthentication(defaultScheme: "Bearer")
    .AddJwtBearer("Bearer", options =>
    {
        // In a single-process setup the issuer and audience are the
        // coordinator itself; JwtBearer uses the local IS authority
        // to discover the signing keys via the test-friendly
        // backchannel provided by WebApplicationFactory.
        options.Authority = "https://localhost:5001";
        options.Audience = IdentityServerResources.WorkerScopeName;
        options.RequireHttpsMetadata = false; // dev + tests over HTTP
        options.MapInboundClaims = false;     // keep original JWT claim names (iat, client_id, scope)
    })
    .AddScheme<AdminBasicAuthenticationOptions, AdminBasicAuthenticationHandler>(
        "AdminBasic",
        options => { });

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
        policy.AddAuthenticationSchemes("AdminBasic");
        policy.RequireAuthenticatedUser();
        policy.RequireRole("admin");
    });
});

// Blazor / Razor Components for the admin web UI. Static render mode
// only — the admin pages do not need interactive circuits, so there is
// no SignalR overhead and no client bundle.
builder.Services.AddRazorComponents();

var app = builder.Build();

// Ensure all three stores create their schema + directory on startup
// by resolving them once. Prevents "file not found" races on the first
// request.
_ = app.Services.GetRequiredService<SqliteWorkQueueStore>();
_ = app.Services.GetRequiredService<SqliteWorkerRegistryStore>();
_ = app.Services.GetRequiredService<SqliteClientRevocationStore>();

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
}));

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
});

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
        BearerToken: string.Empty, // JWT already carried on the request; no additional bearer issued
        InitialWeightVersion: options.InitialWeightVersion,
        RecommendedTokensPerTask: recommendedTokens,
        HeartbeatIntervalSeconds: options.HeartbeatIntervalSeconds,
        ServerTime: now);

    return Results.Ok(response);
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── Admin Blazor page (HTTP Basic auth) ───────────────────────────
app.MapRazorComponents<App>();

// ── Admin rotate endpoint (HTTP Basic auth) ───────────────────────
// Kept as a plain minimal API endpoint so it can receive POSTs from
// both the Blazor page's HTML form AND from operator scripts curling
// the coordinator directly. Supports an optional ?redirect= query so
// the web form can bounce back to /admin/api-keys?rotated={id}; a
// JSON client omits the redirect and gets a JSON body instead.
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
}).RequireAuthorization("AdminPolicy");

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
