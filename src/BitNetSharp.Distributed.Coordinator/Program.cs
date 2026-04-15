using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Claims;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator;
using BitNetSharp.Distributed.Coordinator.Components;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Identity;
using BitNetSharp.Distributed.Coordinator.Cqrs.Commands;
using BitNetSharp.Distributed.Coordinator.Middleware;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using BitNetSharp.Distributed.Coordinator.ViewModels;
using McpServer.Cqrs;
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
// Registry is seeded lazily from IConfiguration so WebApplicationFactory
// integration tests can inject in-memory WorkerClients via
// ConfigureAppConfiguration and have them picked up by the registry.
// The top-level snapshot below is ONLY used for startup-time knobs like
// the JWT authority URL; the real client list lives behind a
// CompositeClientStore that the Duende client lookup consults on every
// request.
builder.Services.AddSingleton<WorkerClientRegistry>(sp =>
{
    var configuration = sp.GetRequiredService<IConfiguration>();
    var section = configuration.GetSection($"{CoordinatorOptions.SectionName}:WorkerClients");
    var clients = section.Get<List<WorkerClientOptions>>() ?? new List<WorkerClientOptions>();
    var registry = new WorkerClientRegistry();
    registry.Seed(clients);
    return registry;
});

var coordinatorSnapshot = builder.Configuration
    .GetSection(CoordinatorOptions.SectionName)
    .Get<CoordinatorOptions>() ?? new CoordinatorOptions();
var coordinatorBaseUrl = coordinatorSnapshot.BaseUrl.TrimEnd('/');

// Duende TestUsers — seeded with the single admin account read from
// CoordinatorOptions.Admin. If the admin credentials are empty at
// startup, an empty list goes in and the login page cannot succeed;
// operators see a log warning on first /admin/api-keys hit.
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
    .AddClientStore<CompositeClientStore>()
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

// CQRS dispatcher + assembly scan so every ICommandHandler /
// IQueryHandler implemented in the coordinator assembly is
// registered automatically. ViewModels pull IDispatcher from DI
// and dispatch through the same pipeline as the minimal API
// endpoints, so there is one canonical way to exercise business
// logic no matter which UI is hosting it.
builder.Services.AddCqrsDispatcher();
builder.Services.AddCqrsHandlers(typeof(CoordinatorHostMarker).Assembly);

// ViewModels are transient so each render gets a fresh instance
// and the static-SSR lifecycle does not leak state across
// unrelated requests.
builder.Services.AddTransient<ApiKeysPageViewModel>();

// Hosted service that transitions stale workers to Gone and
// recycles timed-out task assignments back to Pending.
builder.Services.AddHostedService<StaleSweeperService>();

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
// Every handler below is a thin pass-through: pull the authenticated
// client_id claim off the JWT, build the CQRS command, dispatch it,
// map Result<T> to HTTP. No business logic in the endpoint layer.

app.MapPost("/register", async (
    [FromBody] WorkerRegistrationRequest request,
    HttpContext http,
    IDispatcher dispatcher) =>
{
    var clientId = http.User.FindFirst("client_id")?.Value ?? string.Empty;
    var result = await dispatcher
        .SendAsync<WorkerRegistrationResponse>(new RegisterWorkerCommand(clientId, request))
        .ConfigureAwait(false);

    return result.IsSuccess
        ? Results.Ok(result.Value)
        : Results.Json(
            new ErrorResponse("register_failed", result.Error ?? "unknown"),
            statusCode: StatusCodes.Status400BadRequest);
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── /work — claim the next pending task for this worker ──────────
app.MapGet("/work", async (
    HttpContext http,
    IDispatcher dispatcher) =>
{
    var clientId = http.User.FindFirst("client_id")?.Value ?? string.Empty;
    var result = await dispatcher
        .SendAsync<WorkTaskAssignment?>(new ClaimNextTaskCommand(clientId))
        .ConfigureAwait(false);

    if (!result.IsSuccess)
    {
        return Results.Json(
            new ErrorResponse("work_failed", result.Error ?? "unknown"),
            statusCode: StatusCodes.Status400BadRequest);
    }

    return result.Value is null
        ? Results.StatusCode(StatusCodes.Status204NoContent)
        : Results.Ok(result.Value);
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── /heartbeat — worker pings the coordinator periodically ───────
app.MapPost("/heartbeat", async (
    [FromBody] HeartbeatRequest request,
    HttpContext http,
    IDispatcher dispatcher) =>
{
    var clientId = http.User.FindFirst("client_id")?.Value ?? string.Empty;
    var result = await dispatcher
        .SendAsync<HeartbeatResponse>(new SubmitHeartbeatCommand(clientId, request))
        .ConfigureAwait(false);

    if (result.IsSuccess)
    {
        return Results.Ok(result.Value);
    }

    if (result.Error == SubmitHeartbeatCommandHandler.UnregisteredFailureCode)
    {
        return Results.Json(
            new ErrorResponse("unregistered", "Worker must POST /register before heartbeating."),
            statusCode: StatusCodes.Status410Gone);
    }

    return Results.Json(
        new ErrorResponse("heartbeat_failed", result.Error ?? "unknown"),
        statusCode: StatusCodes.Status400BadRequest);
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── /gradient — worker reports task completion ──────────────────
// D-1 stub: validates ownership, marks the task Done, does NOT yet
// apply the gradient to the global weights. Phase D-4 introduces the
// gradient decoder + weight updater.
app.MapPost("/gradient", async (
    [FromBody] GradientSubmission submission,
    HttpContext http,
    IDispatcher dispatcher) =>
{
    var clientId = http.User.FindFirst("client_id")?.Value ?? string.Empty;
    var result = await dispatcher
        .SendAsync<GradientAcceptance>(new SubmitGradientCommand(clientId, submission))
        .ConfigureAwait(false);

    if (result.IsSuccess)
    {
        return Results.Ok(result.Value);
    }

    if (result.Error == SubmitGradientCommandHandler.WorkerMismatchCode)
    {
        return Results.Json(
            new ErrorResponse("worker_mismatch", "Gradient workerId must match the JWT client_id."),
            statusCode: StatusCodes.Status403Forbidden);
    }

    if (result.Error == SubmitGradientCommandHandler.TaskNotAssignedCode)
    {
        return Results.Json(
            new ErrorResponse("task_not_assigned", "Task is not currently assigned to this worker."),
            statusCode: StatusCodes.Status409Conflict);
    }

    return Results.Json(
        new ErrorResponse("gradient_failed", result.Error ?? "unknown"),
        statusCode: StatusCodes.Status400BadRequest);
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
// Thin pass-through: no business logic lives here, it dispatches a
// RotateClientSecretCommand through the CQRS pipeline and maps the
// result to HTTP. Same handler is reachable from ViewModel / tests /
// scripts.
app.MapPost("/admin/rotate/{clientId}", async (
    string clientId,
    [FromQuery] string? redirect,
    IDispatcher dispatcher) =>
{
    var result = await dispatcher
        .SendAsync<RotationResult>(new RotateClientSecretCommand(clientId))
        .ConfigureAwait(false);

    if (!result.IsSuccess)
    {
        return Results.Json(
            new ErrorResponse("rotate_failed", result.Error ?? "Rotate command failed."),
            statusCode: StatusCodes.Status404NotFound);
    }

    var rotation = result.Value!;
    if (!string.IsNullOrWhiteSpace(redirect))
    {
        var separator = redirect.Contains('?') ? '&' : '?';
        var target = $"{redirect}{separator}rotated={Uri.EscapeDataString(rotation.ClientId)}";
        return Results.Redirect(target);
    }

    return Results.Ok(new
    {
        client_id  = rotation.ClientId,
        new_secret = rotation.NewSecret,
        revoked_at = rotation.RevokedAtUtc
    });
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

namespace BitNetSharp.Distributed.Coordinator
{
    /// <summary>
    /// Named marker class that exists only so the tests project can
    /// say <c>WebApplicationFactory&lt;CoordinatorHostMarker&gt;</c>
    /// without colliding with the Worker project's top-level
    /// <see cref="Program"/> class. WebApplicationFactory uses the
    /// type parameter only to locate the assembly to bootstrap; any
    /// type in the coordinator assembly works.
    /// </summary>
    public sealed class CoordinatorHostMarker;
}
