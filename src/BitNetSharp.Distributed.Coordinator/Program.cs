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
using Microsoft.Extensions.Hosting.WindowsServices;
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

// ── Dev CLI subcommand: seed-tasks ──────────────────────────────
// Allows an operator to inject N pending training tasks into the
// coordinator's SQLite work queue without going through the admin
// web UI. Useful for Phase D-2 smoke-test rigs and scripting.
//
//     dotnet BitNetSharp.Distributed.Coordinator.dll seed-tasks 10
//
// Reads the database path and other config from environment
// variables / appsettings.json the same way the web host does so
// it targets whichever DB the service uses.
if (args.Length > 0 && string.Equals(args[0], "seed-tasks", StringComparison.OrdinalIgnoreCase))
{
    return SeedTasksCommandLine(args);
}

var builder = WebApplication.CreateBuilder(args);

// Running under Windows Service Control Manager sets the current
// working directory to System32, which would break relative
// appsettings.json and database path resolution. WindowsServiceHelpers
// detects the SCM case and repoints content root to the assembly
// directory so the service picks up the same config files as a
// console launch.
builder.Host.UseWindowsService(options =>
{
    options.ServiceName = "BitNetCoordinator";
});

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
builder.Services.AddTransient<TasksPageViewModel>();

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
// Admin action: bulk-enqueue a run of pending tasks. Takes the
// usual shard parameters plus a count and fans out a single
// EnqueueTasksCommand to the handler. Used to seed a training
// run from curl / scripts.
app.MapPost("/admin/tasks/enqueue", async (
    [FromBody] EnqueueTasksCommand command,
    IDispatcher dispatcher) =>
{
    var result = await dispatcher
        .SendAsync<EnqueueTasksResult>(command)
        .ConfigureAwait(false);

    return result.IsSuccess
        ? Results.Ok(result.Value)
        : Results.Json(
            new ErrorResponse("enqueue_failed", result.Error ?? "unknown"),
            statusCode: StatusCodes.Status400BadRequest);
}).RequireAuthorization("AdminPolicy").DisableAntiforgery();

// Form-post shim so the /admin/tasks Razor page's HTML form can
// submit urlencoded data and get a redirect-back instead of the
// JSON 200 response. Accepts the same parameters as the JSON
// endpoint above.
app.MapPost("/admin/tasks/enqueue-form", async (
    [FromForm] string shardId,
    [FromForm] long startOffset,
    [FromForm] long stride,
    [FromForm] long tokensPerTask,
    [FromForm] int kLocalSteps,
    [FromForm] long weightVersion,
    [FromForm] int count,
    [FromForm] string? hpJson,
    IDispatcher dispatcher) =>
{
    var command = new EnqueueTasksCommand(
        ShardId: shardId,
        ShardStartOffset: startOffset,
        ShardStride: stride,
        TokensPerTask: tokensPerTask,
        KLocalSteps: kLocalSteps,
        HyperparametersJson: hpJson ?? "{}",
        WeightVersion: weightVersion,
        Count: count);

    var result = await dispatcher
        .SendAsync<EnqueueTasksResult>(command)
        .ConfigureAwait(false);

    if (result.IsSuccess)
    {
        return Results.Redirect($"/admin/tasks?seeded={Uri.EscapeDataString(result.Value!.Inserted.ToString(System.Globalization.CultureInfo.InvariantCulture))}");
    }

    var errorUrl = $"/admin/tasks?error={Uri.EscapeDataString(result.Error ?? "unknown")}";
    return Results.Redirect(errorUrl);
}).RequireAuthorization("AdminPolicy").DisableAntiforgery();

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
return 0;

static string BuildConnectionString(CoordinatorOptions coord) =>
    $"Data Source={coord.DatabasePath};Cache=Shared";

/// <summary>
/// Seeds pending training tasks into the coordinator's SQLite work
/// queue from the CLI. Invoked when <c>args[0] == "seed-tasks"</c>
/// so it runs before the web host is constructed and exits
/// immediately after the insert loop completes.
/// </summary>
static int SeedTasksCommandLine(string[] args)
{
    try
    {
        var count = 5;
        if (args.Length > 1 && int.TryParse(args[1], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsedCount) && parsedCount > 0)
        {
            count = parsedCount;
        }

        // Mirror the web host's configuration pipeline so this
        // command targets the same database file the service uses.
        var config = new ConfigurationBuilder()
            .SetBasePath(System.IO.Path.GetDirectoryName(typeof(Program).Assembly.Location) ?? ".")
            .AddJsonFile("appsettings.json", optional: true)
            .AddJsonFile($"appsettings.{Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production"}.json", optional: true)
            .AddEnvironmentVariables()
            .Build();

        var coordinator = new CoordinatorOptions();
        config.GetSection(CoordinatorOptions.SectionName).Bind(coordinator);

        if (string.IsNullOrWhiteSpace(coordinator.DatabasePath))
        {
            Console.Error.WriteLine("Coordinator:DatabasePath is not set.");
            return 2;
        }

        using var store = new SqliteWorkQueueStore(
            $"Data Source={coordinator.DatabasePath}",
            TimeProvider.System);

        var now = DateTimeOffset.UtcNow;
        var inserted = 0;
        for (var i = 0; i < count; i++)
        {
            var taskId = $"task-seed-{Guid.NewGuid():N}";
            store.EnqueuePending(new WorkTaskRecord(
                TaskId: taskId,
                WeightVersion: coordinator.InitialWeightVersion,
                ShardId: "shard-seed",
                ShardOffset: (long)i * 8192,
                ShardLength: 8192,
                TokensPerTask: 8192,
                KLocalSteps: 4,
                HyperparametersJson: "{}",
                State: WorkTaskState.Pending,
                AssignedWorkerId: null,
                AssignedAtUtc: null,
                DeadlineUtc: null,
                Attempt: 0,
                CreatedAtUtc: now,
                CompletedAtUtc: null));
            inserted++;
        }

        var pending = store.CountByState(WorkTaskState.Pending);
        Console.WriteLine($"Seeded {inserted} tasks into {coordinator.DatabasePath}. Queue pending count: {pending}.");
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"seed-tasks failed: {ex}");
        return 1;
    }
}

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
