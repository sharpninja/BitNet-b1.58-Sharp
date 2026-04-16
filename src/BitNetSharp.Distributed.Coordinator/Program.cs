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
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
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

if (args.Length > 0 && string.Equals(args[0], "generate-corpus", StringComparison.OrdinalIgnoreCase))
{
    return GenerateCorpusCommandLine(args);
}

if (args.Length > 0 && string.Equals(args[0], "tokenize-corpus", StringComparison.OrdinalIgnoreCase))
{
    return TokenizeCorpusCommandLine(args);
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
builder.Services.AddSingleton(sp =>
{
    var coord = sp.GetRequiredService<CoordinatorOptions>();
    var time  = sp.GetRequiredService<TimeProvider>();
    return new SqliteTelemetryStore(BuildConnectionString(coord), time);
});
builder.Services.AddSingleton(sp =>
{
    var coord = sp.GetRequiredService<CoordinatorOptions>();
    return new SqliteLogStore(BuildConnectionString(coord));
});
builder.Services.AddSingleton<WeightApplicationService>();

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

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

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
builder.Services.AddTransient<InstallPageViewModel>();
builder.Services.AddTransient<DashboardPageViewModel>();
builder.Services.AddTransient<LogViewerPageViewModel>();

// Hosted service that transitions stale workers to Gone and
// recycles timed-out task assignments back to Pending.
builder.Services.AddHostedService<StaleSweeperService>();

// Hourly prune service deletes old telemetry and log rows so the
// SQLite database does not grow without bound.
builder.Services.AddHostedService<TelemetryPruneService>();

var app = builder.Build();

// Ensure all stores create their schema / directories on startup.
_ = app.Services.GetRequiredService<SqliteWorkQueueStore>();
_ = app.Services.GetRequiredService<SqliteWorkerRegistryStore>();
_ = app.Services.GetRequiredService<SqliteClientRevocationStore>();
_ = app.Services.GetRequiredService<FileSystemWeightStore>();
_ = app.Services.GetRequiredService<SqliteTelemetryStore>();
_ = app.Services.GetRequiredService<SqliteLogStore>();

// Eagerly materialize the global weight vector (or load latest
// persisted version from disk) so the first /gradient request has
// a target to apply against.
app.Services.GetRequiredService<WeightApplicationService>().EnsureInitialized();

app.UseAuthentication();
app.UseAuthorization();
app.UseAntiforgery();
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

// ── /logs — structured log ingestion from workers ────────────────
// Workers batch log entries and POST them as a LogBatch. The
// coordinator stamps the authenticated client_id on every entry
// so a compromised worker cannot impersonate another's logs.
app.MapPost("/logs", (
    [FromBody] LogBatch batch,
    HttpContext http,
    SqliteLogStore logStore) =>
{
    if (batch?.Entries is null || batch.Entries.Length == 0)
    {
        return Results.Ok(new { ingested = 0 });
    }

    var clientId = http.User.FindFirst("client_id")?.Value ?? "unknown";
    var rows = new List<LogEntryRow>(batch.Entries.Length);
    foreach (var entry in batch.Entries)
    {
        rows.Add(new LogEntryRow(
            TimestampUnix: entry.Timestamp.ToUnixTimeSeconds(),
            Level: entry.Level ?? "Information",
            Category: entry.Category ?? string.Empty,
            Message: entry.Message ?? string.Empty,
            Exception: entry.Exception,
            WorkerId: clientId));
    }

    var ingested = logStore.InsertBatch(clientId, rows);
    return Results.Ok(new { ingested });
}).RequireAuthorization(IdentityServerResources.WorkerPolicyName);

// ── /corpus/{shardId} — streams a corpus shard to the worker ─────
// Workers download shard data before computing gradients against it.
// The coordinator serves shards from the corpus directory as plain
// text with optional byte-range support.
app.MapGet("/corpus/{shardId}", (
    string shardId,
    CoordinatorOptions options) =>
{
    var corpusDir = System.IO.Path.Combine(
        System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(options.DatabasePath)) ?? ".",
        "corpus");
    var shardPath = System.IO.Path.Combine(corpusDir, $"{shardId}.txt");
    if (!System.IO.File.Exists(shardPath))
    {
        return Results.Json(
            new ErrorResponse("unknown_shard", $"Corpus shard '{shardId}' not found."),
            statusCode: StatusCodes.Status404NotFound);
    }

    return Results.File(
        path: shardPath,
        contentType: "text/plain; charset=utf-8",
        fileDownloadName: $"{shardId}.txt",
        enableRangeProcessing: true);
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
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

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
    // Use the full ClaimTypes URIs so RequireRole("admin") in the
    // AdminPolicy matches. The short "role" string doesn't map to
    // ClaimTypes.Role automatically outside of JwtBearer which has
    // its own mapping table.
    var claims = new List<Claim>
    {
        new(ClaimTypes.Name, user.Username),
        new(ClaimTypes.Role, "admin"),
        new("name", user.Username),
        new("role", "admin"),
        new("sub", user.SubjectId)
    };

    // Sign in on the Duende IS cookie so /connect/authorize works
    // for any future OIDC interactions.
    var idsrvPrincipal = new ClaimsPrincipal(new ClaimsIdentity(claims, IdentityServerConstants.DefaultCookieAuthenticationScheme));
    await http.SignInAsync(IdentityServerConstants.DefaultCookieAuthenticationScheme, idsrvPrincipal);

    // ALSO sign in on the application's own Cookies scheme so the
    // admin dashboard and other [Authorize(Policy="AdminPolicy")]
    // pages work immediately without a self-referential OIDC code
    // exchange redirect chain. The login form IS the authentication
    // event — there is no security benefit to forcing the browser
    // through /connect/authorize → /signin-oidc just to set a
    // second cookie.
    var cookiesPrincipal = new ClaimsPrincipal(new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme));
    await http.SignInAsync(CookieAuthenticationDefaults.AuthenticationScheme, cookiesPrincipal);

    var redirect = string.IsNullOrWhiteSpace(returnUrl) ? "/admin/dashboard" : returnUrl;
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

// Admin dashboard: mark worker Draining or Gone. Both endpoints
// accept a POST with no body, dispatch the CQRS
// MarkWorkerStateCommand, and redirect back to the dashboard so
// the new state shows up on the next render.
app.MapPost("/admin/workers/{workerId}/drain", async (
    string workerId,
    [FromQuery] string? redirect,
    IDispatcher dispatcher) =>
{
    var result = await dispatcher
        .SendAsync<WorkerStateResult>(new MarkWorkerStateCommand(workerId, WorkerState.Draining))
        .ConfigureAwait(false);

    if (!result.IsSuccess)
    {
        return Results.Json(
            new ErrorResponse("drain_failed", result.Error ?? "unknown"),
            statusCode: StatusCodes.Status400BadRequest);
    }

    return string.IsNullOrWhiteSpace(redirect)
        ? Results.Ok(result.Value)
        : Results.Redirect(redirect);
}).RequireAuthorization("AdminPolicy").DisableAntiforgery();

app.MapPost("/admin/workers/{workerId}/gone", async (
    string workerId,
    [FromQuery] string? redirect,
    IDispatcher dispatcher) =>
{
    var result = await dispatcher
        .SendAsync<WorkerStateResult>(new MarkWorkerStateCommand(workerId, WorkerState.Gone))
        .ConfigureAwait(false);

    if (!result.IsSuccess)
    {
        return Results.Json(
            new ErrorResponse("gone_failed", result.Error ?? "unknown"),
            statusCode: StatusCodes.Status400BadRequest);
    }

    return string.IsNullOrWhiteSpace(redirect)
        ? Results.Ok(result.Value)
        : Results.Redirect(redirect);
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

// ── Admin add-client endpoint (cookie auth) ───────────────────────
// Registers a brand-new worker OAuth client at runtime, echoing the
// generated plaintext secret back through a redirect query string
// so the ApiKeys page can highlight it exactly like a rotation.
// Accepts a form-urlencoded POST from the new-client form on
// /admin/api-keys.
app.MapPost("/admin/clients", async (
    [FromForm] string clientId,
    [FromForm] string? displayName,
    IDispatcher dispatcher) =>
{
    var result = await dispatcher
        .SendAsync<AddWorkerClientResult>(new AddWorkerClientCommand(clientId, displayName))
        .ConfigureAwait(false);

    if (!result.IsSuccess)
    {
        var errorUrl = $"/admin/api-keys?error={Uri.EscapeDataString(result.Error ?? "unknown")}";
        return Results.Redirect(errorUrl);
    }

    var addedUrl = $"/admin/api-keys?added={Uri.EscapeDataString(result.Value!.ClientId)}";
    return Results.Redirect(addedUrl);
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

// ── Admin install-script download endpoints ───────────────────────
// Serve the per-client bash + PowerShell install scripts rendered by
// GetWorkerInstallScriptQuery as downloadable files. The admin
// /admin/install Razor page links operators here so they can grab a
// ready-to-run script instead of copy-pasting from the browser.
// Scripts embed the plain-text client secret, so both endpoints stay
// behind AdminPolicy.
app.MapGet("/admin/install/{clientId}.sh", async (
    string clientId,
    IDispatcher dispatcher) =>
{
    var result = await dispatcher
        .QueryAsync<InstallScriptResult>(new GetWorkerInstallScriptQuery(clientId, InstallShell.Bash))
        .ConfigureAwait(false);

    return result.IsSuccess
        ? Results.File(
            System.Text.Encoding.UTF8.GetBytes(result.Value!.Content),
            result.Value!.ContentType,
            result.Value!.Filename)
        : Results.Json(
            new ErrorResponse("install_script_failed", result.Error ?? "unknown"),
            statusCode: StatusCodes.Status404NotFound);
}).RequireAuthorization("AdminPolicy");

app.MapGet("/admin/install/{clientId}.ps1", async (
    string clientId,
    IDispatcher dispatcher) =>
{
    var result = await dispatcher
        .QueryAsync<InstallScriptResult>(new GetWorkerInstallScriptQuery(clientId, InstallShell.PowerShell))
        .ConfigureAwait(false);

    return result.IsSuccess
        ? Results.File(
            System.Text.Encoding.UTF8.GetBytes(result.Value!.Content),
            result.Value!.ContentType,
            result.Value!.Filename)
        : Results.Json(
            new ErrorResponse("install_script_failed", result.Error ?? "unknown"),
            statusCode: StatusCodes.Status404NotFound);
}).RequireAuthorization("AdminPolicy");

// ── Anonymous install-script download (gated by client secret) ─────
// The rendered script already embeds the plain-text client secret,
// so knowledge of the secret is equivalent to being allowed to
// download it. These routes let an operator fetch the script with a
// single `curl`/`iwr` on a remote worker machine without needing
// admin cookies:
//
//     curl -fsSL "https://<coord>/install/<client>.sh?k=<secret>" -o bitnet-worker.sh
//
// The secret is validated with a constant-time compare against
// WorkerClientRegistry; rotating the secret via /admin/rotate
// invalidates any previously-shared download URL.
static IResult HandleAnonymousInstallDownload(
    string clientId,
    string? k,
    InstallShell shell,
    BitNetSharp.Distributed.Coordinator.Identity.WorkerClientRegistry registry,
    IDispatcher dispatcher)
{
    if (string.IsNullOrEmpty(k))
    {
        return Results.Json(
            new ErrorResponse("missing_key", "Query parameter 'k' (client secret) is required."),
            statusCode: StatusCodes.Status401Unauthorized);
    }

    var entry = registry.Find(clientId);
    if (entry is null)
    {
        return Results.Json(
            new ErrorResponse("unknown_client", $"Unknown worker client '{clientId}'."),
            statusCode: StatusCodes.Status404NotFound);
    }

    var expected = System.Text.Encoding.UTF8.GetBytes(entry.PlainTextSecret);
    var actual   = System.Text.Encoding.UTF8.GetBytes(k);
    if (!System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(expected, actual))
    {
        return Results.Json(
            new ErrorResponse("bad_key", "Invalid client secret."),
            statusCode: StatusCodes.Status401Unauthorized);
    }

    var result = dispatcher
        .QueryAsync<InstallScriptResult>(new GetWorkerInstallScriptQuery(clientId, shell))
        .GetAwaiter()
        .GetResult();

    return result.IsSuccess
        ? Results.File(
            System.Text.Encoding.UTF8.GetBytes(result.Value!.Content),
            result.Value!.ContentType,
            result.Value!.Filename)
        : Results.Json(
            new ErrorResponse("install_script_failed", result.Error ?? "unknown"),
            statusCode: StatusCodes.Status500InternalServerError);
}

app.MapGet("/install/{clientId}.sh", (
    string clientId,
    [FromQuery] string? k,
    BitNetSharp.Distributed.Coordinator.Identity.WorkerClientRegistry registry,
    IDispatcher dispatcher) =>
    HandleAnonymousInstallDownload(clientId, k, InstallShell.Bash, registry, dispatcher))
    .AllowAnonymous();

app.MapGet("/install/{clientId}.ps1", (
    string clientId,
    [FromQuery] string? k,
    BitNetSharp.Distributed.Coordinator.Identity.WorkerClientRegistry registry,
    IDispatcher dispatcher) =>
    HandleAnonymousInstallDownload(clientId, k, InstallShell.PowerShell, registry, dispatcher))
    .AllowAnonymous();

app.Run();
return 0;

static string BuildConnectionString(CoordinatorOptions coord) =>
    $"Data Source={coord.DatabasePath};Cache=Shared";

/// <summary>
/// Generates the Truck Mate synthetic training corpus and writes
/// it as sharded text files + a manifest.json into the corpus
/// directory alongside the coordinator's database.
///
///     dotnet BitNetSharp.Distributed.Coordinator.dll generate-corpus [count]
///
/// Default count is 50,000 examples. The corpus is written to the
/// same parent directory as DatabasePath/corpus/.
/// </summary>
static int GenerateCorpusCommandLine(string[] args)
{
    try
    {
        var count = 50_000;
        if (args.Length > 1 && int.TryParse(args[1], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsedCount) && parsedCount > 0)
        {
            count = parsedCount;
        }

        var config = new ConfigurationBuilder()
            .SetBasePath(System.IO.Path.GetDirectoryName(typeof(Program).Assembly.Location) ?? ".")
            .AddJsonFile("appsettings.json", optional: true)
            .AddJsonFile($"appsettings.{Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production"}.json", optional: true)
            .AddEnvironmentVariables()
            .Build();

        var coordinator = new CoordinatorOptions();
        config.GetSection(CoordinatorOptions.SectionName).Bind(coordinator);

        var corpusDir = System.IO.Path.Combine(
            System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(coordinator.DatabasePath)) ?? ".",
            "corpus");

        Console.WriteLine($"Generating {count} Truck Mate training examples into {corpusDir}…");
        var manifest = TruckMateCorpusGenerator.Generate(corpusDir, count);

        Console.WriteLine($"Generated {manifest.TotalExamples} examples across {manifest.Shards.Count} shards.");
        foreach (var shard in manifest.Shards)
        {
            Console.WriteLine($"  {shard.ShardId}: {shard.ExampleCount} examples, {shard.SizeBytes:N0} bytes");
        }

        Console.WriteLine($"Manifest saved to {System.IO.Path.Combine(corpusDir, "manifest.json")}");
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"generate-corpus failed: {ex}");
        return 1;
    }
}

/// <summary>
/// Trains a word-level tokenizer on the text corpus shards, writes
/// the vocabulary as vocab.json, and pre-tokenizes every shard into
/// binary int32 files workers can consume directly. Invoked as:
///
///     dotnet BitNetSharp.Distributed.Coordinator.dll tokenize-corpus [maxVocab]
///
/// Default maxVocab = 8000.
/// </summary>
static int TokenizeCorpusCommandLine(string[] args)
{
    try
    {
        var maxVocab = 8000;
        if (args.Length > 1 && int.TryParse(args[1], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsedVocab) && parsedVocab > 100)
        {
            maxVocab = parsedVocab;
        }

        var config = new ConfigurationBuilder()
            .SetBasePath(System.IO.Path.GetDirectoryName(typeof(Program).Assembly.Location) ?? ".")
            .AddJsonFile("appsettings.json", optional: true)
            .AddJsonFile($"appsettings.{Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production"}.json", optional: true)
            .AddEnvironmentVariables()
            .Build();

        var coordinator = new CoordinatorOptions();
        config.GetSection(CoordinatorOptions.SectionName).Bind(coordinator);

        var corpusDir = System.IO.Path.Combine(
            System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(coordinator.DatabasePath)) ?? ".",
            "corpus");
        var tokenizedDir = System.IO.Path.Combine(corpusDir, "tokenized");
        Directory.CreateDirectory(tokenizedDir);

        // Collect all text shard files
        var shardFiles = Directory.GetFiles(corpusDir, "*.txt")
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (shardFiles.Length == 0)
        {
            Console.Error.WriteLine($"No .txt shard files found in {corpusDir}. Run generate-corpus first.");
            return 2;
        }

        Console.WriteLine($"Training tokenizer on {shardFiles.Length} shards (maxVocab={maxVocab})…");

        // Stream all lines for tokenizer training
        IEnumerable<string> AllLines()
        {
            foreach (var file in shardFiles)
            {
                foreach (var line in File.ReadLines(file))
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        yield return line;
                    }
                }
            }
        }

        var tokenizer = WordLevelTokenizer.TrainFromCorpus(AllLines(), maxVocab);
        var vocabPath = System.IO.Path.Combine(tokenizedDir, "vocab.json");
        tokenizer.SaveToFile(vocabPath);
        Console.WriteLine($"Vocabulary: {tokenizer.VocabSize} tokens → {vocabPath}");

        // Pre-tokenize each shard into a binary int32 file
        long totalTokens = 0;
        foreach (var shardFile in shardFiles)
        {
            var shardName = System.IO.Path.GetFileNameWithoutExtension(shardFile);
            var binPath = System.IO.Path.Combine(tokenizedDir, $"{shardName}.bin");

            using var output = File.Create(binPath);
            var buffer = new byte[4];
            foreach (var line in File.ReadLines(shardFile))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var ids = tokenizer.Encode(line);
                foreach (var id in ids)
                {
                    System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(buffer, id);
                    output.Write(buffer);
                    totalTokens++;
                }
            }

            var binSize = new FileInfo(binPath).Length;
            Console.WriteLine($"  {shardName}.bin: {binSize:N0} bytes ({binSize / 4:N0} tokens)");
        }

        Console.WriteLine($"Total: {totalTokens:N0} tokens across {shardFiles.Length} binary shards.");
        Console.WriteLine($"Tokenized output: {tokenizedDir}");
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"tokenize-corpus failed: {ex}");
        return 1;
    }
}

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
