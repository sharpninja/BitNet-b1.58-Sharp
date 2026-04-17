using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Claims;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator;
using BitNetSharp.Distributed.Coordinator.Authentication;
using BitNetSharp.Distributed.Coordinator.Components;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Identity;
using BitNetSharp.Distributed.Coordinator.Cqrs.Commands;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using BitNetSharp.Distributed.Coordinator.ViewModels;
using McpServer.Cqrs;
using Duende.IdentityServer;
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
//  1. Duende IdentityServer                                 (OIDC provider for admin UI only)
//     └─ Admin interactive login (authorization_code + PKCE)
//
//  2. Coordinator worker API                                (X-Api-Key guarded)
//     ├─ POST /register        — worker-on-startup capability handshake
//     └─ /work /heartbeat /gradient /weights /logs /corpus
//
//  3. Blazor admin UI                                       (cookie + OIDC guarded)
//     ├─ GET  /admin/dashboard — worker/task overview
//     ├─ GET  /Account/Login   — login form presented by IS
//     └─ POST /Account/Login/submit  — credential validator
//
//  Auth schemes stacked in this file:
//      "Cookies"  — admin session cookie set after login
//      "oidc"     — OpenIdConnect challenge pointing at local IS
//      "ApiKey"   — shared X-Api-Key validator for worker endpoints
//      "idsrv"    — Duende's own default cookie scheme for the IS UI
//
//  Worker auth model: single shared API key set by the operator via
//  Coordinator__WorkerApiKey. Every worker sends the same key in the
//  X-Api-Key header and its self-declared id in X-Worker-Id. Rotate
//  the key = edit env var + restart coordinator => every worker is
//  instantly disabled until redeployed with the new key.
// ────────────────────────────────────────────────────────────────────────

// ── Dev CLI subcommand: seed-tasks ──────────────────────────────
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

if (args.Length > 0 && string.Equals(args[0], "seed-real-tasks", StringComparison.OrdinalIgnoreCase))
{
    return SeedRealTasksCommandLine(args);
}

if (args.Length > 0 && string.Equals(args[0], "dump-events", StringComparison.OrdinalIgnoreCase))
{
    return DumpEventsCommandLine(args);
}

if (args.Length > 0 && string.Equals(args[0], "purge-pending", StringComparison.OrdinalIgnoreCase))
{
    return PurgePendingCommandLine(args);
}

if (args.Length > 0 && string.Equals(args[0], "dump-telemetry", StringComparison.OrdinalIgnoreCase))
{
    return DumpTelemetryCommandLine(args);
}

if (args.Length > 0 && string.Equals(args[0], "purge-telemetry", StringComparison.OrdinalIgnoreCase))
{
    return PurgeTelemetryCommandLine(args);
}

if (args.Length > 0 && string.Equals(args[0], "purge-shards", StringComparison.OrdinalIgnoreCase))
{
    return PurgeShardsCommandLine(args);
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
builder.Services.AddSingleton<ICoordinatorModelConfig, CoordinatorModelConfig>();
builder.Services.AddSingleton<WeightApplicationService>();

// ── Duende IdentityServer (admin OIDC only) ───────────────────────
var coordinatorSnapshot = builder.Configuration
    .GetSection(CoordinatorOptions.SectionName)
    .Get<CoordinatorOptions>() ?? new CoordinatorOptions();
var coordinatorBaseUrl = coordinatorSnapshot.BaseUrl.TrimEnd('/');

// Duende TestUsers — seeded with the single admin account read from
// CoordinatorOptions.Admin. If the admin credentials are empty at
// startup, an empty list goes in and the login page cannot succeed;
// operators see a log warning on first /admin/dashboard hit.
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
    .AddInMemoryClients(new[] { IdentityServerResources.BuildAdminUiClient(coordinatorBaseUrl) })
    .AddTestUsers(adminTestUsers)
    .AddDeveloperSigningCredential(persistKey: false);

// ── Authentication schemes ────────────────────────────────────────
builder.Services
    .AddAuthentication(options =>
    {
        // The default scheme used by admin Blazor pages is the cookie
        // the login form drops after a successful credential check.
        // Worker endpoints opt into the X-Api-Key scheme explicitly via
        // their authorization policy so the two planes do not interfere.
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
    .AddScheme<ApiKeyAuthenticationOptions, ApiKeyAuthenticationHandler>(
        ApiKeyAuthenticationHandler.SchemeName,
        options =>
        {
            // Resolve the configured key lazily on each request so the
            // IOptionsMonitor picks up env-var changes without a full
            // restart on hosts that support config reload.
            options.ExpectedKey = null; // wired up post-build from DI.
        });

// Post-configure the ApiKey scheme with the IOptionsMonitor<CoordinatorOptions>
// so the handler can look up the current WorkerApiKey on every request.
builder.Services.AddSingleton<IPostConfigureOptions<ApiKeyAuthenticationOptions>>(sp =>
    new ApiKeyPostConfigure(sp.GetRequiredService<IOptionsMonitor<CoordinatorOptions>>()));

builder.Services.AddAuthorization(options =>
{
    options.AddPolicy(WorkerApiKeyAuth.PolicyName, policy =>
    {
        policy.AddAuthenticationSchemes(ApiKeyAuthenticationHandler.SchemeName);
        policy.RequireAuthenticatedUser();
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

// CQRS dispatcher + assembly scan.
builder.Services.AddCqrsDispatcher();
builder.Services.AddCqrsHandlers(typeof(CoordinatorHostMarker).Assembly);

// ViewModels are transient so each render gets a fresh instance
// and the static-SSR lifecycle does not leak state across
// unrelated requests.
builder.Services.AddTransient<TasksPageViewModel>();
builder.Services.AddTransient<DashboardPageViewModel>();
builder.Services.AddTransient<LogViewerPageViewModel>();
builder.Services.AddTransient<TaskBrowserPageViewModel>();

// Hosted service that transitions stale workers to Gone and
// recycles timed-out task assignments back to Pending.
builder.Services.AddHostedService<StaleSweeperService>();

// Hourly prune service deletes old telemetry and log rows so the
// SQLite database does not grow without bound.
builder.Services.AddHostedService<TelemetryPruneService>();

var app = builder.Build();

// Warn at startup if the operator forgot to set the shared worker API
// key — without it the worker endpoints will reject every request.
var startupLogger = app.Services.GetRequiredService<ILoggerFactory>()
    .CreateLogger("BitNetSharp.Distributed.Coordinator.Startup");
if (string.IsNullOrWhiteSpace(app.Services.GetRequiredService<CoordinatorOptions>().WorkerApiKey))
{
    startupLogger.LogWarning(
        "Coordinator:WorkerApiKey is not configured. Set the Coordinator__WorkerApiKey environment variable on the coordinator host and restart; every worker must present that same key in the X-Api-Key header.");
}

// Ensure all stores create their schema / directories on startup.
_ = app.Services.GetRequiredService<SqliteWorkQueueStore>();
_ = app.Services.GetRequiredService<SqliteWorkerRegistryStore>();
_ = app.Services.GetRequiredService<FileSystemWeightStore>();
_ = app.Services.GetRequiredService<SqliteTelemetryStore>();
_ = app.Services.GetRequiredService<SqliteLogStore>();

// Resolve + log the chosen model preset once at startup so the
// operator sees the canonical flat-vector shape every worker is
// expected to agree on. This is the Track 7 "model config banner".
var modelConfig = app.Services.GetRequiredService<ICoordinatorModelConfig>();
startupLogger.LogInformation(
    "Coordinator model configuration: {Display}. Weight vector bytes on wire = {Bytes:N0} (header {Header} + {Elements:N0} x 4).",
    modelConfig.ToDisplayString(),
    WeightBlobCodec.HeaderSize + 4L * modelConfig.FlatLength,
    WeightBlobCodec.HeaderSize,
    modelConfig.FlatLength);

// Eagerly materialize the global weight vector (or load latest
// persisted version from disk) so the first /gradient request has
// a target to apply against.
app.Services.GetRequiredService<WeightApplicationService>().EnsureInitialized();

app.UseAuthentication();
app.UseAuthorization();
app.UseAntiforgery();
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
    SqliteWorkerRegistryStore workerRegistryStore) =>
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
            active     = workerRegistryStore.CountByState(WorkerState.Active),
            draining   = workerRegistryStore.CountByState(WorkerState.Draining),
            gone       = workerRegistryStore.CountByState(WorkerState.Gone)
        },
        time = DateTimeOffset.UtcNow
    });
}).AllowAnonymous();

// ── Worker endpoints (X-Api-Key-protected) ────────────────────────
// Every handler below is a thin pass-through: pull the calling
// worker's self-declared id off the X-Worker-Id header (surfaced as
// the "client_id" claim by the auth handler), build the CQRS
// command, dispatch it, map Result<T> to HTTP.

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
}).RequireAuthorization(WorkerApiKeyAuth.PolicyName);

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
}).RequireAuthorization(WorkerApiKeyAuth.PolicyName);

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
}).RequireAuthorization(WorkerApiKeyAuth.PolicyName);

// ── /gradient — worker reports task completion ──────────────────
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
            new ErrorResponse("worker_mismatch", "Gradient workerId must match the X-Worker-Id header."),
            statusCode: StatusCodes.Status403Forbidden);
    }

    if (result.Error == SubmitGradientCommandHandler.TaskNotAssignedCode)
    {
        return Results.Json(
            new ErrorResponse("task_not_assigned", "Task is not currently assigned to this worker."),
            statusCode: StatusCodes.Status409Conflict);
    }

    // Track 7: surface gradient shape mismatches with an
    // explicit length_mismatch body so workers can log a precise
    // "expected N, got M" diagnostic without having to regex the
    // free-form gradient_failed message.
    if (result.Error is not null
        && result.Error.StartsWith(SubmitGradientCommandHandler.GradientShapeCode, StringComparison.Ordinal))
    {
        return Results.Json(
            new ErrorResponse("length_mismatch", result.Error),
            statusCode: StatusCodes.Status400BadRequest);
    }

    return Results.Json(
        new ErrorResponse("gradient_failed", result.Error ?? "unknown"),
        statusCode: StatusCodes.Status400BadRequest);
}).RequireAuthorization(WorkerApiKeyAuth.PolicyName);

// ── /logs — structured log ingestion from workers ────────────────
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
}).RequireAuthorization(WorkerApiKeyAuth.PolicyName);

// ── /corpus/{shardId} — streams a corpus shard to the worker ─────
// Resolution order: .bin (tokenized int32 stream, preferred — matches
// what CorpusClient on the worker expects) → .txt (legacy synthetic) →
// bare id (already has extension). Content-Type flips accordingly so
// Range requests on the tokenized path stay byte-exact.
app.MapGet("/corpus/{shardId}", (
    string shardId,
    CoordinatorOptions options) =>
{
    var shardPath = BitNetSharp.Distributed.Coordinator.Services.CorpusShardLocator
        .TryResolve(options.DatabasePath, shardId);
    if (shardPath is null)
    {
        return Results.Json(
            new ErrorResponse("unknown_shard", $"Corpus shard '{shardId}' not found."),
            statusCode: StatusCodes.Status404NotFound);
    }

    var ext = System.IO.Path.GetExtension(shardPath);
    var (contentType, downloadName) = ext.Equals(".bin", System.StringComparison.OrdinalIgnoreCase)
        ? ("application/octet-stream", shardId + ".bin")
        : ("text/plain; charset=utf-8",  shardId + ".txt");

    return Results.File(
        path: shardPath,
        contentType: contentType,
        fileDownloadName: downloadName,
        enableRangeProcessing: true);
}).RequireAuthorization(WorkerApiKeyAuth.PolicyName);

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
}).RequireAuthorization(WorkerApiKeyAuth.PolicyName);

// ── Admin Blazor UI (cookie + OIDC) ───────────────────────────────
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

// ── Account/Login POST handler ────────────────────────────────────
app.MapPost("/Account/Login/submit", async (
    [FromForm] string username,
    [FromForm] string password,
    [FromForm] string? returnUrl,
    HttpContext http,
    TestUserStore users) =>
{
    if (!users.ValidateCredentials(username, password))
    {
        var safeReturn = string.IsNullOrWhiteSpace(returnUrl) ? "/admin/dashboard" : returnUrl;
        return Results.Redirect(
            $"/Account/Login?error={Uri.EscapeDataString("Invalid credentials")}&returnUrl={Uri.EscapeDataString(safeReturn)}");
    }

    var user = users.FindByUsername(username);
    var claims = new List<Claim>
    {
        new(ClaimTypes.Name, user.Username),
        new(ClaimTypes.Role, "admin"),
        new("name", user.Username),
        new("role", "admin"),
        new("sub", user.SubjectId)
    };

    var idsrvPrincipal = new ClaimsPrincipal(new ClaimsIdentity(claims, IdentityServerConstants.DefaultCookieAuthenticationScheme));
    await http.SignInAsync(IdentityServerConstants.DefaultCookieAuthenticationScheme, idsrvPrincipal);

    var cookiesPrincipal = new ClaimsPrincipal(new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme));
    await http.SignInAsync(CookieAuthenticationDefaults.AuthenticationScheme, cookiesPrincipal);

    var redirect = string.IsNullOrWhiteSpace(returnUrl) ? "/admin/dashboard" : returnUrl;
    return Results.Redirect(redirect);
}).DisableAntiforgery();

// Admin action: bulk-enqueue a run of pending tasks.
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

// Admin dashboard: mark worker Draining or Gone.
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
// JSON 200 response.
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

app.Run();
return 0;

static string BuildConnectionString(CoordinatorOptions coord) =>
    $"Data Source={coord.DatabasePath};Cache=Shared";

static int GenerateCorpusCommandLine(string[] args)
{
    try
    {
        var count = 50_000;
        var seed = 42;
        var poolVersion = CorpusPoolVersion.V1;
        var manifestName = "truckmate-v1";
        var examplesPerShard = 5_000;

        if (args.Length > 1 && int.TryParse(args[1], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsedCount) && parsedCount > 0)
        {
            count = parsedCount;
        }

        for (var i = 2; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--seed" when i + 1 < args.Length:
                    if (int.TryParse(args[++i], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsedSeed))
                    {
                        seed = parsedSeed;
                    }
                    break;
                case "--pool" when i + 1 < args.Length:
                    poolVersion = args[++i].Equals("v2", StringComparison.OrdinalIgnoreCase)
                        ? CorpusPoolVersion.V2
                        : CorpusPoolVersion.V1;
                    break;
                case "--name" when i + 1 < args.Length:
                    manifestName = args[++i];
                    break;
                case "--examples-per-shard" when i + 1 < args.Length:
                    if (int.TryParse(args[++i], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsedEps) && parsedEps > 0)
                    {
                        examplesPerShard = parsedEps;
                    }
                    break;
            }
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

        Console.WriteLine($"Generating {count} Truck Mate training examples into {corpusDir} (seed={seed}, pool={poolVersion}, name={manifestName})…");
        var manifest = TruckMateCorpusGenerator.Generate(corpusDir, count, examplesPerShard, seed, poolVersion, manifestName);

        Console.WriteLine($"Generated {manifest.TotalExamples} examples across {manifest.Shards.Count} shards.");
        foreach (var shard in manifest.Shards)
        {
            Console.WriteLine($"  {shard.ShardId}: {shard.ExampleCount} examples, {shard.SizeBytes:N0} bytes");
        }

        Console.WriteLine($"Manifest saved to {System.IO.Path.Combine(corpusDir, $"manifest.{manifestName}.json")}");
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"generate-corpus failed: {ex}");
        return 1;
    }
}

static int TokenizeCorpusCommandLine(string[] args)
{
    const int TokenizerVocabCap = 5174;
    try
    {
        var maxVocab = TokenizerVocabCap;
        string? shardPrefix = null;

        if (args.Length > 1 && int.TryParse(args[1], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsedVocab) && parsedVocab > 100)
        {
            maxVocab = parsedVocab;
        }

        if (args.Length > 2 && !args[2].StartsWith("-", StringComparison.Ordinal))
        {
            shardPrefix = args[2];
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

        // Require literal "-shard-" so prefix "truckmate" cannot
        // accidentally slurp up "truckmate-v2-shard-*.txt".
        var pattern = shardPrefix is null ? "*.txt" : $"{shardPrefix}-shard-*.txt";
        var shardFiles = Directory.GetFiles(corpusDir, pattern)
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (shardFiles.Length == 0)
        {
            Console.Error.WriteLine($"No .txt shard files matching '{pattern}' found in {corpusDir}. Run generate-corpus first.");
            return 2;
        }

        Console.WriteLine($"Training tokenizer on {shardFiles.Length} shards (maxVocab={maxVocab}, pattern={pattern})…");

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

        if (tokenizer.VocabSize > TokenizerVocabCap)
        {
            Console.Error.WriteLine($"Vocab size {tokenizer.VocabSize} exceeds cap {TokenizerVocabCap}; aborting to preserve weight compatibility.");
            return 3;
        }

        var vocabPath = System.IO.Path.Combine(tokenizedDir, "vocab.json");
        if (shardPrefix is not null && File.Exists(vocabPath))
        {
            var backup = System.IO.Path.Combine(tokenizedDir, "vocab.v1.json");
            if (!File.Exists(backup))
            {
                File.Copy(vocabPath, backup);
                Console.WriteLine($"Backed up existing vocab to {backup}");
            }
        }
        tokenizer.SaveToFile(vocabPath);
        Console.WriteLine($"Vocabulary: {tokenizer.VocabSize} tokens → {vocabPath}");

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

static int SeedTasksCommandLine(string[] args)
{
    try
    {
        // Realistic per-task token budget. At ~1K tok/s of real training
        // throughput on a calibrated worker, 262,144 tokens ≈ a 4-minute
        // task — well within the 10-minute target duration the worker
        // capability calibration pass provisions for, and 32× the
        // previous 8,192-token stub that completed in under a second
        // and caused excessive weight-version churn.
        const long DefaultTokensPerTask = 262_144L;

        var count = 5;
        var tokensPerTask = DefaultTokensPerTask;
        if (args.Length > 1 && int.TryParse(args[1], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsedCount) && parsedCount > 0)
        {
            count = parsedCount;
        }

        if (args.Length > 2 && long.TryParse(args[2], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsedTokens) && parsedTokens > 0)
        {
            tokensPerTask = parsedTokens;
        }

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
                ShardOffset: (long)i * tokensPerTask,
                ShardLength: tokensPerTask,
                TokensPerTask: tokensPerTask,
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
        Console.WriteLine($"Seeded {inserted} tasks ({tokensPerTask:N0} tokens each) into {coordinator.DatabasePath}. Queue pending count: {pending}.");
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"seed-tasks failed: {ex}");
        return 1;
    }
}

/// <summary>
/// Enqueues tasks that point at real tokenized shards on disk.
/// Walks <c>&lt;dbDir&gt;/corpus/tokenized/*.bin</c>, slices each shard
/// into chunks of <c>tokensPerTask</c> int32s, and enqueues one task
/// per chunk with the real shardId + byte offset/length. Usage:
///   seed-real-tasks [tokensPerTask|auto] [maxTasksPerShard] [--shard-prefix NAME]
/// Passing <c>auto</c> for <c>tokensPerTask</c> sizes each task to
/// fit the configured <c>TargetTaskDurationSeconds</c> window at the
/// fleet-wide measured throughput from <c>gradient_events</c>; falls
/// back to 16,384 if the telemetry table has no recent events. The
/// optional <c>--shard-prefix</c> flag restricts seeding to
/// <c>{prefix}-shard-*.bin</c> files only, so v2 shards can be
/// queued without re-enqueuing v1 tasks that are already in flight.
/// </summary>
static int SeedRealTasksCommandLine(string[] args)
{
    try
    {
        const long DefaultTokensPerTask = 16_384L;
        var tokensPerTask = DefaultTokensPerTask;
        var maxPerShard = int.MaxValue;
        string? shardPrefix = null;
        var autoSize = false;
        if (args.Length > 1)
        {
            if (string.Equals(args[1], "auto", StringComparison.OrdinalIgnoreCase))
            {
                autoSize = true;
            }
            else if (long.TryParse(args[1], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var t) && t > 0)
            {
                tokensPerTask = t;
            }
        }
        if (args.Length > 2 && int.TryParse(args[2], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var m) && m > 0)
        {
            maxPerShard = m;
        }
        for (var i = 1; i < args.Length - 1; i++)
        {
            if (args[i] == "--shard-prefix")
            {
                shardPrefix = args[i + 1];
                break;
            }
        }

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

        if (autoSize)
        {
            using var telemetry = new SqliteTelemetryStore(
                $"Data Source={coordinator.DatabasePath}",
                TimeProvider.System);
            var globalTps = telemetry.GetGlobalMeasuredTokensPerSecond();
            if (globalTps is { } tps)
            {
                var targetSeconds = Math.Max(60, coordinator.TargetTaskDurationSeconds);
                // Round to the nearest multiple of 512 so shard
                // offsets stay 4-byte-aligned and task sizing moves
                // in perceptible steps as tps drifts.
                var raw = (long)Math.Round(tps * targetSeconds);
                tokensPerTask = Math.Max(512L, (raw / 512L) * 512L);
                Console.WriteLine(
                    $"auto-size: fleet tps={tps:F1} tok/s × target {targetSeconds}s → tokensPerTask={tokensPerTask:N0}");
            }
            else
            {
                tokensPerTask = DefaultTokensPerTask;
                Console.WriteLine(
                    $"auto-size: no recent gradient_events; falling back to tokensPerTask={tokensPerTask:N0}");
            }
        }

        var corpusDir = BitNetSharp.Distributed.Coordinator.Services.CorpusShardLocator
            .GetCorpusDirectory(coordinator.DatabasePath);
        var tokenizedDir = System.IO.Path.Combine(corpusDir, "tokenized");
        if (!System.IO.Directory.Exists(tokenizedDir))
        {
            Console.Error.WriteLine($"Tokenized corpus dir missing: {tokenizedDir}");
            return 3;
        }

        // Require literal "-shard-" so prefix "truckmate" doesn't
        // also sweep in "truckmate-v2-shard-*.bin" rows.
        var binPattern = shardPrefix is null ? "*.bin" : $"{shardPrefix}-shard-*.bin";
        var binFiles = System.IO.Directory.GetFiles(tokenizedDir, binPattern);
        Array.Sort(binFiles, StringComparer.Ordinal);
        if (binFiles.Length == 0)
        {
            Console.Error.WriteLine($"No tokenized .bin shards matching '{binPattern}' found in {tokenizedDir}");
            return 4;
        }
        if (shardPrefix is not null)
        {
            Console.WriteLine($"Restricting to shard-prefix '{shardPrefix}' → {binFiles.Length} shards");
        }

        using var store = new SqliteWorkQueueStore(
            $"Data Source={coordinator.DatabasePath}",
            TimeProvider.System);

        var bytesPerTask = tokensPerTask * sizeof(int);
        var now = DateTimeOffset.UtcNow;
        var totalInserted = 0;
        foreach (var file in binFiles)
        {
            var shardId = System.IO.Path.GetFileNameWithoutExtension(file);
            var fileSize = new System.IO.FileInfo(file).Length;
            var tokensInShard = fileSize / sizeof(int);
            var taskCount = (int)Math.Min(maxPerShard, tokensInShard / tokensPerTask);
            for (var i = 0; i < taskCount; i++)
            {
                var taskId = $"task-real-{Guid.NewGuid():N}";
                var offset = (long)i * bytesPerTask;
                store.EnqueuePending(new WorkTaskRecord(
                    TaskId: taskId,
                    WeightVersion: coordinator.InitialWeightVersion,
                    ShardId: shardId,
                    ShardOffset: offset,
                    ShardLength: bytesPerTask,
                    TokensPerTask: tokensPerTask,
                    // 1 local step keeps task wall-clock near the 10-minute
                    // target. K=4 caused 40-min tasks that outran lease
                    // calibration and loop-rejected on ownership expiry.
                    KLocalSteps: 1,
                    HyperparametersJson: "{}",
                    State: WorkTaskState.Pending,
                    AssignedWorkerId: null,
                    AssignedAtUtc: null,
                    DeadlineUtc: null,
                    Attempt: 0,
                    CreatedAtUtc: now,
                    CompletedAtUtc: null));
                totalInserted++;
            }
            Console.WriteLine($"  {shardId}: {taskCount} tasks ({fileSize:N0} bytes, {tokensInShard:N0} tokens)");
        }

        var pending = store.CountByState(WorkTaskState.Pending);
        Console.WriteLine($"Seeded {totalInserted} real tasks ({tokensPerTask:N0} tokens each) into {coordinator.DatabasePath}. Queue pending count: {pending}.");
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"seed-real-tasks failed: {ex}");
        return 1;
    }
}

/// <summary>
/// Dumps the most recent worker_logs rows to stdout. Diagnostic CLI
/// for triaging worker behavior without needing a browser / OIDC
/// session. Usage: dump-events [limit=40] [minLevel]
/// </summary>
static int DumpEventsCommandLine(string[] args)
{
    try
    {
        var limit = 40;
        string? minLevel = null;
        if (args.Length > 1 && int.TryParse(args[1], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var n) && n > 0)
        {
            limit = n;
        }
        if (args.Length > 2)
        {
            minLevel = args[2];
        }

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

        using var store = new SqliteLogStore($"Data Source={coordinator.DatabasePath}");
        var rows = store.Query(limit: limit, minLevel: minLevel);
        foreach (var r in rows)
        {
            var ts = DateTimeOffset.FromUnixTimeSeconds(r.TimestampUnix).ToLocalTime().ToString("HH:mm:ss");
            var worker = (r.WorkerId ?? "-").Length > 8 ? r.WorkerId!.Substring(0, 8) : (r.WorkerId ?? "-");
            Console.WriteLine($"{ts} {r.Level,-5} {worker,-8} {r.Category,-30} {r.Message}");
            if (!string.IsNullOrEmpty(r.Exception))
            {
                Console.WriteLine($"  ex: {r.Exception}");
            }
        }
        Console.WriteLine($"-- {rows.Count} rows, total={store.TotalCount()}");
        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"dump-events failed: {ex}");
        return 1;
    }
}

/// <summary>
/// Deletes rows in the work queue whose state matches one of the
/// args (default: <c>Pending</c>). Coordinator must be stopped first
/// or the write will race the service's own writes — returns 2 if
/// the DB is locked.
/// </summary>
static int PurgePendingCommandLine(string[] args)
{
    try
    {
        // Parse states. No args => Pending only. --all-queued => Pending+Assigned.
        var states = new List<WorkTaskState> { WorkTaskState.Pending };
        for (var i = 1; i < args.Length; i++)
        {
            if (string.Equals(args[i], "--all-queued", StringComparison.OrdinalIgnoreCase))
            {
                states = new List<WorkTaskState> { WorkTaskState.Pending, WorkTaskState.Assigned };
            }
            else if (string.Equals(args[i], "--include-assigned", StringComparison.OrdinalIgnoreCase))
            {
                if (!states.Contains(WorkTaskState.Assigned)) states.Add(WorkTaskState.Assigned);
            }
        }

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

        using var store = new SqliteWorkQueueStore($"Data Source={coordinator.DatabasePath}", TimeProvider.System);
        var deleted = store.DeleteByStates(states);
        Console.WriteLine($"purge-pending: deleted {deleted} rows in states [{string.Join(", ", states)}]");
        return 0;
    }
    catch (Microsoft.Data.Sqlite.SqliteException ex) when (ex.SqliteErrorCode == 5)
    {
        Console.Error.WriteLine($"purge-pending: database locked — stop the BitNetCoordinator service first. ({ex.Message})");
        return 2;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"purge-pending failed: {ex}");
        return 1;
    }
}

/// <summary>
/// Deletes pending tasks whose shard_id starts with the given
/// prefix. Use to retire an older corpus (e.g. v1) without touching
/// Assigned rows that a worker may still be computing. Service must
/// be stopped so the write doesn't race active claims.
/// </summary>
static int PurgeShardsCommandLine(string[] args)
{
    if (args.Length < 2 || string.IsNullOrWhiteSpace(args[1]))
    {
        Console.Error.WriteLine("usage: purge-shards <shard-prefix>  (e.g. purge-shards truckmate-shard)");
        return 2;
    }

    var prefix = args[1];

    try
    {
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

        using var store = new SqliteWorkQueueStore($"Data Source={coordinator.DatabasePath}", TimeProvider.System);
        var deleted = store.DeletePendingByShardPrefix(prefix);
        Console.WriteLine($"purge-shards: deleted {deleted} Pending rows with shard_id LIKE '{prefix}%'");
        return 0;
    }
    catch (Microsoft.Data.Sqlite.SqliteException ex) when (ex.SqliteErrorCode == 5)
    {
        Console.Error.WriteLine($"purge-shards: database locked — stop the BitNetCoordinator service first. ({ex.Message})");
        return 2;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"purge-shards failed: {ex}");
        return 1;
    }
}

/// <summary>
/// Deletes every row from gradient_events. Used to drop legacy
/// synthetic benchmark samples whose tokens/sec is three orders of
/// magnitude above real backprop; those rows otherwise poison
/// lease calibration until the 30-minute window decays them.
/// Service must be stopped first.
/// </summary>
static int PurgeTelemetryCommandLine(string[] args)
{
    try
    {
        var config = new ConfigurationBuilder()
            .SetBasePath(System.IO.Path.GetDirectoryName(typeof(Program).Assembly.Location) ?? ".")
            .AddJsonFile("appsettings.json", optional: true)
            .AddJsonFile($"appsettings.{Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production"}.json", optional: true)
            .AddEnvironmentVariables()
            .Build();
        var coord = new CoordinatorOptions();
        config.GetSection(CoordinatorOptions.SectionName).Bind(coord);
        if (string.IsNullOrWhiteSpace(coord.DatabasePath))
        {
            Console.Error.WriteLine("Coordinator:DatabasePath is not set.");
            return 2;
        }

        using var conn = new Microsoft.Data.Sqlite.SqliteConnection(
            $"Data Source={coord.DatabasePath}");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "DELETE FROM gradient_events;";
        var deleted = cmd.ExecuteNonQuery();
        Console.WriteLine($"purge-telemetry: deleted {deleted} gradient_events rows");
        return 0;
    }
    catch (Microsoft.Data.Sqlite.SqliteException ex) when (ex.SqliteErrorCode == 5)
    {
        Console.Error.WriteLine($"purge-telemetry: database locked — stop the BitNetCoordinator service first. ({ex.Message})");
        return 2;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"purge-telemetry failed: {ex}");
        return 1;
    }
}

/// <summary>
/// Reads gradient_events + tasks directly against the coordinator DB
/// and prints a small diagnostic table. Used to sanity-check lease
/// calibration (<c>GetMeasuredTokensPerSecond</c>) and to see which
/// tasks are currently Assigned with what deadlines.
/// </summary>
static int DumpTelemetryCommandLine(string[] args)
{
    try
    {
        var config = new ConfigurationBuilder()
            .SetBasePath(System.IO.Path.GetDirectoryName(typeof(Program).Assembly.Location) ?? ".")
            .AddJsonFile("appsettings.json", optional: true)
            .AddJsonFile($"appsettings.{Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production"}.json", optional: true)
            .AddEnvironmentVariables()
            .Build();
        var coord = new CoordinatorOptions();
        config.GetSection(CoordinatorOptions.SectionName).Bind(coord);
        if (string.IsNullOrWhiteSpace(coord.DatabasePath))
        {
            Console.Error.WriteLine("Coordinator:DatabasePath is not set.");
            return 2;
        }

        using var conn = new Microsoft.Data.Sqlite.SqliteConnection(
            $"Data Source={coord.DatabasePath};Mode=ReadOnly;Cache=Shared");
        conn.Open();

        Console.WriteLine("== Per-worker gradient_events rollup ==");
        using (var cmd = conn.CreateCommand())
        {
            cmd.CommandText = "SELECT client_id, COUNT(1), SUM(tokens_seen), SUM(wall_clock_ms), MAX(received_at) " +
                              "FROM gradient_events GROUP BY client_id ORDER BY COUNT(1) DESC";
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                var cid = r.GetString(0);
                var n = r.GetInt64(1);
                var tok = r.GetInt64(2);
                var ms = r.GetInt64(3);
                var last = r.GetInt64(4);
                var tps = ms > 0 ? tok / (ms / 1000.0) : 0.0;
                var lastStr = DateTimeOffset.FromUnixTimeSeconds(last).ToLocalTime().ToString("HH:mm:ss");
                Console.WriteLine($"  {cid,-40}  n={n,-5} tok={tok,-10} ms={ms,-10} tps={tps,-10:F2} last={lastStr}");
            }
        }

        Console.WriteLine();
        Console.WriteLine("== Last 12 gradient events (by id desc) ==");
        using (var cmd = conn.CreateCommand())
        {
            cmd.CommandText = "SELECT id, client_id, tokens_seen, wall_clock_ms, received_at " +
                              "FROM gradient_events ORDER BY id DESC LIMIT 12";
            using var r = cmd.ExecuteReader();
            while (r.Read())
            {
                var id = r.GetInt64(0);
                var cid = r.GetString(1);
                var tok = r.GetInt64(2);
                var ms = r.GetInt64(3);
                var at = DateTimeOffset.FromUnixTimeSeconds(r.GetInt64(4)).ToLocalTime().ToString("HH:mm:ss");
                Console.WriteLine($"  id={id,-7} {cid,-40} tok={tok,-6} ms={ms,-8} at={at}");
            }
        }

        Console.WriteLine();
        Console.WriteLine("== Current Pending/Assigned (top 10 by assigned_at desc) ==");
        using (var cmd = conn.CreateCommand())
        {
            cmd.CommandText = "SELECT task_id, assigned_to, assigned_at, deadline_at, tokens_per_task, k_local_steps, state " +
                              "FROM tasks WHERE state IN ('Assigned','Pending') " +
                              "ORDER BY COALESCE(assigned_at, 0) DESC LIMIT 10";
            using var r = cmd.ExecuteReader();
            var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            while (r.Read())
            {
                var tid = r.GetString(0);
                var worker = r.IsDBNull(1) ? "-" : r.GetString(1);
                var aa = r.IsDBNull(2) ? "-" : DateTimeOffset.FromUnixTimeSeconds(r.GetInt64(2)).ToLocalTime().ToString("HH:mm:ss");
                string da = "-";
                string remaining = "-";
                if (!r.IsDBNull(3))
                {
                    var deadlineUnix = r.GetInt64(3);
                    da = DateTimeOffset.FromUnixTimeSeconds(deadlineUnix).ToLocalTime().ToString("HH:mm:ss");
                    remaining = $"{deadlineUnix - now}s";
                }
                var tok = r.GetInt64(4);
                var k = r.GetInt32(5);
                var st = r.GetString(6);
                Console.WriteLine($"  {tid,-42} state={st,-9} worker={worker,-38} assigned={aa} deadline={da} rem={remaining,-6} tok={tok} K={k}");
            }
        }

        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"dump-telemetry failed: {ex}");
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

    /// <summary>
    /// Authorization-policy name applied to every protected worker
    /// endpoint. Centralized here so tests, middleware, and the host
    /// share the same string.
    /// </summary>
    public static class WorkerApiKeyAuth
    {
        public const string PolicyName = "WorkerApiKeyPolicy";
    }

    /// <summary>
    /// Wires the <see cref="Authentication.ApiKeyAuthenticationOptions.ExpectedKey"/>
    /// factory to the live <see cref="CoordinatorOptions"/> snapshot.
    /// Kept as a class so DI can resolve its dependency chain after
    /// the auth scheme is registered.
    /// </summary>
    internal sealed class ApiKeyPostConfigure :
        IPostConfigureOptions<Authentication.ApiKeyAuthenticationOptions>
    {
        private readonly IOptionsMonitor<CoordinatorOptions> _coordinator;

        public ApiKeyPostConfigure(IOptionsMonitor<CoordinatorOptions> coordinator)
        {
            _coordinator = coordinator;
        }

        public void PostConfigure(string? name, Authentication.ApiKeyAuthenticationOptions options)
        {
            options.ExpectedKey = () => _coordinator.CurrentValue.WorkerApiKey;
        }
    }
}
