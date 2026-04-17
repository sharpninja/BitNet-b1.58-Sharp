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

        var shardFiles = Directory.GetFiles(corpusDir, "*.txt")
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (shardFiles.Length == 0)
        {
            Console.Error.WriteLine($"No .txt shard files found in {corpusDir}. Run generate-corpus first.");
            return 2;
        }

        Console.WriteLine($"Training tokenizer on {shardFiles.Length} shards (maxVocab={maxVocab})…");

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
///   seed-real-tasks [tokensPerTask] [maxTasksPerShard]
/// </summary>
static int SeedRealTasksCommandLine(string[] args)
{
    try
    {
        var tokensPerTask = 16_384L;
        var maxPerShard = int.MaxValue;
        if (args.Length > 1 && long.TryParse(args[1], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var t) && t > 0)
        {
            tokensPerTask = t;
        }
        if (args.Length > 2 && int.TryParse(args[2], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var m) && m > 0)
        {
            maxPerShard = m;
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

        var corpusDir = BitNetSharp.Distributed.Coordinator.Services.CorpusShardLocator
            .GetCorpusDirectory(coordinator.DatabasePath);
        var tokenizedDir = System.IO.Path.Combine(corpusDir, "tokenized");
        if (!System.IO.Directory.Exists(tokenizedDir))
        {
            Console.Error.WriteLine($"Tokenized corpus dir missing: {tokenizedDir}");
            return 3;
        }

        var binFiles = System.IO.Directory.GetFiles(tokenizedDir, "*.bin");
        Array.Sort(binFiles, StringComparer.Ordinal);
        if (binFiles.Length == 0)
        {
            Console.Error.WriteLine($"No tokenized .bin shards found in {tokenizedDir}");
            return 4;
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
                    KLocalSteps: 4,
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
