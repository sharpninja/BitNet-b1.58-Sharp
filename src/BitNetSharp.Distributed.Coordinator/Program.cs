using System;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

// ────────────────────────────────────────────────────────────────────────
//  BitNetSharp.Distributed.Coordinator — Phase D-1 skeleton
// ────────────────────────────────────────────────────────────────────────
//  This is a minimal ASP.NET Core host that exposes only the /health
//  endpoint and a placeholder /status endpoint wired to the SQLite work
//  queue store. Subsequent D-1 commits will layer on /register, /work,
//  /gradient, /weights, and /heartbeat controllers plus bearer-token
//  auth. The skeleton is here first so every follow-up commit ships
//  against a running process with real SQL state rather than against a
//  vacuum.
// ────────────────────────────────────────────────────────────────────────

var builder = WebApplication.CreateBuilder(args);

var databasePath = builder.Configuration["Coordinator:DatabasePath"]
    ?? Environment.GetEnvironmentVariable("BITNET_COORDINATOR_DB")
    ?? "coordinator.db";
var connectionString = $"Data Source={databasePath};Cache=Shared";

builder.Services.AddSingleton<TimeProvider>(TimeProvider.System);
builder.Services.AddSingleton<SqliteWorkQueueStore>(serviceProvider =>
    new SqliteWorkQueueStore(connectionString, serviceProvider.GetRequiredService<TimeProvider>()));

var app = builder.Build();

app.MapGet("/health", () => Results.Ok(new
{
    status = "ok",
    time   = DateTimeOffset.UtcNow,
    phase  = "D-1-skeleton"
}));

app.MapGet("/status", (SqliteWorkQueueStore store) =>
{
    var pending  = store.CountByState(WorkTaskState.Pending);
    var assigned = store.CountByState(WorkTaskState.Assigned);
    var done     = store.CountByState(WorkTaskState.Done);
    var failed   = store.CountByState(WorkTaskState.Failed);
    return Results.Ok(new
    {
        tasks = new
        {
            pending,
            assigned,
            done,
            failed
        },
        database = databasePath,
        time     = DateTimeOffset.UtcNow
    });
});

app.Run();

// Exposed so WebApplicationFactory-based integration tests in
// BitNetSharp.Tests can bootstrap the coordinator without spawning a
// separate process. The empty partial here is enough for the test SDK
// to pick the Program type up.
public partial class Program;
