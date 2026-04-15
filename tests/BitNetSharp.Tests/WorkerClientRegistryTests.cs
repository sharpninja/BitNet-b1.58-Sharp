#if NET10_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Identity;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd tests for the in-memory OAuth client registry that the
/// coordinator seeds from <see cref="CoordinatorOptions.WorkerClients"/>
/// at startup and mutates via admin rotate.
/// </summary>
public sealed class WorkerClientRegistryTests
{
    private static List<WorkerClientOptions> SampleOptions() => new()
    {
        new WorkerClientOptions
        {
            ClientId     = "worker-alpha",
            ClientSecret = "alpha-secret",
            DisplayName  = "Alpha Box"
        },
        new WorkerClientOptions
        {
            ClientId     = "worker-beta",
            ClientSecret = "beta-secret",
            DisplayName  = "Beta Box"
        }
    };

    [Fact]
    public void Seed_loads_the_configured_clients_in_order()
    {
        var registry = new WorkerClientRegistry();
        registry.Seed(SampleOptions());

        var list = registry.ListAll();
        Assert.Equal(2, list.Count);
        Assert.Equal("worker-alpha", list[0].ClientId);
        Assert.Equal("worker-beta",  list[1].ClientId);
    }

    [Fact]
    public void Seed_falls_back_to_client_id_when_display_name_blank()
    {
        var registry = new WorkerClientRegistry();
        registry.Seed(new[]
        {
            new WorkerClientOptions
            {
                ClientId     = "worker-no-name",
                ClientSecret = "secret",
                DisplayName  = ""
            }
        });

        var entry = registry.Find("worker-no-name");
        Assert.NotNull(entry);
        Assert.Equal("worker-no-name", entry!.DisplayName);
    }

    [Fact]
    public void Seed_rejects_blank_client_id()
    {
        var registry = new WorkerClientRegistry();
        Assert.Throws<InvalidOperationException>(() => registry.Seed(new[]
        {
            new WorkerClientOptions { ClientId = "", ClientSecret = "secret" }
        }));
    }

    [Fact]
    public void Seed_rejects_blank_client_secret()
    {
        var registry = new WorkerClientRegistry();
        Assert.Throws<InvalidOperationException>(() => registry.Seed(new[]
        {
            new WorkerClientOptions { ClientId = "worker-no-secret", ClientSecret = "" }
        }));
    }

    [Fact]
    public void Find_returns_null_for_unknown_client()
    {
        var registry = new WorkerClientRegistry();
        registry.Seed(SampleOptions());
        Assert.Null(registry.Find("worker-missing"));
    }

    [Fact]
    public void Rotate_generates_a_new_secret_and_replaces_the_existing_entry()
    {
        var registry = new WorkerClientRegistry();
        registry.Seed(SampleOptions());
        var originalSecret = registry.Find("worker-alpha")!.PlainTextSecret;

        var freshSecret = registry.Rotate("worker-alpha");

        Assert.NotEqual(originalSecret, freshSecret);
        Assert.Equal(freshSecret, registry.Find("worker-alpha")!.PlainTextSecret);
    }

    [Fact]
    public void Rotate_generated_secret_is_url_safe_and_long()
    {
        var registry = new WorkerClientRegistry();
        registry.Seed(SampleOptions());
        var freshSecret = registry.Rotate("worker-alpha");

        Assert.True(freshSecret.Length >= 32, "Fresh secret should be at least 32 characters long.");
        Assert.DoesNotContain('+', freshSecret);
        Assert.DoesNotContain('/', freshSecret);
        Assert.DoesNotContain('=', freshSecret);
    }

    [Fact]
    public void Rotate_throws_on_unknown_client()
    {
        var registry = new WorkerClientRegistry();
        registry.Seed(SampleOptions());
        Assert.Throws<KeyNotFoundException>(() => registry.Rotate("no-such-worker"));
    }

    [Fact]
    public void ToDuendeClients_emits_one_client_per_registry_entry()
    {
        var registry = new WorkerClientRegistry();
        registry.Seed(SampleOptions());

        var duendeClients = registry.ToDuendeClients(accessTokenLifetimeSeconds: 900).ToList();

        Assert.Equal(2, duendeClients.Count);
        var alpha = duendeClients.First(c => c.ClientId == "worker-alpha");
        Assert.Contains(IdentityServerResources.WorkerScopeName, alpha.AllowedScopes);
        Assert.Equal(900, alpha.AccessTokenLifetime);
    }

    [Fact]
    public void IsEmpty_flips_after_seeding()
    {
        var registry = new WorkerClientRegistry();
        Assert.True(registry.IsEmpty);
        Assert.Equal(0, registry.Count);

        registry.Seed(SampleOptions());

        Assert.False(registry.IsEmpty);
        Assert.Equal(2, registry.Count);
    }
}
#endif
