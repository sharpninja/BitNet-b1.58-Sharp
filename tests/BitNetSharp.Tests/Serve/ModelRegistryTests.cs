using System.Collections.Generic;
using BitNetSharp.App;
using BitNetSharp.App.Serve;

namespace BitNetSharp.Tests.Serve;

public sealed class ModelRegistryTests
{
    [Fact]
    public void Register_EnumerateReturnsInOrder()
    {
        var stub = new StubHostedAgentModel();
        var registry = new ModelRegistry();
        var card = ModelCard.ForHostedModel(stub, 10L, 40L, new Dictionary<string, object?>());
        registry.Register(stub, card);
        var list = registry.Enumerate();
        Assert.Single(list);
        Assert.Equal(stub, list[0].Model);
    }

    [Fact]
    public void TryResolve_MatchesBaseName()
    {
        var stub = new StubHostedAgentModel();
        var registry = new ModelRegistry();
        var card = ModelCard.ForHostedModel(stub, 10L, 40L, new Dictionary<string, object?>());
        registry.Register(stub, card);
        Assert.True(registry.TryResolve("bitnet-b1.58-sharp", out var a));
        Assert.Equal(stub, a!.Model);
    }

    [Fact]
    public void TryResolve_MatchesLatestTagSuffix()
    {
        var stub = new StubHostedAgentModel();
        var registry = new ModelRegistry();
        var card = ModelCard.ForHostedModel(stub, 10L, 40L, new Dictionary<string, object?>());
        registry.Register(stub, card);
        Assert.True(registry.TryResolve("bitnet-b1.58-sharp:latest", out var a));
        Assert.Equal(stub, a!.Model);
    }

    [Fact]
    public void TryResolve_StripsCustomTagAndMatchesBase()
    {
        var stub = new StubHostedAgentModel();
        var registry = new ModelRegistry();
        var card = ModelCard.ForHostedModel(stub, 10L, 40L, new Dictionary<string, object?>());
        registry.Register(stub, card);
        Assert.True(registry.TryResolve("bitnet-b1.58-sharp:v0.2", out var a));
        Assert.Equal(stub, a!.Model);
    }

    [Fact]
    public void TryResolve_UnknownName_ReturnsFalse()
    {
        var registry = new ModelRegistry();
        Assert.False(registry.TryResolve("unknown-model", out _));
    }
}
