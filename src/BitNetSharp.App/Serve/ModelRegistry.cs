using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace BitNetSharp.App.Serve;

/// <summary>
/// Registry of loaded hosted models keyed by client-requested name. Resolution
/// tolerates Ollama's mandatory <c>:latest</c> tag suffix because Open WebUI
/// always appends it. Name matching is case-insensitive to match Ollama's
/// server behavior.
/// </summary>
public sealed class ModelRegistry
{
    private readonly Dictionary<string, RegisteredModel> _byName = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<RegisteredModel> _ordered = new();

    public void Register(IHostedAgentModel model, ModelCard card)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(card);

        var entry = new RegisteredModel(model, card);
        _byName[card.Name] = entry;
        _byName[card.NameWithTag] = entry;
        _ordered.Add(entry);
    }

    public bool TryResolve(string? requested, [NotNullWhen(true)] out RegisteredModel? model)
    {
        model = null;
        if (string.IsNullOrWhiteSpace(requested))
        {
            return false;
        }

        if (_byName.TryGetValue(requested, out var direct))
        {
            model = direct;
            return true;
        }

        // Strip trailing :tag and retry.
        int colon = requested.IndexOf(':', StringComparison.Ordinal);
        if (colon > 0)
        {
            string baseName = requested.Substring(0, colon);
            if (_byName.TryGetValue(baseName, out var stripped))
            {
                model = stripped;
                return true;
            }
        }

        return false;
    }

    public IReadOnlyList<RegisteredModel> Enumerate() => _ordered;
}

public sealed record RegisteredModel(IHostedAgentModel Model, ModelCard Card);
