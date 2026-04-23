using System;
using System.Collections.Generic;
using System.Linq;
using BitNetSharp.App.Serve.Dto;
using Microsoft.Extensions.AI;

namespace BitNetSharp.App.Serve.Inference;

/// <summary>
/// Translates an Ollama/OpenAI messages array into the single-string prompt
/// that <see cref="IHostedAgentModel.GetResponseAsync"/> expects. Reuses the
/// repo's canonical flattener via <see cref="PromptTemplate.FlattenHistory"/>
/// so responses match the <c>chat</c> CLI subcommand exactly.
/// </summary>
internal static class ChatPromptAssembler
{
    public static string Assemble(string systemPrompt, IEnumerable<OllamaChatMessage> messages)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var history = messages
            .Select(m => new ChatMessage(MapRole(m.Role), m.Content ?? string.Empty))
            .ToList();
        return PromptTemplate.FlattenHistory(systemPrompt, history);
    }

    public static string Assemble(string systemPrompt, IEnumerable<OpenAIChatMessage> messages)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var history = messages
            .Select(m => new ChatMessage(MapRole(m.Role), m.Content ?? string.Empty))
            .ToList();
        return PromptTemplate.FlattenHistory(systemPrompt, history);
    }

    private static ChatRole MapRole(string role) => role?.ToLowerInvariant() switch
    {
        "system" => ChatRole.System,
        "assistant" => ChatRole.Assistant,
        "tool" => ChatRole.Tool,
        _ => ChatRole.User,
    };
}
