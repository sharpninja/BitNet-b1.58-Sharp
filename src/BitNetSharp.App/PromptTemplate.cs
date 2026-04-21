using System.Text;
using Microsoft.Extensions.AI;

namespace BitNetSharp.App;

/// <summary>
/// Flattens a multi-turn chat history into a single model prompt using a simple
/// role-tagged template. Centralized so both <see cref="IHostedAgentModel"/> default
/// streaming and callers that bypass streaming produce identical prompts.
/// </summary>
public static class PromptTemplate
{
    public const string SystemTag = "[SYSTEM]";
    public const string UserTag = "[USER]";
    public const string AssistantTag = "[ASSISTANT]";
    public const string ToolTag = "[TOOL]";

    /// <summary>
    /// Produces a single prompt string: system prompt first, then turn-tagged history,
    /// terminated with an open assistant tag so the model continues the assistant turn.
    /// </summary>
    public static string FlattenHistory(string systemPrompt, IReadOnlyList<ChatMessage> history)
    {
        ArgumentNullException.ThrowIfNull(history);

        var builder = new StringBuilder();
        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            builder.Append(SystemTag).Append('\n').Append(systemPrompt).Append('\n');
        }

        foreach (var message in history)
        {
            var text = message.Text;
            if (string.IsNullOrEmpty(text))
            {
                continue;
            }

            var tag = TagFor(message.Role);
            builder.Append(tag).Append('\n').Append(text).Append('\n');
        }

        builder.Append(AssistantTag).Append('\n');
        return builder.ToString();
    }

    private static string TagFor(ChatRole role)
    {
        if (role == ChatRole.System) return SystemTag;
        if (role == ChatRole.User) return UserTag;
        if (role == ChatRole.Assistant) return AssistantTag;
        if (role == ChatRole.Tool) return ToolTag;
        return UserTag;
    }
}
