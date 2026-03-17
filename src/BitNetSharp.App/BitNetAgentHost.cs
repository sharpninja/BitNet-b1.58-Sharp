using BitNetSharp.Core;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace BitNetSharp.App;

public sealed record BitNetHostSummary(
    string AgentName,
    string PrimaryLanguage,
    string HostingFramework,
    VerbosityLevel Verbosity);

public static class BitNetAgentHost
{
    public static IHost Build(BitNetPaperModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        var builder = Host.CreateApplicationBuilder();
        var chatClient = new BitNetChatClient(model);

        builder.Services.AddSingleton(model);
        builder.Services.AddSingleton<IChatClient>(chatClient);
        builder.Services.AddSingleton(new BitNetHostSummary(
            "bitnet-b1.58-sharp",
            model.Options.PrimaryLanguage,
            "Microsoft Agent Framework",
            model.Options.Verbosity));

        builder.AddAIAgent(
                "bitnet-b1.58-sharp",
                "Respond in clear American English using the paper-aligned BitNet b1.58 transformer diagnostics.",
                chatClient)
            .WithInMemorySessionStore();

        return builder.Build();
    }
}
