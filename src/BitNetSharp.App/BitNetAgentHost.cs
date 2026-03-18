using BitNetSharp.Core;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace BitNetSharp.App;

public sealed record BitNetHostSummary(
    string AgentName,
    string ModelId,
    string DisplayName,
    string PrimaryLanguage,
    string HostingFramework,
    VerbosityLevel Verbosity);

public static class BitNetAgentHost
{
    public static IHost Build(BitNetPaperModel model) => Build(new BitNetHostedAgentModel(model));

    public static IHost Build(IHostedAgentModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        var builder = Host.CreateApplicationBuilder();
        var chatClient = new HostedModelChatClient(model);

        builder.Services.AddSingleton(model);
        builder.Services.AddSingleton<IChatClient>(chatClient);
        builder.Services.AddSingleton(new BitNetHostSummary(
            model.AgentName,
            model.ModelId,
            model.DisplayName,
            model.PrimaryLanguage,
            "Microsoft Agent Framework",
            model.Verbosity));

        builder.AddAIAgent(
                model.AgentName,
                model.SystemPrompt,
                chatClient)
            .WithInMemorySessionStore();

        if (model is BitNetHostedAgentModel bitNetModel)
        {
            builder.Services.AddSingleton(bitNetModel.Model);
        }

        return builder.Build();
    }
}
