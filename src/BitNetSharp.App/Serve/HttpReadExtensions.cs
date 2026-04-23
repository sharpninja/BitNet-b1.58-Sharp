using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace BitNetSharp.App.Serve;

internal static class HttpReadExtensions
{
    public static async Task<T?> ReadJsonAsync<T>(this HttpRequest request, CancellationToken cancellationToken = default)
    {
        return await JsonSerializer.DeserializeAsync<T>(request.Body, ServeJson.Options, cancellationToken).ConfigureAwait(false);
    }
}
