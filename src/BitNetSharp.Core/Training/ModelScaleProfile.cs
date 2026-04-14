using System.Text.Json;

namespace BitNetSharp.Core.Training;

/// <summary>
/// Complete scale profile for all layers in a model, used by the
/// integer training pass. Serializable to JSON for persistence.
/// </summary>
public sealed record ModelScaleProfile
{
    public required string ModelId { get; init; }
    public required string ArchitectureHash { get; init; }
    public required DateTimeOffset CalibratedAt { get; init; }
    public required int CalibrationSteps { get; init; }
    public required IReadOnlyList<LayerScaleProfile> Layers { get; init; }

    public LayerScaleProfile? GetLayer(string layerName) =>
        Layers.FirstOrDefault(l => string.Equals(l.LayerName, layerName, StringComparison.Ordinal));

    public void SaveToFile(string path) =>
        File.WriteAllText(path, JsonSerializer.Serialize(this,
            new JsonSerializerOptions { WriteIndented = true }));

    public static ModelScaleProfile LoadFromFile(string path) =>
        JsonSerializer.Deserialize<ModelScaleProfile>(File.ReadAllText(path))
        ?? throw new InvalidDataException($"Could not deserialize scale profile from {path}");
}
