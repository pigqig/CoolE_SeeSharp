using System.Text.Json;
using System.Text.Json.Serialization;
using VisionStudio.Domain;

namespace VisionStudio.Services;

public sealed class StoragePaths
{
    public string RepoRoot { get; }
    public string DataRoot { get; }
    public string ModelsRoot { get; }
    public string DatasetsRoot { get; }
    public string JobsRoot { get; }
    public string UploadsRoot { get; }
    public string GalleryRoot { get; }
    public string HistoryRoot { get; }
    public string SamplesRoot { get; }
    public string PythonRoot { get; }
    public string RuntimeRoot { get; }

    public StoragePaths(IWebHostEnvironment env)
    {
        var content = env.ContentRootPath;
        var candidates = new[]
        {
            Path.GetFullPath(Path.Combine(content, "..", "..")),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..")),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory)),
            content
        };

        RepoRoot = candidates.FirstOrDefault(p =>
            File.Exists(Path.Combine(p, "models", "face_detection_yunet_2023mar.onnx")) ||
            File.Exists(Path.Combine(p, "docs", "VISION_PLAN.md"))) ?? content;

        var appData = Path.Combine(AppContext.BaseDirectory, "App_Data");
        ModelsRoot = FirstExisting(
            Path.Combine(RepoRoot, "models"),
            Path.Combine(appData, "models"),
            Path.Combine(content, "App_Data", "models"));

        SamplesRoot = FirstExisting(
            Path.Combine(RepoRoot, "data", "samples"),
            Path.Combine(appData, "samples"),
            Path.Combine(content, "App_Data", "samples"));

        PythonRoot = FirstExisting(
            Path.Combine(RepoRoot, "python"),
            Path.Combine(appData, "python"),
            Path.Combine(content, "App_Data", "python"));

        DataRoot = Path.Combine(RepoRoot, "data", "runtime");
        DatasetsRoot = Path.Combine(DataRoot, "datasets");
        JobsRoot = Path.Combine(DataRoot, "jobs");
        UploadsRoot = Path.Combine(DataRoot, "uploads");
        GalleryRoot = Path.Combine(DataRoot, "gallery");
        HistoryRoot = Path.Combine(DataRoot, "history");
        RuntimeRoot = DataRoot;

        Directory.CreateDirectory(DatasetsRoot);
        Directory.CreateDirectory(JobsRoot);
        Directory.CreateDirectory(UploadsRoot);
        Directory.CreateDirectory(GalleryRoot);
        Directory.CreateDirectory(HistoryRoot);
        Directory.CreateDirectory(Path.Combine(DataRoot, "models"));
    }

    private static string FirstExisting(params string[] paths)
    {
        foreach (var p in paths)
        {
            if (Directory.Exists(p) || File.Exists(p))
                return p;
        }
        return paths[0];
    }
}

public static class JsonStore
{
    public static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() }
    };

    public static T Load<T>(string path, T fallback) where T : class
    {
        if (!File.Exists(path))
            return fallback;
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<T>(json, Options) ?? fallback;
    }

    public static void Save<T>(string path, T value)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(value, Options));
    }
}
