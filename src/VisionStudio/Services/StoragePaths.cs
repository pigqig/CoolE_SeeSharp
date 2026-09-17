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

    public StoragePaths(IWebHostEnvironment env, IConfiguration config)
    {
        var content = env.ContentRootPath;
        var baseDir = AppContext.BaseDirectory;
        var publishedAppData = Path.Combine(baseDir, "App_Data");
        var contentAppData = Path.Combine(content, "App_Data");
        var published = Directory.Exists(Path.Combine(publishedAppData, "models"))
            || File.Exists(Path.Combine(publishedAppData, "models", "face_detection_yunet_2023mar.onnx"))
            || Directory.Exists(Path.Combine(contentAppData, "models"));

        var candidates = new[]
        {
            Path.GetFullPath(Path.Combine(content, "..", "..")),
            Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..")),
            Path.GetFullPath(baseDir),
            content
        };

        RepoRoot = candidates.FirstOrDefault(p =>
            File.Exists(Path.Combine(p, "models", "face_detection_yunet_2023mar.onnx")) ||
            File.Exists(Path.Combine(p, "docs", "VISION_PLAN.md"))) ?? content;

        var appData = Directory.Exists(Path.Combine(publishedAppData, "models"))
            ? publishedAppData
            : Directory.Exists(Path.Combine(contentAppData, "models"))
                ? contentAppData
                : Path.Combine(content, "App_Data");

        ModelsRoot = FirstExisting(
            Path.Combine(appData, "models"),
            Path.Combine(RepoRoot, "models"),
            Path.Combine(publishedAppData, "models"));

        SamplesRoot = FirstExisting(
            Path.Combine(appData, "samples"),
            Path.Combine(RepoRoot, "data", "samples"),
            Path.Combine(publishedAppData, "samples"));

        PythonRoot = FirstExisting(
            Path.Combine(appData, "python"),
            Path.Combine(RepoRoot, "python"),
            Path.Combine(publishedAppData, "python"));

        var configuredData = config["Vision:DataRoot"];
        DataRoot = !string.IsNullOrWhiteSpace(configuredData)
            ? Path.GetFullPath(configuredData)
            : published
                ? Path.Combine(appData, "runtime")
                : Path.Combine(RepoRoot, "data", "runtime");

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
        Directory.CreateDirectory(Path.Combine(content, "logs"));
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
