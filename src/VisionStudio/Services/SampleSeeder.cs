using OpenCvSharp;
using VisionStudio.Domain;

namespace VisionStudio.Services;

public sealed class SampleSeeder
{
    private readonly StoragePaths _paths;
    private readonly DatasetService _datasets;
    private readonly FaceEngine _faces;
    private readonly GalleryService _gallery;
    private readonly PlateEngine _plates;
    private readonly ILogger<SampleSeeder> _log;

    public SampleSeeder(
        StoragePaths paths,
        DatasetService datasets,
        FaceEngine faces,
        GalleryService gallery,
        PlateEngine plates,
        ILogger<SampleSeeder> log)
    {
        _paths = paths;
        _datasets = datasets;
        _faces = faces;
        _gallery = gallery;
        _plates = plates;
        _log = log;
    }

    public void Seed()
    {
        try
        {
            if (!_datasets.List().Any())
                SeedDatasets();
            if (!_gallery.List().Any())
                SeedGallery();
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "示範資料初始化失敗");
        }
    }

    private void SeedDatasets()
    {
        var faces = _datasets.Create("示範人臉", "detect", new[] { "face" }, "yolo26n");
        var plates = _datasets.Create("示範車牌", "detect", new[] { "plate" }, "yolo26n-obb");
        var mixed = _datasets.Create("人臉與車牌混合", "detect", new[] { "face", "plate" }, "yolo26s");

        ImportFolder(faces.Id, Path.Combine(_paths.SamplesRoot, "faces"), autoFace: true);
        ImportFolder(plates.Id, Path.Combine(_paths.SamplesRoot, "plates"), autoPlate: true);
        ImportFolder(mixed.Id, Path.Combine(_paths.SamplesRoot, "mixed"), autoFace: true, autoPlate: true);
    }

    private void ImportFolder(string datasetId, string folder, bool autoFace = false, bool autoPlate = false)
    {
        if (!Directory.Exists(folder)) return;
        var ds = _datasets.Get(datasetId);
        if (ds == null) return;
        foreach (var file in Directory.GetFiles(folder, "*.jpg"))
        {
            using var fs = File.OpenRead(file);
            var img = _datasets.AddImage(datasetId, Path.GetFileName(file), fs);
            using var mat = Cv2.ImRead(Path.Combine(_datasets.Root(datasetId), "images", img.FileName));
            var boxes = new List<AnnotationBox>();
            if (autoFace && _faces.DetectorReady)
            {
                foreach (var hit in _faces.Detect(mat, 0.55f))
                {
                    var cls = ds.Classes.FindIndex(c => c == "face");
                    if (cls < 0) cls = 0;
                    boxes.Add(new AnnotationBox { ClassId = cls, X = hit.Box.X, Y = hit.Box.Y, Width = hit.Box.Width, Height = hit.Box.Height });
                }
            }
            if (autoPlate)
            {
                foreach (var (box, _, _) in _plates.Detect(mat, readText: false))
                {
                    var cls = ds.Classes.FindIndex(c => c == "plate");
                    if (cls < 0) cls = Math.Max(0, ds.Classes.Count - 1);
                    boxes.Add(new AnnotationBox { ClassId = cls, X = box.X, Y = box.Y, Width = box.Width, Height = box.Height });
                }
            }
            if (boxes.Count > 0)
                _datasets.SaveBoxes(datasetId, img.FileName, boxes);
        }
    }

    private void SeedGallery()
    {
        var names = new (string File, string Name, string Note)[]
        {
            ("person_1.jpg", "林佳穎", "示範身份 A"),
            ("person_5.jpg", "陳志明", "示範身份 B"),
            ("person_12.jpg", "黃淑芬", "示範身份 C"),
            ("person_32.jpg", "吳建宏", "示範身份 D"),
            ("person_47.jpg", "張雅婷", "示範身份 E"),
        };
        var dir = Path.Combine(_paths.SamplesRoot, "faces");
        foreach (var (file, name, note) in names)
        {
            var path = Path.Combine(dir, file);
            if (!File.Exists(path) || !_faces.DetectorReady) continue;
            using var mat = Cv2.ImRead(path);
            var faces = _faces.Detect(mat, 0.5f);
            if (faces.Count == 0) continue;
            var hit = faces[0];
            var emb = _faces.Embed(mat, hit);
            if (emb == null) continue;
            using var thumb = new Mat(mat, hit.Box);
            _gallery.Add(name, note, thumb, emb);
        }
    }
}
