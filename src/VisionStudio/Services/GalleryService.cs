using OpenCvSharp;
using VisionStudio.Domain;

namespace VisionStudio.Services;

public sealed class GalleryService
{
    private readonly StoragePaths _paths;
    private readonly object _gate = new();
    private List<FaceIdentity> _faces;

    public GalleryService(StoragePaths paths)
    {
        _paths = paths;
        _faces = JsonStore.Load(IndexPath, new List<FaceIdentity>());
    }

    private string IndexPath => Path.Combine(_paths.GalleryRoot, "faces.json");

    public IReadOnlyList<FaceIdentity> List() => _faces.ToList();

    public FaceIdentity Add(string name, string note, Mat thumbnailBgr, float[] embedding)
    {
        var id = new FaceIdentity { Name = name, Note = note };
        if (!thumbnailBgr.Empty())
        {
            var file = Path.Combine(_paths.GalleryRoot, $"{id.Id}.jpg");
            Cv2.ImWrite(file, thumbnailBgr);
            id.Thumbnail = $"/media/gallery/{id.Id}.jpg";
        }
        id.Embeddings.Add(embedding);
        lock (_gate)
        {
            _faces.Add(id);
            JsonStore.Save(IndexPath, _faces);
        }
        return id;
    }

    public void Remove(string id)
    {
        lock (_gate)
        {
            _faces = _faces.Where(f => f.Id != id).ToList();
            JsonStore.Save(IndexPath, _faces);
        }
        var file = Path.Combine(_paths.GalleryRoot, $"{id}.jpg");
        if (File.Exists(file)) File.Delete(file);
    }

    public (FaceIdentity? Identity, float Score) Match(float[] embedding, float threshold)
    {
        FaceIdentity? best = null;
        var bestScore = threshold;
        foreach (var face in _faces)
        {
            foreach (var vec in face.Embeddings)
            {
                var s = FaceEngine.Cosine(embedding, vec);
                if (s > bestScore)
                {
                    bestScore = s;
                    best = face;
                }
            }
        }
        return (best, best == null ? 0 : bestScore);
    }
}

public sealed class PlateLogService
{
    private readonly StoragePaths _paths;
    private readonly object _gate = new();
    private List<PlateRecord> _items;
    private List<HistoryItem> _history;

    public PlateLogService(StoragePaths paths)
    {
        _paths = paths;
        _items = JsonStore.Load(Path.Combine(paths.HistoryRoot, "plates.json"), new List<PlateRecord>());
        _history = JsonStore.Load(Path.Combine(paths.HistoryRoot, "infer.json"), new List<HistoryItem>());
    }

    public IReadOnlyList<PlateRecord> Plates() => _items.TakeLast(200).Reverse().ToList();
    public IReadOnlyList<HistoryItem> History() => _history.TakeLast(50).Reverse().ToList();

    public void AddPlate(string text, float conf, string source, Mat? thumb)
    {
        if (string.IsNullOrWhiteSpace(text)) return;
        var rec = new PlateRecord { Text = text, Confidence = conf, Source = source };
        if (thumb != null && !thumb.Empty())
        {
            var file = Path.Combine(_paths.HistoryRoot, $"plate_{rec.Id}.jpg");
            Cv2.ImWrite(file, thumb);
            rec.Thumbnail = $"/media/history/plate_{rec.Id}.jpg";
        }
        lock (_gate)
        {
            _items.Add(rec);
            if (_items.Count > 500) _items = _items.TakeLast(400).ToList();
            JsonStore.Save(Path.Combine(_paths.HistoryRoot, "plates.json"), _items);
        }
    }

    public void AddHistory(HistoryItem item)
    {
        lock (_gate)
        {
            _history.Add(item);
            if (_history.Count > 80) _history = _history.TakeLast(60).ToList();
            JsonStore.Save(Path.Combine(_paths.HistoryRoot, "infer.json"), _history);
        }
    }
}
