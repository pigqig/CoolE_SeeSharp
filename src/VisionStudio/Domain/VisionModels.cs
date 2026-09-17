namespace VisionStudio.Domain;

public enum DetectionKind
{
    Face,
    Plate,
    Custom
}

public sealed class BoundingBox
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Width { get; set; }
    public float Height { get; set; }

    public int Xi => (int)Math.Round(X);
    public int Yi => (int)Math.Round(Y);
    public int Wi => (int)Math.Round(Width);
    public int Hi => (int)Math.Round(Height);
}

public sealed class Detection
{
    public DetectionKind Kind { get; set; }
    public string Label { get; set; } = "";
    public string? Identity { get; set; }
    public string? PlateText { get; set; }
    public float Confidence { get; set; }
    public BoundingBox Box { get; set; } = new();
    public int? TrackId { get; set; }
    public string Engine { get; set; } = "";
}

public sealed class InferRequest
{
    public bool DetectFaces { get; set; } = true;
    public bool DetectPlates { get; set; } = true;
    public bool IdentifyFaces { get; set; } = true;
    public bool ReadPlates { get; set; } = true;
    public bool PreferYolo { get; set; } = true;
    public float FaceThreshold { get; set; } = 0.7f;
    public float PlateThreshold { get; set; } = 0.35f;
    public float IdentityThreshold { get; set; } = 0.45f;
}

public sealed class InferResult
{
    public int Width { get; set; }
    public int Height { get; set; }
    public long ElapsedMs { get; set; }
    public string OverlayJpegBase64 { get; set; } = "";
    public List<Detection> Detections { get; set; } = new();
    public List<string> Notes { get; set; } = new();
    public string EngineSummary { get; set; } = "";
}

public sealed class DatasetInfo
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Task { get; set; } = "detect";
    public List<string> Classes { get; set; } = new() { "face", "plate" };
    public string YoloFamily { get; set; } = "yolo26n";
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
    public int ImageCount { get; set; }
    public int LabeledCount { get; set; }
}

public sealed class DatasetImage
{
    public string FileName { get; set; } = "";
    public int Width { get; set; }
    public int Height { get; set; }
    public List<AnnotationBox> Boxes { get; set; } = new();
}

public sealed class AnnotationBox
{
    public int ClassId { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float Width { get; set; }
    public float Height { get; set; }
}

public sealed class TrainingJob
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..12];
    public string DatasetId { get; set; } = "";
    public string ModelFamily { get; set; } = "yolo26n";
    public int Epochs { get; set; } = 50;
    public int ImageSize { get; set; } = 640;
    public int Batch { get; set; } = 8;
    public string Status { get; set; } = "queued";
    public string? Message { get; set; }
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
    public DateTime? FinishedUtc { get; set; }
    public List<string> Logs { get; set; } = new();
    public string? BestWeights { get; set; }
    public string? ExportedOnnx { get; set; }
}

public sealed class FaceIdentity
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..10];
    public string Name { get; set; } = "";
    public string Note { get; set; } = "";
    public string Thumbnail { get; set; } = "";
    public List<float[]> Embeddings { get; set; } = new();
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
}

public sealed class PlateRecord
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..10];
    public string Text { get; set; } = "";
    public float Confidence { get; set; }
    public DateTime SeenUtc { get; set; } = DateTime.UtcNow;
    public string Source { get; set; } = "";
    public string Thumbnail { get; set; } = "";
}

public sealed class HistoryItem
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..10];
    public DateTime AtUtc { get; set; } = DateTime.UtcNow;
    public string Source { get; set; } = "";
    public int FaceCount { get; set; }
    public int PlateCount { get; set; }
    public string Summary { get; set; } = "";
    public string Overlay { get; set; } = "";
}

public sealed class RegisteredModel
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Path { get; set; } = "";
    public string Task { get; set; } = "detect";
    public string Family { get; set; } = "yolo11";
    public List<string> Classes { get; set; } = new();
    public bool Active { get; set; }
}
