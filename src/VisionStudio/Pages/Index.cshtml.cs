using Microsoft.AspNetCore.Mvc.RazorPages;
using VisionStudio.Domain;
using VisionStudio.Services;

namespace VisionStudio.Pages;

public class IndexModel : PageModel
{
    private readonly GalleryService _gallery;
    private readonly DatasetService _datasets;
    private readonly TrainingService _train;
    private readonly PlateLogService _log;
    private readonly FaceEngine _faces;
    private readonly YoloOnnxDetector _yolo;

    public IndexModel(GalleryService gallery, DatasetService datasets, TrainingService train, PlateLogService log, FaceEngine faces, YoloOnnxDetector yolo)
    {
        _gallery = gallery;
        _datasets = datasets;
        _train = train;
        _log = log;
        _faces = faces;
        _yolo = yolo;
    }

    public int FaceCount { get; private set; }
    public int DatasetCount { get; private set; }
    public int JobCount { get; private set; }
    public int PlateCount { get; private set; }
    public bool FaceDetector { get; private set; }
    public bool FaceRecognizer { get; private set; }
    public bool YoloReady { get; private set; }
    public string? YoloPath { get; private set; }
    public IReadOnlyList<HistoryItem> History { get; private set; } = Array.Empty<HistoryItem>();

    public void OnGet()
    {
        FaceCount = _gallery.List().Count;
        DatasetCount = _datasets.List().Count;
        JobCount = _train.List().Count;
        PlateCount = _log.Plates().Count;
        FaceDetector = _faces.DetectorReady;
        FaceRecognizer = _faces.RecognizerReady;
        YoloReady = _yolo.Ready;
        YoloPath = _yolo.ModelPath;
        History = _log.History();
    }
}
