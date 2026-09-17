using System.Diagnostics;
using OpenCvSharp;
using VisionStudio.Domain;

namespace VisionStudio.Services;

public sealed class ModelRegistry
{
    private readonly StoragePaths _paths;
    private readonly YoloOnnxDetector _yolo;
    private List<RegisteredModel> _models;

    public ModelRegistry(StoragePaths paths, YoloOnnxDetector yolo)
    {
        _paths = paths;
        _yolo = yolo;
        _models = JsonStore.Load(IndexPath, new List<RegisteredModel>());
        EnsureBuiltins();
        TryActivate();
    }

    private string IndexPath => Path.Combine(_paths.DataRoot, "models", "registry.json");

    public IReadOnlyList<RegisteredModel> List() => _models.ToList();
    public YoloOnnxDetector Yolo => _yolo;

    public RegisteredModel Register(string name, string path, string family, IEnumerable<string> classes, string task = "detect")
    {
        var m = new RegisteredModel
        {
            Id = Guid.NewGuid().ToString("N")[..8],
            Name = name,
            Path = path,
            Family = family,
            Task = task,
            Classes = classes.ToList(),
            Active = false
        };
        _models.Add(m);
        JsonStore.Save(IndexPath, _models);
        return m;
    }

    public void Activate(string id)
    {
        foreach (var m in _models) m.Active = m.Id == id;
        JsonStore.Save(IndexPath, _models);
        var active = _models.First(m => m.Id == id);
        if (File.Exists(active.Path))
            _yolo.Load(active.Path, active.Classes, active.Family);
        else
            throw new FileNotFoundException("模型檔不存在", active.Path);
    }

    public void Deactivate()
    {
        foreach (var m in _models) m.Active = false;
        JsonStore.Save(IndexPath, _models);
        _yolo.Unload();
    }

    private void TryActivate()
    {
        var active = _models.FirstOrDefault(m => m.Active && File.Exists(m.Path));
        if (active != null)
        {
            try { _yolo.Load(active.Path, active.Classes, active.Family); }
            catch { /* 啟動時略過損毀模型 */ }
        }
    }

    private void EnsureBuiltins()
    {
        var yunet = Path.Combine(_paths.ModelsRoot, "face_detection_yunet_2023mar.onnx");
        if (File.Exists(yunet) && _models.All(m => m.Path != yunet))
        {
            _models.Add(new RegisteredModel
            {
                Id = "yunet",
                Name = "YuNet 人臉偵測（內建）",
                Path = yunet,
                Family = "yunet",
                Task = "face-detect",
                Classes = new() { "face" },
                Active = false
            });
        }
        var sface = Path.Combine(_paths.ModelsRoot, "face_recognition_sface_2021dec.onnx");
        if (File.Exists(sface) && _models.All(m => m.Path != sface))
        {
            _models.Add(new RegisteredModel
            {
                Id = "sface",
                Name = "SFace 人臉特徵（內建）",
                Path = sface,
                Family = "sface",
                Task = "face-embed",
                Classes = new() { "identity" }
            });
        }
        JsonStore.Save(IndexPath, _models);
    }
}

public sealed class VisionPipeline
{
    private readonly FaceEngine _faces;
    private readonly PlateEngine _plates;
    private readonly GalleryService _gallery;
    private readonly PlateLogService _log;
    private readonly YoloOnnxDetector _yolo;
    private readonly IoUTracker _tracker = new();

    public VisionPipeline(FaceEngine faces, PlateEngine plates, GalleryService gallery, PlateLogService log, YoloOnnxDetector yolo)
    {
        _faces = faces;
        _plates = plates;
        _gallery = gallery;
        _log = log;
        _yolo = yolo;
    }

    public InferResult Run(Mat bgr, InferRequest req, string source = "image", bool track = false, bool record = true)
    {
        var sw = Stopwatch.StartNew();
        var detections = new List<Detection>();
        var notes = new List<string>();

        if (req.PreferYolo && _yolo.Ready)
        {
            var yoloDets = _yolo.Detect(bgr, req.PlateThreshold);
            detections.AddRange(yoloDets);
            notes.Add($"YOLO ONNX：{Path.GetFileName(_yolo.ModelPath)}（{_yolo.Family}）");
        }

        if (req.DetectFaces)
        {
            var hasYoloFace = detections.Any(d => d.Kind == DetectionKind.Face);
            if (!hasYoloFace)
            {
                if (!_faces.DetectorReady)
                    notes.Add("人臉偵測模型未就緒。");
                else
                {
                    var faces = _faces.Detect(bgr, req.FaceThreshold);
                    foreach (var hit in faces)
                    {
                        string? identity = null;
                        if (req.IdentifyFaces && _faces.RecognizerReady)
                        {
                            var emb = _faces.Embed(bgr, hit);
                            if (emb != null)
                            {
                                var match = _gallery.Match(emb, req.IdentityThreshold);
                                if (match.Identity != null)
                                    identity = $"{match.Identity.Name} {match.Score:0.00}";
                            }
                        }
                        detections.Add(new Detection
                        {
                            Kind = DetectionKind.Face,
                            Label = identity ?? "人臉",
                            Identity = identity,
                            Confidence = hit.Score,
                            Engine = "YuNet + SFace",
                            Box = new BoundingBox { X = hit.Box.X, Y = hit.Box.Y, Width = hit.Box.Width, Height = hit.Box.Height }
                        });
                    }
                    notes.Add($"YuNet 偵測到 {faces.Count} 張人臉");
                }
            }
            else if (req.IdentifyFaces)
            {
                notes.Add("YOLO 人臉框沒有五官點，身份比對請用 YuNet，或之後改成含 landmark 的模型。");
            }
        }

        if (req.DetectPlates)
        {
            var hasYoloPlate = detections.Any(d => d.Kind == DetectionKind.Plate);
            if (!hasYoloPlate)
            {
                var plates = _plates.Detect(bgr, req.ReadPlates);
                foreach (var (box, text, score) in plates)
                {
                    detections.Add(new Detection
                    {
                        Kind = DetectionKind.Plate,
                        Label = string.IsNullOrWhiteSpace(text) ? "車牌" : text,
                        PlateText = text,
                        Confidence = score,
                        Engine = "色彩輪廓 + 字元辨識",
                        Box = new BoundingBox { X = box.X, Y = box.Y, Width = box.Width, Height = box.Height }
                    });
                    if (record && !string.IsNullOrWhiteSpace(text))
                    {
                        using var crop = new Mat(bgr, box);
                        _log.AddPlate(text, score, source, crop);
                    }
                }
                notes.Add($"車牌候選 {plates.Count} 處");
            }
            else if (req.ReadPlates)
            {
                foreach (var det in detections.Where(d => d.Kind == DetectionKind.Plate))
                {
                    var rect = ToRect(det.Box, bgr.Size());
                    using var crop = new Mat(bgr, rect);
                    var (text, score) = _plates.Read(crop);
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        det.PlateText = text;
                        det.Label = text;
                        det.Confidence = Math.Max(det.Confidence, score);
                        if (record) _log.AddPlate(text, score, source, crop);
                    }
                }
            }
        }

        if (track)
            _tracker.Update(detections);

        using var overlay = bgr.Clone();
        Draw(overlay, detections);
        var jpeg = overlay.ToBytes(".jpg", new ImageEncodingParam(ImwriteFlags.JpegQuality, 85));
        sw.Stop();

        var result = new InferResult
        {
            Width = bgr.Width,
            Height = bgr.Height,
            ElapsedMs = sw.ElapsedMilliseconds,
            OverlayJpegBase64 = Convert.ToBase64String(jpeg),
            Detections = detections,
            Notes = notes,
            EngineSummary = EngineSummary(req)
        };

        if (record)
        {
            _log.AddHistory(new HistoryItem
            {
                Source = source,
                FaceCount = detections.Count(d => d.Kind == DetectionKind.Face),
                PlateCount = detections.Count(d => d.Kind == DetectionKind.Plate),
                Summary = string.Join("、", detections.Select(d => d.Label).Distinct().Take(6))
            });
        }

        return result;
    }

    public InferResult RunBytes(byte[] imageBytes, InferRequest req, string source = "image", bool track = false, bool record = true)
    {
        using var mat = Cv2.ImDecode(imageBytes, ImreadModes.Color);
        if (mat.Empty())
            throw new InvalidOperationException("無法解讀影像");
        return Run(mat, req, source, track, record);
    }

    private string EngineSummary(InferRequest req)
    {
        var parts = new List<string>();
        if (req.PreferYolo && _yolo.Ready) parts.Add($"YOLO {_yolo.Family}");
        if (_faces.DetectorReady) parts.Add("YuNet");
        if (_faces.RecognizerReady) parts.Add("SFace");
        parts.Add("車牌定位器");
        return string.Join(" + ", parts);
    }

    private static void Draw(Mat img, List<Detection> dets)
    {
        foreach (var d in dets)
        {
            var color = d.Kind switch
            {
                DetectionKind.Face => new Scalar(191, 207, 62),
                DetectionKind.Plate => new Scalar(41, 180, 240),
                _ => new Scalar(220, 220, 220)
            };
            var rect = new Rect(d.Box.Xi, d.Box.Yi, d.Box.Wi, d.Box.Hi);
            Cv2.Rectangle(img, rect, color, 2);
            var caption = d.TrackId is int tid
                ? $"#{tid} {d.Label} {d.Confidence:0.00}"
                : $"{d.Label} {d.Confidence:0.00}";
            var origin = new Point(rect.X, Math.Max(18, rect.Y - 6));
            var size = Cv2.GetTextSize(caption, HersheyFonts.HersheySimplex, 0.55, 1, out _);
            Cv2.Rectangle(img, new Rect(origin.X, origin.Y - size.Height - 4, size.Width + 8, size.Height + 8), color, -1);
            Cv2.PutText(img, caption, new Point(origin.X + 4, origin.Y), HersheyFonts.HersheySimplex, 0.55, Scalar.Black, 1);
        }
    }

    private static Rect ToRect(BoundingBox box, Size size)
    {
        var x = Math.Clamp(box.Xi, 0, size.Width - 1);
        var y = Math.Clamp(box.Yi, 0, size.Height - 1);
        var w = Math.Clamp(box.Wi, 1, size.Width - x);
        var h = Math.Clamp(box.Hi, 1, size.Height - y);
        return new Rect(x, y, w, h);
    }
}

/// <summary>簡易 IoU 追蹤，供影片與串流維持同一物件編號。</summary>
public sealed class IoUTracker
{
    private readonly List<(int Id, Detection Det)> _tracks = new();
    private int _next = 1;

    public void Update(List<Detection> detections, float iouTh = 0.35f)
    {
        var used = new HashSet<int>();
        foreach (var det in detections.OrderByDescending(d => d.Confidence))
        {
            var bestI = -1;
            var best = iouTh;
            for (var i = 0; i < _tracks.Count; i++)
            {
                if (used.Contains(i)) continue;
                if (_tracks[i].Det.Kind != det.Kind) continue;
                var iou = Iou(det.Box, _tracks[i].Det.Box);
                if (iou > best) { best = iou; bestI = i; }
            }
            if (bestI >= 0)
            {
                det.TrackId = _tracks[bestI].Id;
                _tracks[bestI] = (det.TrackId.Value, det);
                used.Add(bestI);
            }
            else
            {
                det.TrackId = _next++;
                _tracks.Add((det.TrackId.Value, det));
                used.Add(_tracks.Count - 1);
            }
        }
        for (var i = _tracks.Count - 1; i >= 0; i--)
            if (!used.Contains(i)) _tracks.RemoveAt(i);
    }

    private static float Iou(BoundingBox a, BoundingBox b)
    {
        var x1 = Math.Max(a.X, b.X);
        var y1 = Math.Max(a.Y, b.Y);
        var x2 = Math.Min(a.X + a.Width, b.X + b.Width);
        var y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);
        var inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        var union = a.Width * a.Height + b.Width * b.Height - inter;
        return union <= 0 ? 0 : inter / union;
    }
}

public sealed class VideoProcessor
{
    private readonly VisionPipeline _pipeline;
    private readonly StoragePaths _paths;

    public VideoProcessor(VisionPipeline pipeline, StoragePaths paths)
    {
        _pipeline = pipeline;
        _paths = paths;
    }

    public async Task<(string OverlayDir, List<Detection> Events, int Frames)> ProcessFileAsync(
        string videoPath, InferRequest req, IProgress<string>? progress, CancellationToken ct)
    {
        var work = Path.Combine(_paths.UploadsRoot, "video-" + Guid.NewGuid().ToString("N")[..8]);
        var framesDir = Path.Combine(work, "frames");
        var overlayDir = Path.Combine(work, "overlay");
        Directory.CreateDirectory(framesDir);
        Directory.CreateDirectory(overlayDir);

        progress?.Report("正在擷取影片影格（每秒 4 張）…");
        var psi = new ProcessStartInfo
        {
            FileName = ToolPaths.Ffmpeg(),
            Arguments = $"-y -i \"{videoPath}\" -vf fps=4 \"{Path.Combine(framesDir, "f_%04d.jpg")}\"",
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false
        };
        using (var p = Process.Start(psi) ?? throw new InvalidOperationException("找不到 ffmpeg"))
        {
            await p.WaitForExitAsync(ct);
            if (p.ExitCode != 0)
                throw new InvalidOperationException("ffmpeg 擷取影格失敗");
        }

        var files = Directory.GetFiles(framesDir, "*.jpg").OrderBy(f => f).ToArray();
        var events = new List<Detection>();
        var i = 0;
        foreach (var file in files)
        {
            ct.ThrowIfCancellationRequested();
            i++;
            using var mat = Cv2.ImRead(file);
            var result = _pipeline.Run(mat, req, "video", track: true, record: i % 4 == 0);
            var jpg = Convert.FromBase64String(result.OverlayJpegBase64);
            await File.WriteAllBytesAsync(Path.Combine(overlayDir, Path.GetFileName(file)), jpg, ct);
            events.AddRange(result.Detections);
            if (i % 5 == 0)
                progress?.Report($"已處理 {i}/{files.Length} 張影格");
        }
        progress?.Report($"完成 {files.Length} 張影格");
        return (overlayDir, events, files.Length);
    }
}
