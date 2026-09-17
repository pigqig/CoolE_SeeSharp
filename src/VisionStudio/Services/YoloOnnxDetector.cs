using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using VisionStudio.Domain;

namespace VisionStudio.Services;

/// <summary>
/// 讀取 Ultralytics 匯出的 YOLO ONNX（YOLOv8/11 的 1x(4+nc)x8400，以及 YOLO26 NMS-free 的 1x300x6）。
/// </summary>
public sealed class YoloOnnxDetector : IDisposable
{
    private InferenceSession? _session;
    private readonly object _gate = new();
    public string? ModelPath { get; private set; }
    public string Family { get; private set; } = "";
    public string[] Classes { get; private set; } = Array.Empty<string>();
    public bool Ready => _session != null;

    public void Load(string path, IEnumerable<string>? classes = null, string family = "yolo11")
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("找不到 ONNX 模型", path);

        lock (_gate)
        {
            _session?.Dispose();
            var so = new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2)
            };
            _session = new InferenceSession(path, so);
            ModelPath = path;
            Family = family;
            Classes = classes?.ToArray() ?? GuessClasses(path);
        }
    }

    public void Unload()
    {
        lock (_gate)
        {
            _session?.Dispose();
            _session = null;
            ModelPath = null;
        }
    }

    public List<Detection> Detect(Mat bgr, float confThreshold = 0.25f, float iouThreshold = 0.45f)
    {
        if (_session == null || bgr.Empty())
            return new();

        const int size = 640;
        var (tensor, scale, padX, padY) = Letterbox(bgr, size);
        var inputName = _session.InputMetadata.Keys.First();

        lock (_gate)
        {
            using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) });
            var output = results.First().AsTensor<float>();
            return Parse(output, bgr.Width, bgr.Height, scale, padX, padY, size, confThreshold, iouThreshold);
        }
    }

    private List<Detection> Parse(Tensor<float> output, int origW, int origH, float scale, float padX, float padY, int size, float confTh, float iouTh)
    {
        var dims = output.Dimensions.ToArray();
        var boxes = new List<Detection>();

        // YOLO26 e2e: [1, 300, 6] => x1,y1,x2,y2,conf,cls
        if (dims.Length == 3 && dims[2] == 6)
        {
            var n = dims[1];
            for (var i = 0; i < n; i++)
            {
                var x1 = output[0, i, 0];
                var y1 = output[0, i, 1];
                var x2 = output[0, i, 2];
                var y2 = output[0, i, 3];
                var conf = output[0, i, 4];
                var cls = (int)output[0, i, 5];
                if (conf < confTh) continue;
                boxes.Add(ToDetection(x1, y1, x2, y2, conf, cls, origW, origH, scale, padX, padY));
            }
            return boxes.Where(b => b.Box.Width > 1 && b.Box.Height > 1).ToList();
        }

        // YOLOv8/11: [1, 4+nc, 8400] 或 [1, 8400, 4+nc]
        int channels, preds;
        bool transposed;
        if (dims.Length == 3)
        {
            if (dims[1] < dims[2])
            {
                channels = dims[1];
                preds = dims[2];
                transposed = false;
            }
            else
            {
                preds = dims[1];
                channels = dims[2];
                transposed = true;
            }
        }
        else
        {
            return boxes;
        }

        var nc = Math.Max(1, channels - 4);
        for (var i = 0; i < preds; i++)
        {
            float cx, cy, w, h;
            var best = 0;
            var bestScore = 0f;
            if (!transposed)
            {
                cx = output[0, 0, i];
                cy = output[0, 1, i];
                w = output[0, 2, i];
                h = output[0, 3, i];
                for (var c = 0; c < nc; c++)
                {
                    var s = output[0, 4 + c, i];
                    if (s > bestScore) { bestScore = s; best = c; }
                }
            }
            else
            {
                cx = output[0, i, 0];
                cy = output[0, i, 1];
                w = output[0, i, 2];
                h = output[0, i, 3];
                for (var c = 0; c < nc; c++)
                {
                    var s = output[0, i, 4 + c];
                    if (s > bestScore) { bestScore = s; best = c; }
                }
            }
            if (bestScore < confTh) continue;
            var x1 = cx - w / 2;
            var y1 = cy - h / 2;
            var x2 = cx + w / 2;
            var y2 = cy + h / 2;
            boxes.Add(ToDetection(x1, y1, x2, y2, bestScore, best, origW, origH, scale, padX, padY));
        }

        return Nms(boxes, iouTh);
    }

    private Detection ToDetection(float x1, float y1, float x2, float y2, float conf, int cls, int origW, int origH, float scale, float padX, float padY)
    {
        var rx1 = (x1 - padX) / scale;
        var ry1 = (y1 - padY) / scale;
        var rx2 = (x2 - padX) / scale;
        var ry2 = (y2 - padY) / scale;
        rx1 = Math.Clamp(rx1, 0, origW);
        ry1 = Math.Clamp(ry1, 0, origH);
        rx2 = Math.Clamp(rx2, 0, origW);
        ry2 = Math.Clamp(ry2, 0, origH);
        var name = cls >= 0 && cls < Classes.Length ? Classes[cls] : $"class_{cls}";
        var kind = name.Contains("plate", StringComparison.OrdinalIgnoreCase) ? DetectionKind.Plate
            : name.Contains("face", StringComparison.OrdinalIgnoreCase) || name.Contains("person", StringComparison.OrdinalIgnoreCase)
                ? DetectionKind.Face
                : DetectionKind.Custom;
        return new Detection
        {
            Kind = kind,
            Label = name,
            Confidence = conf,
            Engine = $"YOLO ONNX ({Family})",
            Box = new BoundingBox
            {
                X = rx1,
                Y = ry1,
                Width = Math.Max(1, rx2 - rx1),
                Height = Math.Max(1, ry2 - ry1)
            }
        };
    }

    private static List<Detection> Nms(List<Detection> boxes, float iouTh)
    {
        var ordered = boxes.OrderByDescending(b => b.Confidence).ToList();
        var kept = new List<Detection>();
        var suppressed = new bool[ordered.Count];
        for (var i = 0; i < ordered.Count; i++)
        {
            if (suppressed[i]) continue;
            kept.Add(ordered[i]);
            for (var j = i + 1; j < ordered.Count; j++)
            {
                if (suppressed[j]) continue;
                if (ordered[i].Label != ordered[j].Label) continue;
                if (Iou(ordered[i].Box, ordered[j].Box) > iouTh)
                    suppressed[j] = true;
            }
        }
        return kept;
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

    private static (DenseTensor<float> Tensor, float Scale, float PadX, float PadY) Letterbox(Mat bgr, int size)
    {
        var scale = Math.Min(size / (float)bgr.Width, size / (float)bgr.Height);
        var nw = (int)Math.Round(bgr.Width * scale);
        var nh = (int)Math.Round(bgr.Height * scale);
        var padX = (size - nw) / 2f;
        var padY = (size - nh) / 2f;
        using var resized = new Mat();
        Cv2.Resize(bgr, resized, new Size(nw, nh));
        using var canvas = new Mat(new Size(size, size), MatType.CV_8UC3, new Scalar(114, 114, 114));
        resized.CopyTo(new Mat(canvas, new Rect((int)padX, (int)padY, nw, nh)));
        using var rgb = new Mat();
        Cv2.CvtColor(canvas, rgb, ColorConversionCodes.BGR2RGB);
        var tensor = new DenseTensor<float>(new[] { 1, 3, size, size });
        for (var y = 0; y < size; y++)
        {
            for (var x = 0; x < size; x++)
            {
                var px = rgb.At<Vec3b>(y, x);
                tensor[0, 0, y, x] = px.Item0 / 255f;
                tensor[0, 1, y, x] = px.Item1 / 255f;
                tensor[0, 2, y, x] = px.Item2 / 255f;
            }
        }
        return (tensor, scale, padX, padY);
    }

    private static string[] GuessClasses(string path)
    {
        var sidecar = Path.ChangeExtension(path, ".txt");
        if (File.Exists(sidecar))
            return File.ReadAllLines(sidecar).Select(l => l.Trim()).Where(l => l.Length > 0).ToArray();
        var yaml = Path.Combine(Path.GetDirectoryName(path) ?? "", "classes.txt");
        if (File.Exists(yaml))
            return File.ReadAllLines(yaml).Select(l => l.Trim()).Where(l => l.Length > 0).ToArray();
        return new[] { "face", "plate" };
    }

    public void Dispose() => _session?.Dispose();
}
