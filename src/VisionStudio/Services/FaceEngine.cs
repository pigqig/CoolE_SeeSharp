using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace VisionStudio.Services;

public sealed class FaceHit
{
    public Rect Box { get; init; }
    public float Score { get; init; }
    public Point2f[] Landmarks { get; init; } = Array.Empty<Point2f>();
}

/// <summary>
/// 人臉偵測（YuNet）與身份特徵（SFace ONNX）。YOLO 只負責找位置，辨識身份必須走特徵向量。
/// </summary>
public sealed class FaceEngine : IDisposable
{
    private readonly object _gate = new();
    private readonly Size _netSize = new(320, 320);
    private FaceDetectorYN? _detector;
    private InferenceSession? _sface;
    public string DetectorPath { get; }
    public string RecognizerPath { get; }
    public bool DetectorReady => _detector != null;
    public bool RecognizerReady => _sface != null;

    private static readonly Point2f[] AlignDst =
    {
        new(38.2946f, 51.6963f),
        new(73.5318f, 51.5014f),
        new(56.0252f, 71.7366f)
    };

    public FaceEngine(StoragePaths paths)
    {
        DetectorPath = Path.Combine(paths.ModelsRoot, "face_detection_yunet_2023mar.onnx");
        RecognizerPath = Path.Combine(paths.ModelsRoot, "face_recognition_sface_2021dec.onnx");

        if (File.Exists(DetectorPath))
        {
            try
            {
                _detector = FaceDetectorYN.Create(DetectorPath, "", _netSize, 0.6f, 0.3f, 5000, Backend.DEFAULT, Target.CPU);
            }
            catch (Exception)
            {
                _detector = null;
            }
        }

        if (File.Exists(RecognizerPath))
        {
            var so = new Microsoft.ML.OnnxRuntime.SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            _sface = new InferenceSession(RecognizerPath, so);
        }
    }

    public List<FaceHit> Detect(Mat bgr, float scoreThreshold = 0.7f)
    {
        var list = new List<FaceHit>();
        if (_detector == null || bgr.Empty())
            return list;

        var scaleX = bgr.Width / (float)_netSize.Width;
        var scaleY = bgr.Height / (float)_netSize.Height;
        using var resized = new Mat();
        Cv2.Resize(bgr, resized, _netSize);

        lock (_gate)
        {
            using var faces = new Mat();
            _detector.Detect(resized, faces);
            if (faces.Empty() || faces.Rows <= 0)
                return list;

            for (var i = 0; i < faces.Rows; i++)
            {
                var score = faces.At<float>(i, 14);
                if (score < scoreThreshold) continue;
                var x = faces.At<float>(i, 0) * scaleX;
                var y = faces.At<float>(i, 1) * scaleY;
                var w = faces.At<float>(i, 2) * scaleX;
                var h = faces.At<float>(i, 3) * scaleY;
                var rect = ClampRect(x, y, w, h, bgr.Width, bgr.Height);
                if (rect.Width < 12 || rect.Height < 12) continue;
                var lm = new Point2f[5];
                for (var k = 0; k < 5; k++)
                {
                    lm[k] = new Point2f(
                        faces.At<float>(i, 4 + k * 2) * scaleX,
                        faces.At<float>(i, 5 + k * 2) * scaleY);
                }
                list.Add(new FaceHit { Box = rect, Score = score, Landmarks = lm });
            }
        }

        return list;
    }

    public float[]? Embed(Mat bgr, FaceHit face)
    {
        if (_sface == null || bgr.Empty())
            return null;

        using var aligned = Align(bgr, face);
        var tensor = ToTensor(aligned);
        lock (_gate)
        {
            using var results = _sface.Run(new[] { NamedOnnxValue.CreateFromTensor("data", tensor) });
            var output = results.First().AsEnumerable<float>().ToArray();
            return Normalize(output);
        }
    }

    public static float Cosine(float[] a, float[] b)
    {
        if (a.Length == 0 || b.Length == 0 || a.Length != b.Length)
            return 0;
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        var denom = Math.Sqrt(na) * Math.Sqrt(nb);
        return denom <= 1e-9 ? 0 : (float)(dot / denom);
    }

    private static Mat Align(Mat bgr, FaceHit face)
    {
        var aligned = new Mat();
        if (face.Landmarks.Length >= 3)
        {
            var src = new[] { face.Landmarks[0], face.Landmarks[1], face.Landmarks[2] };
            using var m = Cv2.GetAffineTransform(src, AlignDst);
            Cv2.WarpAffine(bgr, aligned, m, new Size(112, 112), InterpolationFlags.Linear, BorderTypes.Constant, new Scalar(0, 0, 0));
            return aligned;
        }

        var r = face.Box;
        using var crop = new Mat(bgr, r);
        Cv2.Resize(crop, aligned, new Size(112, 112));
        return aligned;
    }

    private static DenseTensor<float> ToTensor(Mat bgr112)
    {
        using var rgb = new Mat();
        Cv2.CvtColor(bgr112, rgb, ColorConversionCodes.BGR2RGB);
        var tensor = new DenseTensor<float>(new[] { 1, 3, 112, 112 });
        for (var y = 0; y < 112; y++)
        {
            for (var x = 0; x < 112; x++)
            {
                var px = rgb.At<Vec3b>(y, x);
                tensor[0, 0, y, x] = px.Item0;
                tensor[0, 1, y, x] = px.Item1;
                tensor[0, 2, y, x] = px.Item2;
            }
        }
        return tensor;
    }

    private static float[] Normalize(float[] v)
    {
        double n = 0;
        foreach (var x in v) n += x * x;
        n = Math.Sqrt(n);
        if (n < 1e-9) return v;
        for (var i = 0; i < v.Length; i++) v[i] = (float)(v[i] / n);
        return v;
    }

    private static Rect ClampRect(float x, float y, float w, float h, int maxW, int maxH)
    {
        var ix = (int)Math.Floor(x);
        var iy = (int)Math.Floor(y);
        var iw = (int)Math.Ceiling(w);
        var ih = (int)Math.Ceiling(h);
        if (ix < 0) { iw += ix; ix = 0; }
        if (iy < 0) { ih += iy; iy = 0; }
        if (ix + iw > maxW) iw = maxW - ix;
        if (iy + ih > maxH) ih = maxH - iy;
        return new Rect(ix, iy, Math.Max(1, iw), Math.Max(1, ih));
    }

    public void Dispose()
    {
        _detector?.Dispose();
        _sface?.Dispose();
    }
}
