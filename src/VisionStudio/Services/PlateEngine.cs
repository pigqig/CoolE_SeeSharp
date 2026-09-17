using System.Text.RegularExpressions;
using OpenCvSharp;
using VisionStudio.Domain;

namespace VisionStudio.Services;

/// <summary>
/// 車牌定位（色彩＋輪廓，YOLO 模型就緒後可改走 OBB）與字元辨識（Tesseract 或樣板比對）。
/// </summary>
public sealed class PlateEngine
{
    private static readonly Regex TaiwanPlate = new(
        @"^[A-Z]{2,3}-?\d{3,4}$",
        RegexOptions.Compiled);

    private readonly Dictionary<char, Mat> _templates = new();

    public PlateEngine()
    {
        BuildTemplates();
    }

    public List<(Rect Box, string Text, float Score)> Detect(Mat bgr, bool readText = true)
    {
        var results = new List<(Rect, string, float)>();
        if (bgr.Empty())
            return results;

        using var hsv = new Mat();
        Cv2.CvtColor(bgr, hsv, ColorConversionCodes.BGR2HSV);

        using var mask = ColorMask(hsv);
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(17, 5));
        Cv2.MorphologyEx(mask, mask, MorphTypes.Close, kernel);
        Cv2.MorphologyEx(mask, mask, MorphTypes.Open, kernel);

        Cv2.FindContours(mask, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        var candidates = new List<Rect>();
        foreach (var c in contours)
        {
            var rect = Cv2.BoundingRect(c);
            var area = rect.Width * rect.Height;
            if (area < 800 || area > bgr.Width * bgr.Height * 0.45)
                continue;
            var ratio = rect.Width / (float)rect.Height;
            if (ratio < 1.8f || ratio > 6.2f)
                continue;
            if (rect.Height < 18)
                continue;
            candidates.Add(Expand(rect, bgr.Size(), 4));
        }

        // 後備：灰階邊緣找矩形，補抓白底車牌
        if (candidates.Count == 0)
            candidates.AddRange(EdgeCandidates(bgr));

        candidates = Merge(candidates);

        foreach (var rect in candidates.Take(8))
        {
            var text = "";
            var score = 0.45f;
            if (readText)
            {
                using var crop = new Mat(bgr, rect);
                (text, score) = Read(crop);
            }
            if (readText && string.IsNullOrWhiteSpace(text) && score < 0.35f)
                continue;
            results.Add((rect, text, score));
        }

        return results.OrderByDescending(r => r.Item3).ToList();
    }

    public (string Text, float Score) Read(Mat plateBgr)
    {
        if (plateBgr.Empty())
            return ("", 0);

        var tess = TryTesseract(plateBgr);
        if (!string.IsNullOrWhiteSpace(tess.Text))
            return tess;

        return TemplateRead(plateBgr);
    }

    public static string NormalizePlate(string raw)
    {
        var t = raw.ToUpperInvariant();
        t = Regex.Replace(t, @"[^A-Z0-9]", "");
        // 常見 OCR：把框線讀成開頭的 I／1
        if (t.Length >= 6 && (t[0] is 'I' or '1') && char.IsLetter(t[1]))
            t = t[1..];
        if (t.Length is >= 5 and <= 7)
        {
            var letters = new string(t.TakeWhile(char.IsLetter).ToArray());
            var digits = new string(t.Skip(letters.Length).ToArray());
            if (letters.Length is 2 or 3 && digits.Length is 3 or 4)
                return $"{letters}-{digits}";
        }
        return t;
    }

    public static bool LooksLikeTaiwanPlate(string text)
        => TaiwanPlate.IsMatch(text.Replace(" ", ""));

    private (string Text, float Score) TemplateRead(Mat plateBgr)
    {
        using var gray = new Mat();
        Cv2.CvtColor(plateBgr, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.Resize(gray, gray, new Size(Math.Max(240, gray.Width * 2), Math.Max(80, gray.Height * 2)));
        Cv2.GaussianBlur(gray, gray, new Size(3, 3), 0);
        using var bin = new Mat();
        Cv2.Threshold(gray, bin, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
        // 字深底淺；若白字則反相
        if (Cv2.Mean(bin).Val0 < 127)
            Cv2.BitwiseNot(bin, bin);

        var chars = SplitCharacters(bin);
        if (chars.Count < 5)
            return ("", 0.2f);

        var sb = new System.Text.StringBuilder();
        var scores = new List<float>();
        foreach (var ch in chars)
        {
            var (glyph, s) = MatchChar(ch);
            if (glyph == '\0')
                continue;
            sb.Append(glyph);
            scores.Add(s);
            ch.Dispose();
        }

        var raw = sb.ToString();
        if (raw.Length >= 5 && raw.Length <= 7 && !raw.Contains('-'))
        {
            var letters = new string(raw.TakeWhile(char.IsLetter).ToArray());
            if (letters.Length is 2 or 3)
                raw = letters + "-" + raw[letters.Length..];
        }

        var avg = scores.Count == 0 ? 0 : scores.Average();
        var text = NormalizePlate(raw);
        if (LooksLikeTaiwanPlate(text))
            avg = Math.Min(0.99f, avg + 0.12f);
        return (text, (float)avg);
    }

    private List<Mat> SplitCharacters(Mat bin)
    {
        var list = new List<Mat>();
        var col = new int[bin.Width];
        for (var x = 0; x < bin.Width; x++)
        {
            var ink = 0;
            for (var y = 0; y < bin.Height; y++)
            {
                if (bin.At<byte>(y, x) == 0) ink++;
            }
            col[x] = ink;
        }

        var minInk = Math.Max(2, bin.Height / 12);
        var inGlyph = false;
        var start = 0;
        for (var x = 0; x < col.Length; x++)
        {
            if (!inGlyph && col[x] > minInk)
            {
                inGlyph = true;
                start = x;
            }
            else if (inGlyph && col[x] <= minInk)
            {
                inGlyph = false;
                var w = x - start;
                if (w >= 6)
                    list.Add(CropGlyph(bin, start, w));
            }
        }
        if (inGlyph)
        {
            var w = col.Length - start;
            if (w >= 6)
                list.Add(CropGlyph(bin, start, w));
        }

        // 去掉分隔線（太窄）
        return list.Where(m => m.Width >= 8 && m.Width < bin.Width * 0.35).ToList();
    }

    private static Mat CropGlyph(Mat bin, int x, int w)
    {
        var roi = new Rect(Math.Max(0, x - 1), 0, Math.Min(bin.Width - x + 1, w + 2), bin.Height);
        using var slice = new Mat(bin, roi);
        var rowStart = 0;
        var rowEnd = slice.Height - 1;
        for (var y = 0; y < slice.Height; y++)
        {
            var ink = 0;
            for (var xx = 0; xx < slice.Width; xx++)
                if (slice.At<byte>(y, xx) == 0) ink++;
            if (ink > 1) { rowStart = y; break; }
        }
        for (var y = slice.Height - 1; y >= 0; y--)
        {
            var ink = 0;
            for (var xx = 0; xx < slice.Width; xx++)
                if (slice.At<byte>(y, xx) == 0) ink++;
            if (ink > 1) { rowEnd = y; break; }
        }
        var h = Math.Max(8, rowEnd - rowStart + 1);
        var rect = new Rect(0, rowStart, slice.Width, h);
        var crop = new Mat(slice, rect);
        var sized = new Mat();
        Cv2.Resize(crop, sized, new Size(28, 40));
        crop.Dispose();
        return sized;
    }

    private (char Glyph, float Score) MatchChar(Mat glyph)
    {
        char best = '\0';
        var bestScore = 0.35f;
        foreach (var (ch, tmpl) in _templates)
        {
            using var diff = new Mat();
            Cv2.MatchTemplate(glyph, tmpl, diff, TemplateMatchModes.CCoeffNormed);
            Cv2.MinMaxLoc(diff, out _, out var maxVal, out _, out _);
            if (maxVal > bestScore)
            {
                bestScore = (float)maxVal;
                best = ch;
            }
        }
        return (best, bestScore);
    }

    private void BuildTemplates()
    {
        const string glyphs = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";
        foreach (var ch in glyphs)
        {
            using var img = new Mat(40, 28, MatType.CV_8UC1, Scalar.White);
            Cv2.PutText(img, ch.ToString(), new Point(2, 32), HersheyFonts.HersheySimplex, 1.05, Scalar.Black, 2, LineTypes.AntiAlias);
            var bin = new Mat();
            Cv2.Threshold(img, bin, 127, 255, ThresholdTypes.Binary);
            _templates[ch] = bin;
        }
    }

    private static Mat ColorMask(Mat hsv)
    {
        using var white = new Mat();
        using var yellow = new Mat();
        using var green = new Mat();
        using var red1 = new Mat();
        using var red2 = new Mat();
        using var red = new Mat();
        Cv2.InRange(hsv, new Scalar(0, 0, 150), new Scalar(180, 70, 255), white);
        Cv2.InRange(hsv, new Scalar(15, 70, 120), new Scalar(40, 255, 255), yellow);
        Cv2.InRange(hsv, new Scalar(40, 50, 60), new Scalar(95, 255, 255), green);
        Cv2.InRange(hsv, new Scalar(0, 70, 70), new Scalar(10, 255, 255), red1);
        Cv2.InRange(hsv, new Scalar(165, 70, 70), new Scalar(180, 255, 255), red2);
        Cv2.BitwiseOr(red1, red2, red);
        var mask = new Mat();
        Cv2.BitwiseOr(white, yellow, mask);
        Cv2.BitwiseOr(mask, green, mask);
        Cv2.BitwiseOr(mask, red, mask);
        return mask;
    }

    private static List<Rect> EdgeCandidates(Mat bgr)
    {
        var list = new List<Rect>();
        using var gray = new Mat();
        Cv2.CvtColor(bgr, gray, ColorConversionCodes.BGR2GRAY);
        using var blur = new Mat();
        Cv2.BilateralFilter(gray, blur, 7, 50, 50);
        using var edges = new Mat();
        Cv2.Canny(blur, edges, 60, 160);
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(15, 4));
        Cv2.MorphologyEx(edges, edges, MorphTypes.Close, kernel);
        Cv2.FindContours(edges, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        foreach (var c in contours)
        {
            var rect = Cv2.BoundingRect(c);
            var ratio = rect.Width / (float)Math.Max(1, rect.Height);
            var area = rect.Width * rect.Height;
            if (ratio is >= 2.0f and <= 5.8f && area > 900 && rect.Height >= 20)
                list.Add(rect);
        }
        return list;
    }

    private static List<Rect> Merge(List<Rect> rects)
    {
        var kept = new List<Rect>();
        foreach (var r in rects.OrderByDescending(x => x.Width * x.Height))
        {
            if (kept.Any(k => Overlap(k, r) > 0.55))
                continue;
            kept.Add(r);
        }
        return kept;
    }

    private static double Overlap(Rect a, Rect b)
    {
        var inter = a & b;
        if (inter.Width <= 0 || inter.Height <= 0) return 0;
        var union = a.Width * a.Height + b.Width * b.Height - inter.Width * inter.Height;
        return union <= 0 ? 0 : (inter.Width * inter.Height) / (double)union;
    }

    private static Rect Expand(Rect r, Size size, int pad)
    {
        var x = Math.Max(0, r.X - pad);
        var y = Math.Max(0, r.Y - pad);
        var w = Math.Min(size.Width - x, r.Width + pad * 2);
        var h = Math.Min(size.Height - y, r.Height + pad * 2);
        return new Rect(x, y, w, h);
    }

    private static string? FindTesseract()
    {
        var names = new[] { "tesseract", "/usr/bin/tesseract" };
        foreach (var n in names)
        {
            try
            {
                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = n,
                    Arguments = "--version",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false
                };
                using var p = System.Diagnostics.Process.Start(psi);
                p?.WaitForExit(2000);
                if (p?.ExitCode == 0)
                    return n;
            }
            catch { /* ignore */ }
        }
        return null;
    }

    private (string Text, float Score) TryTesseract(Mat plateBgr)
    {
        try
        {
            var exe = FindTesseract();
            if (exe == null) return ("", 0);
            var tmp = Path.Combine(Path.GetTempPath(), $"plate_{Guid.NewGuid():N}.png");
            Cv2.ImWrite(tmp, plateBgr);
            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = exe,
                Arguments = $"{tmp} stdout -l eng --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            using var p = System.Diagnostics.Process.Start(psi);
            var output = p?.StandardOutput.ReadToEnd() ?? "";
            p?.WaitForExit(4000);
            File.Delete(tmp);
            var text = NormalizePlate(output.Trim());
            var score = LooksLikeTaiwanPlate(text) ? 0.86f : string.IsNullOrWhiteSpace(text) ? 0 : 0.5f;
            return (text, score);
        }
        catch
        {
            return ("", 0);
        }
    }
}
