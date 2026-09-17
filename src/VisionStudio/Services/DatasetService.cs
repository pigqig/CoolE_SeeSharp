using System.Diagnostics;
using System.Text;
using OpenCvSharp;
using VisionStudio.Domain;

namespace VisionStudio.Services;

public sealed class DatasetService
{
    private readonly StoragePaths _paths;

    public DatasetService(StoragePaths paths) => _paths = paths;

    public IReadOnlyList<DatasetInfo> List()
    {
        var list = new List<DatasetInfo>();
        foreach (var dir in Directory.GetDirectories(_paths.DatasetsRoot))
        {
            var meta = Path.Combine(dir, "meta.json");
            if (!File.Exists(meta)) continue;
            var info = JsonStore.Load(meta, new DatasetInfo());
            if (string.IsNullOrWhiteSpace(info.Id)) continue;
            RefreshCounts(info, dir);
            list.Add(info);
        }
        return list.OrderByDescending(x => x.CreatedUtc).ToList();
    }

    public DatasetInfo Create(string name, string task, IEnumerable<string> classes, string yoloFamily)
    {
        var info = new DatasetInfo
        {
            Id = Slug(name) + "-" + Guid.NewGuid().ToString("N")[..6],
            Name = name,
            Task = task,
            Classes = classes.Select(c => c.Trim()).Where(c => c.Length > 0).ToList(),
            YoloFamily = yoloFamily
        };
        var dir = Root(info.Id);
        Directory.CreateDirectory(Path.Combine(dir, "images"));
        Directory.CreateDirectory(Path.Combine(dir, "labels"));
        Save(info);
        return info;
    }

    public DatasetInfo? Get(string id)
    {
        var meta = Path.Combine(Root(id), "meta.json");
        if (!File.Exists(meta)) return null;
        var info = JsonStore.Load(meta, new DatasetInfo());
        if (string.IsNullOrWhiteSpace(info.Id)) return null;
        RefreshCounts(info, Root(id));
        return info;
    }

    public string Root(string id) => Path.Combine(_paths.DatasetsRoot, id);

    public IReadOnlyList<DatasetImage> Images(string id)
    {
        var dir = Path.Combine(Root(id), "images");
        if (!Directory.Exists(dir)) return Array.Empty<DatasetImage>();
        var list = new List<DatasetImage>();
        foreach (var file in Directory.GetFiles(dir).OrderBy(f => f))
        {
            var name = Path.GetFileName(file);
            var img = new DatasetImage { FileName = name };
            using var mat = Cv2.ImRead(file);
            if (!mat.Empty())
            {
                img.Width = mat.Width;
                img.Height = mat.Height;
            }
            var label = Path.Combine(Root(id), "labels", Path.ChangeExtension(name, ".txt"));
            if (File.Exists(label) && img.Width > 0)
                img.Boxes = ParseYolo(File.ReadAllText(label), img.Width, img.Height);
            list.Add(img);
        }
        return list;
    }

    public DatasetImage AddImage(string id, string fileName, Stream data)
    {
        var safe = $"{DateTime.UtcNow:yyyyMMddHHmmss}-{Sanitize(fileName)}";
        var dest = Path.Combine(Root(id), "images", safe);
        using (var fs = File.Create(dest))
            data.CopyTo(fs);
        using var mat = Cv2.ImRead(dest);
        File.WriteAllText(Path.Combine(Root(id), "labels", Path.ChangeExtension(safe, ".txt")), "");
        var info = Get(id);
        if (info != null) Save(info);
        return new DatasetImage { FileName = safe, Width = mat.Width, Height = mat.Height };
    }

    public void SaveBoxes(string id, string fileName, List<AnnotationBox> boxes)
    {
        var imgPath = Path.Combine(Root(id), "images", fileName);
        using var mat = Cv2.ImRead(imgPath);
        if (mat.Empty()) return;
        var sb = new StringBuilder();
        foreach (var b in boxes)
        {
            var xc = (b.X + b.Width / 2f) / mat.Width;
            var yc = (b.Y + b.Height / 2f) / mat.Height;
            var w = b.Width / mat.Width;
            var h = b.Height / mat.Height;
            sb.AppendLine($"{b.ClassId} {xc:0.######} {yc:0.######} {w:0.######} {h:0.######}");
        }
        File.WriteAllText(Path.Combine(Root(id), "labels", Path.ChangeExtension(fileName, ".txt")), sb.ToString());
        var info = Get(id);
        if (info != null) Save(info);
    }

    public string ExportYaml(string id)
    {
        var info = Get(id) ?? throw new InvalidOperationException("資料集不存在");
        var dir = Root(id);
        var yamlPath = Path.Combine(dir, "data.yaml");
        var sb = new StringBuilder();
        sb.AppendLine($"path: {dir}");
        sb.AppendLine("train: images");
        sb.AppendLine("val: images");
        sb.AppendLine("names:");
        for (var i = 0; i < info.Classes.Count; i++)
            sb.AppendLine($"  {i}: {info.Classes[i]}");
        File.WriteAllText(yamlPath, sb.ToString());
        File.WriteAllLines(Path.Combine(dir, "classes.txt"), info.Classes);
        return yamlPath;
    }

    public void ImportFiles(string id, IEnumerable<string> sourceFiles)
    {
        foreach (var src in sourceFiles)
        {
            if (!File.Exists(src)) continue;
            using var fs = File.OpenRead(src);
            AddImage(id, Path.GetFileName(src), fs);
        }
    }

    private void Save(DatasetInfo info)
    {
        RefreshCounts(info, Root(info.Id));
        JsonStore.Save(Path.Combine(Root(info.Id), "meta.json"), info);
    }

    private static void RefreshCounts(DatasetInfo info, string dir)
    {
        var images = Directory.Exists(Path.Combine(dir, "images"))
            ? Directory.GetFiles(Path.Combine(dir, "images")).Length : 0;
        var labeled = Directory.Exists(Path.Combine(dir, "labels"))
            ? Directory.GetFiles(Path.Combine(dir, "labels"), "*.txt").Count(f => new FileInfo(f).Length > 2)
            : 0;
        info.ImageCount = images;
        info.LabeledCount = labeled;
    }

    private static List<AnnotationBox> ParseYolo(string text, int w, int h)
    {
        var boxes = new List<AnnotationBox>();
        foreach (var line in text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var p = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (p.Length < 5) continue;
            if (!int.TryParse(p[0], out var cls)) continue;
            var xc = float.Parse(p[1]) * w;
            var yc = float.Parse(p[2]) * h;
            var bw = float.Parse(p[3]) * w;
            var bh = float.Parse(p[4]) * h;
            boxes.Add(new AnnotationBox
            {
                ClassId = cls,
                X = xc - bw / 2,
                Y = yc - bh / 2,
                Width = bw,
                Height = bh
            });
        }
        return boxes;
    }

    private static string Slug(string name)
    {
        var s = new string(name.Select(c => char.IsLetterOrDigit(c) ? c : '-').ToArray());
        return string.IsNullOrWhiteSpace(s) ? "ds" : s.Trim('-').ToLowerInvariant();
    }

    private static string Sanitize(string fileName)
    {
        var name = Path.GetFileName(fileName);
        foreach (var c in Path.GetInvalidFileNameChars())
            name = name.Replace(c, '_');
        return name;
    }
}

public sealed class TrainingService
{
    private readonly StoragePaths _paths;
    private readonly DatasetService _datasets;
    private readonly Dictionary<string, TrainingJob> _jobs = new();
    private readonly object _gate = new();

    public TrainingService(StoragePaths paths, DatasetService datasets)
    {
        _paths = paths;
        _datasets = datasets;
        foreach (var file in Directory.GetFiles(_paths.JobsRoot, "*.json"))
        {
            var job = JsonStore.Load(file, new TrainingJob());
            if (!string.IsNullOrWhiteSpace(job.Id) && job.Id != new TrainingJob().Id)
                _jobs[job.Id] = job;
            else if (!string.IsNullOrWhiteSpace(job.DatasetId))
                _jobs[job.Id] = job;
        }
    }

    public IReadOnlyList<TrainingJob> List() => _jobs.Values.OrderByDescending(j => j.CreatedUtc).ToList();
    public TrainingJob? Get(string id) => _jobs.TryGetValue(id, out var j) ? j : null;

    public TrainingJob Start(string datasetId, string modelFamily, int epochs, int imgsz, int batch)
    {
        var ds = _datasets.Get(datasetId) ?? throw new InvalidOperationException("資料集不存在");
        if (ds.ImageCount == 0) throw new InvalidOperationException("資料集沒有相片");
        var yaml = _datasets.ExportYaml(datasetId);
        var job = new TrainingJob
        {
            DatasetId = datasetId,
            ModelFamily = modelFamily,
            Epochs = epochs,
            ImageSize = imgsz,
            Batch = batch,
            Status = "running"
        };
        lock (_gate) _jobs[job.Id] = job;
        Persist(job);

        _ = Task.Run(() => Run(job, yaml));
        return job;
    }

    private void Run(TrainingJob job, string yaml)
    {
        try
        {
            Log(job, $"資料集設定已匯出：{yaml}");
            Log(job, $"建議架構：{job.ModelFamily}  | epochs={job.Epochs}  imgsz={job.ImageSize}  batch={job.Batch}");

            var python = FindPython();
            var script = Path.Combine(_paths.PythonRoot, "train_yolo.py");
            if (python == null || !File.Exists(script))
            {
                job.Status = "blocked";
                job.Message = "尚未安裝 Ultralytics。資料集已匯出，可在有 GPU 的主機執行 python/train_yolo.py。";
                Log(job, job.Message);
                WriteRecipe(job, yaml);
                job.FinishedUtc = DateTime.UtcNow;
                Persist(job);
                return;
            }

            var outDir = Path.Combine(_paths.JobsRoot, job.Id);
            Directory.CreateDirectory(outDir);
            var psi = new ProcessStartInfo
            {
                FileName = python,
                Arguments = $"\"{script}\" --data \"{yaml}\" --model {job.ModelFamily} --epochs {job.Epochs} --imgsz {job.ImageSize} --batch {job.Batch} --project \"{outDir}\" --name run",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                WorkingDirectory = _paths.RepoRoot
            };
            psi.Environment["PYTHONUNBUFFERED"] = "1";
            using var proc = new Process { StartInfo = psi };
            proc.OutputDataReceived += (_, e) => { if (e.Data != null) Log(job, e.Data); };
            proc.ErrorDataReceived += (_, e) => { if (e.Data != null) Log(job, e.Data); };
            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            if (!proc.WaitForExit((int)TimeSpan.FromHours(6).TotalMilliseconds))
            {
                proc.Kill(true);
                throw new TimeoutException("訓練逾時");
            }
            if (proc.ExitCode != 0)
            {
                job.Status = "failed";
                job.Message = $"訓練行程結束代碼 {proc.ExitCode}";
            }
            else
            {
                job.Status = "completed";
                job.Message = "訓練完成";
                var best = Directory.GetFiles(outDir, "best.pt", SearchOption.AllDirectories).FirstOrDefault();
                var onnx = Directory.GetFiles(outDir, "*.onnx", SearchOption.AllDirectories).FirstOrDefault();
                job.BestWeights = best;
                job.ExportedOnnx = onnx;
                if (onnx != null)
                {
                    var dest = Path.Combine(_paths.DataRoot, "models", $"{job.ModelFamily}-{job.Id}.onnx");
                    Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
                    File.Copy(onnx, dest, true);
                    job.ExportedOnnx = dest;
                    Log(job, $"已匯出 ONNX：{dest}");
                }
            }
        }
        catch (Exception ex)
        {
            job.Status = "failed";
            job.Message = ex.Message;
            Log(job, "錯誤：" + ex.Message);
        }
        finally
        {
            job.FinishedUtc = DateTime.UtcNow;
            Persist(job);
        }
    }

    private void WriteRecipe(TrainingJob job, string yaml)
    {
        var recipe = Path.Combine(_paths.JobsRoot, job.Id + ".command.txt");
        File.WriteAllText(recipe, $"""
# VisionStudio 訓練指令（請在已安裝 ultralytics 的環境執行）
python python/train_yolo.py \
  --data "{yaml}" \
  --model {job.ModelFamily} \
  --epochs {job.Epochs} \
  --imgsz {job.ImageSize} \
  --batch {job.Batch} \
  --project "{Path.Combine(_paths.JobsRoot, job.Id)}" \
  --name run
""");
        Log(job, "已寫入訓練指令：" + recipe);
    }

    private static string? FindPython()
    {
        foreach (var name in new[] { "python3", "python" })
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = name,
                    Arguments = "-c \"import ultralytics; print('ok')\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false
                };
                using var p = Process.Start(psi);
                p?.WaitForExit(8000);
                if (p?.ExitCode == 0)
                    return name;
            }
            catch { /* ignore */ }
        }
        return null;
    }

    private void Log(TrainingJob job, string line)
    {
        lock (_gate)
        {
            job.Logs.Add($"[{DateTime.Now:HH:mm:ss}] {line}");
            if (job.Logs.Count > 400)
                job.Logs = job.Logs.TakeLast(300).ToList();
            Persist(job);
        }
    }

    private void Persist(TrainingJob job)
        => JsonStore.Save(Path.Combine(_paths.JobsRoot, job.Id + ".json"), job);
}
