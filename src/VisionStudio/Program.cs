using System.Text.Json;
using Microsoft.AspNetCore.Http.Features;
using OpenCvSharp;
using VisionStudio.Domain;
using VisionStudio.Services;

var builder = WebApplication.CreateBuilder(args);
if (!IsHostedByIis())
    builder.WebHost.UseUrls(builder.Configuration["Urls"] ?? "http://0.0.0.0:43173");
builder.Services.Configure<FormOptions>(o =>
{
    o.MultipartBodyLengthLimit = 220L * 1024 * 1024;
});
builder.WebHost.ConfigureKestrel(o => o.Limits.MaxRequestBodySize = 220L * 1024 * 1024);

builder.Services.AddRazorPages();
builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
    o.SerializerOptions.Converters.Add(new System.Text.Json.Serialization.JsonStringEnumConverter());
});
builder.Services.AddSingleton<StoragePaths>();
builder.Services.AddSingleton<FaceEngine>();
builder.Services.AddSingleton<PlateEngine>();
builder.Services.AddSingleton<YoloOnnxDetector>();
builder.Services.AddSingleton<GalleryService>();
builder.Services.AddSingleton<PlateLogService>();
builder.Services.AddSingleton<DatasetService>();
builder.Services.AddSingleton<TrainingService>();
builder.Services.AddSingleton<ModelRegistry>();
builder.Services.AddSingleton<VisionPipeline>();
builder.Services.AddSingleton<VideoProcessor>();
builder.Services.AddSingleton<SampleSeeder>();

var app = builder.Build();
app.Services.GetRequiredService<SampleSeeder>().Seed();

if (!app.Environment.IsDevelopment())
    app.UseExceptionHandler("/Error");

app.Use(async (ctx, next) =>
{
    try { await next(); }
    catch (Exception ex) when (ctx.Request.Path.StartsWithSegments("/api"))
    {
        ctx.Response.StatusCode = 400;
        ctx.Response.ContentType = "text/plain; charset=utf-8";
        await ctx.Response.WriteAsync(ex.Message);
    }
});

app.UseStaticFiles();
app.UseRouting();
app.MapRazorPages();

app.MapGet("/media/gallery/{file}", (string file, StoragePaths paths) => Serve(paths.GalleryRoot, file));
app.MapGet("/media/history/{file}", (string file, StoragePaths paths) => Serve(paths.HistoryRoot, file));
app.MapGet("/media/samples/{**rest}", (string rest, StoragePaths paths) => ServeNested(paths.SamplesRoot, rest));
app.MapGet("/media/dataset/{id}/{file}", (string id, string file, DatasetService ds) =>
    Serve(Path.Combine(ds.Root(id), "images"), file));
app.MapGet("/media/overlay/{job}/{file}", (string job, string file, StoragePaths paths) =>
    Serve(Path.Combine(paths.UploadsRoot, job, "overlay"), file));

app.MapGet("/api/status", (FaceEngine faces, YoloOnnxDetector yolo, GalleryService gallery, DatasetService ds) =>
    Results.Json(new
    {
        faceDetector = faces.DetectorReady,
        faceRecognizer = faces.RecognizerReady,
        yolo = yolo.Ready,
        yoloModel = yolo.ModelPath,
        identities = gallery.List().Count,
        datasets = ds.List().Count
    }));

app.MapPost("/api/infer/image", async (HttpRequest req, VisionPipeline pipeline) =>
{
    var form = await req.ReadFormAsync();
    var file = form.Files.GetFile("file") ?? throw new InvalidOperationException("請上傳相片");
    using var ms = new MemoryStream();
    await file.CopyToAsync(ms);
    var options = ParseInfer(form);
    var result = pipeline.RunBytes(ms.ToArray(), options, file.FileName);
    return Results.Json(result);
});

app.MapPost("/api/infer/frame", async (HttpRequest req, VisionPipeline pipeline) =>
{
    var form = await req.ReadFormAsync();
    var file = form.Files.GetFile("file") ?? throw new InvalidOperationException("沒有影格");
    using var ms = new MemoryStream();
    await file.CopyToAsync(ms);
    var options = ParseInfer(form);
    var result = pipeline.RunBytes(ms.ToArray(), options, "stream", track: true, record: false);
    return Results.Json(new
    {
        result.ElapsedMs,
        result.Detections,
        result.Notes,
        overlay = "data:image/jpeg;base64," + result.OverlayJpegBase64
    });
});

app.MapPost("/api/infer/video", async (HttpRequest req, VideoProcessor video, StoragePaths paths) =>
{
    var form = await req.ReadFormAsync();
    var file = form.Files.GetFile("file") ?? throw new InvalidOperationException("請上傳影片");
    var dest = Path.Combine(paths.UploadsRoot, DateTime.UtcNow.ToString("yyyyMMddHHmmss") + "-" + Path.GetFileName(file.FileName));
    await using (var fs = File.Create(dest))
        await file.CopyToAsync(fs);
    var options = ParseInfer(form);
    var jobName = Path.GetFileName(Path.GetDirectoryName(dest) == paths.UploadsRoot ? dest : dest);
    var (overlayDir, events, frames) = await video.ProcessFileAsync(dest, options, null, req.HttpContext.RequestAborted);
    var job = Path.GetFileName(Path.GetDirectoryName(overlayDir)!);
    var thumbs = Directory.GetFiles(overlayDir, "*.jpg").OrderBy(f => f).Take(40)
        .Select(f => $"/media/overlay/{job}/{Path.GetFileName(f)}").ToList();
    return Results.Json(new
    {
        frames,
        eventCount = events.Count,
        faces = events.Count(e => e.Kind == DetectionKind.Face),
        plates = events.Select(e => e.PlateText).Where(t => !string.IsNullOrWhiteSpace(t)).Distinct(),
        identities = events.Select(e => e.Identity).Where(t => !string.IsNullOrWhiteSpace(t)).Distinct(),
        thumbs
    });
});

app.MapGet("/api/faces", (GalleryService g) => Results.Json(g.List().Select(f => new { f.Id, f.Name, f.Note, f.Thumbnail, embeddings = f.Embeddings.Count })));

app.MapPost("/api/faces/enroll", async (HttpRequest req, FaceEngine faces, GalleryService gallery) =>
{
    var form = await req.ReadFormAsync();
    var name = form["name"].ToString();
    var note = form["note"].ToString();
    var file = form.Files.GetFile("file") ?? throw new InvalidOperationException("請上傳人臉相片");
    if (string.IsNullOrWhiteSpace(name)) throw new InvalidOperationException("請輸入姓名");
    using var ms = new MemoryStream();
    await file.CopyToAsync(ms);
    using var mat = Cv2.ImDecode(ms.ToArray(), ImreadModes.Color);
    var detected = faces.Detect(mat, 0.5f);
    if (detected.Count == 0) throw new InvalidOperationException("相片中找不到人臉");
    var hit = detected[0];
    var emb = faces.Embed(mat, hit) ?? throw new InvalidOperationException("無法擷取人臉特徵");
    using var thumb = new Mat(mat, hit.Box);
    var id = gallery.Add(name, note, thumb, emb);
    return Results.Json(new { id.Id, id.Name, id.Thumbnail });
});

app.MapPost("/api/faces/{id}/delete", (string id, GalleryService gallery) =>
{
    gallery.Remove(id);
    return Results.Ok();
});

app.MapGet("/api/plates", (PlateLogService log) => Results.Json(log.Plates()));
app.MapGet("/api/history", (PlateLogService log) => Results.Json(log.History()));

app.MapGet("/api/datasets", (DatasetService ds) => Results.Json(ds.List()));
app.MapGet("/api/datasets/{id}", (string id, DatasetService ds) =>
{
    var info = ds.Get(id) ?? throw new InvalidOperationException("資料集不存在");
    return Results.Json(new { info, images = ds.Images(id) });
});

app.MapPost("/api/datasets", async (HttpRequest req, DatasetService ds) =>
{
    var form = await req.ReadFormAsync();
    var info = ds.Create(
        form["name"].ToString(),
        form["task"].ToString() is { Length: > 0 } t ? t : "detect",
        (form["classes"].ToString() is { Length: > 0 } c ? c : "face,plate").Split(','),
        form["yoloFamily"].ToString() is { Length: > 0 } y ? y : "yolo26n");
    return Results.Json(info);
});

app.MapPost("/api/datasets/{id}/images", async (string id, HttpRequest req, DatasetService ds) =>
{
    var form = await req.ReadFormAsync();
    var added = new List<DatasetImage>();
    foreach (var file in form.Files)
    {
        await using var s = file.OpenReadStream();
        added.Add(ds.AddImage(id, file.FileName, s));
    }
    return Results.Json(added);
});

app.MapPost("/api/datasets/{id}/annotate", async (string id, HttpRequest req, DatasetService ds) =>
{
    var payload = await JsonSerializer.DeserializeAsync<AnnotatePayload>(req.Body, JsonStore.Options)
        ?? throw new InvalidOperationException("標註內容不正確");
    ds.SaveBoxes(id, payload.FileName, payload.Boxes);
    return Results.Ok();
});

app.MapPost("/api/datasets/{id}/export", (string id, DatasetService ds) =>
{
    var yaml = ds.ExportYaml(id);
    return Results.Json(new { yaml });
});

app.MapGet("/api/train", (TrainingService t) => Results.Json(t.List()));
app.MapGet("/api/train/{id}", (string id, TrainingService t) => Results.Json(t.Get(id)));
app.MapPost("/api/train", async (HttpRequest req, TrainingService t) =>
{
    var form = await req.ReadFormAsync();
    var job = t.Start(
        form["datasetId"].ToString(),
        form["modelFamily"].ToString() is { Length: > 0 } m ? m : "yolo26n",
        int.TryParse(form["epochs"], out var e) ? e : 50,
        int.TryParse(form["imgsz"], out var s) ? s : 640,
        int.TryParse(form["batch"], out var b) ? b : 8);
    return Results.Json(job);
});

app.MapGet("/api/models", (ModelRegistry reg, FaceEngine faces) => Results.Json(new
{
    items = reg.List(),
    yoloReady = reg.Yolo.Ready,
    yoloPath = reg.Yolo.ModelPath,
    faceDetector = faces.DetectorReady,
    faceRecognizer = faces.RecognizerReady
}));

app.MapPost("/api/models/register", async (HttpRequest req, ModelRegistry reg, StoragePaths paths) =>
{
    var form = await req.ReadFormAsync();
    var file = form.Files.GetFile("file") ?? throw new InvalidOperationException("請上傳 ONNX");
    var dest = Path.Combine(paths.DataRoot, "models", DateTime.UtcNow.ToString("yyyyMMddHHmmss") + "-" + Path.GetFileName(file.FileName));
    await using (var fs = File.Create(dest))
        await file.CopyToAsync(fs);
    var classes = (form["classes"].ToString() is { Length: > 0 } c ? c : "face,plate").Split(',', StringSplitOptions.TrimEntries);
    var m = reg.Register(form["name"].ToString() is { Length: > 0 } n ? n : file.FileName,
        dest,
        form["family"].ToString() is { Length: > 0 } f ? f : "yolo11",
        classes,
        form["task"].ToString() is { Length: > 0 } t ? t : "detect");
    return Results.Json(m);
});

app.MapPost("/api/models/{id}/activate", (string id, ModelRegistry reg) =>
{
    if (id == "none") { reg.Deactivate(); return Results.Ok(); }
    reg.Activate(id);
    return Results.Ok();
});

app.MapGet("/healthz", () => Results.Ok(new { ok = true }));

app.Run();

static IResult Serve(string folder, string file)
{
    var safe = Path.GetFileName(file);
    var path = Path.Combine(folder, safe);
    if (!File.Exists(path)) return Results.NotFound();
    var content = Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".png" => "image/png",
        ".webp" => "image/webp",
        ".mp4" => "video/mp4",
        _ => "image/jpeg"
    };
    return Results.File(path, content);
}

static IResult ServeNested(string root, string rest)
{
    var full = Path.GetFullPath(Path.Combine(root, rest.Replace('\\', '/')));
    if (!full.StartsWith(Path.GetFullPath(root))) return Results.BadRequest();
    if (!File.Exists(full)) return Results.NotFound();
    return Results.File(full, "image/jpeg");
}

static InferRequest ParseInfer(IFormCollection form) => new()
{
    DetectFaces = form["faces"] != "0",
    DetectPlates = form["plates"] != "0",
    IdentifyFaces = form["identify"] != "0",
    ReadPlates = form["ocr"] != "0",
    PreferYolo = form["yolo"] != "0"
};

static bool IsHostedByIis() =>
    !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ASPNETCORE_PORT")) ||
    !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ASPNETCORE_IIS_PHYSICAL_PATH")) ||
    string.Equals(Environment.GetEnvironmentVariable("ASPNETCORE_HOSTINGSTARTUPASSEMBLIES"), "Microsoft.AspNetCore.Server.IISIntegration", StringComparison.OrdinalIgnoreCase);

public sealed record AnnotatePayload(string FileName, List<AnnotationBox> Boxes);
