using System.Diagnostics;

namespace VisionStudio.Services;

public static class ToolPaths
{
    public static string Ffmpeg()
    {
        var configured = Environment.GetEnvironmentVariable("VISION_FFMPEG");
        if (!string.IsNullOrWhiteSpace(configured) && File.Exists(configured))
            return configured;

        var names = OperatingSystem.IsWindows()
            ? new[] { "ffmpeg.exe", "ffmpeg" }
            : new[] { "ffmpeg" };
        foreach (var name in names)
        {
            if (ExistsOnPath(name))
                return name;
        }

        if (OperatingSystem.IsWindows())
        {
            var extras = new[]
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "ffmpeg", "bin", "ffmpeg.exe"),
                @"C:\ffmpeg\bin\ffmpeg.exe",
                Path.Combine(AppContext.BaseDirectory, "ffmpeg.exe")
            };
            foreach (var p in extras)
                if (File.Exists(p)) return p;
        }

        return "ffmpeg";
    }

    private static bool ExistsOnPath(string fileName)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = "-version",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            using var p = Process.Start(psi);
            p?.WaitForExit(2000);
            return p?.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }
}
