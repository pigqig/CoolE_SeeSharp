# 在 Windows 發佈到 IIS 目錄
param(
    [string]$Output = "C:\inetpub\wwwroot\VisionStudio"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

dotnet publish "$root\src\VisionStudio\VisionStudio.csproj" `
    -c Release `
    -r win-x64 `
    --self-contained false `
    -o $Output

Write-Host "已發佈到 $Output"
Write-Host "請依 docs\IIS安裝說明.md 建立應用程式集區與網站。"
