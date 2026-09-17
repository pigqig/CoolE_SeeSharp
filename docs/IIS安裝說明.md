# 在 IIS 安裝 VisionStudio（Windows 測試）

這是 ASP.NET Core 8 網站，**不能**把原始碼資料夾直接當舊版 ASP.NET 網站開。請用「發佈後的檔案」或在 Windows 上執行發佈。

## 一、伺服器先裝這些

1. **IIS**（含「WWW 服務」與「靜態內容」）
2. **ASP.NET Core 8.0 Hosting Bundle（x64）**  
   下載：[.NET 8 Hosting Bundle](https://dotnet.microsoft.com/download/dotnet/8.0)  
   裝完若 IIS 已在跑，請在系統管理員命令提示字元執行：
   ```bat
   net stop was /y
   net start w3svc
   ```
3. **Visual C++ 2015-2022 x64 可轉散發套件**（OpenCV／ONNX 需要）
4. （建議）**ffmpeg** 加進系統 PATH，影片辨識才會動  
   例如解壓到 `C:\ffmpeg\bin` 後把該路徑加入 PATH
5. （建議）**Tesseract OCR**，車牌讀字比較穩  
   https://github.com/UB-Mannheim/tesseract/wiki

不需要在 IIS 應用程式集區選 .NET CLR 4.0。集區必須是 **No Managed Code**。

## 二、取得要放到 IIS 的檔案

### 方法 A：在 Windows 用原始碼發佈（建議）

1. 安裝 [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
2. 打開專案資料夾，PowerShell：

```powershell
cd src\VisionStudio
dotnet publish -c Release -r win-x64 --self-contained false -o C:\inetpub\wwwroot\VisionStudio
```

或用 Visual Studio：開啟 `VisionStudio.sln` → 對 `VisionStudio` 專案右鍵「發佈」→ 選設定檔 `IIS`。

### 方法 B：用現成發佈資料夾

若你已有 `dist\iis\` 或 `VisionStudio-IIS-win-x64.zip`：

1. 解壓到例如 `C:\inetpub\wwwroot\VisionStudio`
2. 確認裡面有 `VisionStudio.dll`、`web.config`、`App_Data\models\`

## 三、IIS 站台設定

1. 開啟「IIS 管理員」
2. **應用程式集區** → 新增：
   - 名稱：`VisionStudio`
   - .NET CLR 版本：**無 Managed 程式碼**
   - 受控管線：**整合**
3. 集區進階設定：
   - **啟用 32 位元應用程式** = False
   - **識別** = `ApplicationPoolIdentity`（或一個本機帳號）
   - **載入使用者設定檔** = True（原生 DLL 比較不容易缺環境）
4. **網站** → 新增網站：
   - 名稱：`VisionStudio`
   - 應用程式集區：剛才的 `VisionStudio`
   - 實體路徑：`C:\inetpub\wwwroot\VisionStudio`
   - 繫結：`http`、埠 `8088`（或你要的埠；80 若已被佔就改）
5. 對實體路徑按右鍵 → **編輯權限**：
   - 加入 `IIS AppPool\VisionStudio`
   - 允許 **修改**（要寫入 `App_Data\runtime` 與 `logs`）

瀏覽器開：`http://伺服器IP:8088/`

健康檢查：`http://伺服器IP:8088/healthz` 應回 `{"ok":true}`

## 四、常見問題

| 現象 | 處理 |
| --- | --- |
| 500.19 / 找不到 AspNetCoreModuleV2 | 未裝 Hosting Bundle，或裝完沒重開 IIS |
| 500.30 啟動失敗 | 看站台目錄 `logs\stdout*.log`；多半是權限或缺 VC++ |
| 人臉模型載不到 | 確認 `App_Data\models` 有兩個 `.onnx` |
| 影片沒結果 | IIS 集區帳號的 PATH 找不到 ffmpeg，把 `ffmpeg.exe` 放到站台目錄或設系統 PATH 後重開集區 |
| 上傳影片失敗 | `web.config` 已放大到約 220MB；IIS 要求篩選也要允許 |
| 攝影機串流沒畫面 | 請用 https 或 localhost；瀏覽器才會開放攝影機 |

YOLO 訓練請在有 GPU、已 `pip install ultralytics` 的機器做，不必在 IIS 上訓。IIS 這台負責標註、推論與身份庫即可。
