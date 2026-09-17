# 影像訓練與辨識規劃（CoolE_SeeSharp 開發備註）

適用：人臉辨識、車牌辨識、相片／影片／串流，以及 C# .NET 8 Web 工作台。

## 結論

1. **新專案偵測模型用 YOLO26n（不夠再升 s）**。2026 年 1 月 Ultralytics 最新版；NMS-free 可簡化部署；YOLO26n 的 CPU ONNX 約 38.9 ms，官方數字比 YOLO11n 快約 43%。
2. **穩定量產若已在 YOLO11，可繼續用 YOLO11n／s**，不必為換代而換代。
3. **不要用 YOLO12、YOLO13 做量產**（官方：注意力層訓練不穩、記憶體高、CPU 慢；YOLO13 增益有限且再現性差）。
4. **YOLO 只負責「框在哪」**。人臉「是誰」走特徵向量（SFace／ArcFace）；車牌「幾號」走 OCR。
5. **C# .NET 8 做 Web、標註、ONNX 推論、串流**；**訓練仍用 Python Ultralytics**。在 C# 反傳 YOLO 不划算。

## YOLO 版本怎麼選

| 情境 | 選型 | 理由 |
| --- | --- | --- |
| 新專案、CPU／邊緣、資料量普通 | YOLO26n | 參數約 2.4M；部署簡單 |
| 現場要更高召回 | YOLO26s | 約 9.5M，平衡速度與精度 |
| GPU 伺服器、複雜場景 | YOLO26m 以上 | T4 TensorRT 仍可即時 |
| 傾斜、側向、空拍車牌 | YOLO26n-obb | 旋轉框；官方稱 OBB 相對 YOLO11 最高約 +3.4 mAP（DOTA） |
| 遠距小車牌 | `yolo26n-p2.yaml` 自行訓練 | P2 小物體頭，沒有現成 pt |
| 影片要穩定 ID | YOLO26 + ByteTrack | Python `model.track`；C# 第一版用 IoU |
| 舊專案已驗證 | YOLO11n／s | 五種任務權重齊、生態熟 |
| .NET 推論匯出 | YOLO26：ONNX opset **18**；v8～v12：opset **17** | 對齊 YoloDotNet／YoloSharp |

YOLOv8 仍能用，但 2026 年新專案沒必要從頭選它。YOLOv5 僅維護舊專案。

C# 推論套件：本工作台內建 ONNX 解碼（YOLOv8／11 的 `1×(4+nc)×8400`，以及 YOLO26 e2e 的 `1×300×6`）。也可以換成 [YoloSharp](https://github.com/dme-compunet/YoloSharp) 或 [YoloDotNet](https://github.com/NickSwardh/YoloDotNet)。

## 人臉：兩段式

```
相片／影格 → 偵測（YuNet 或 YOLO-face）→ 對齊五官
          → 特徵（SFace／ArcFace）→ 與身份庫餘弦比對
```

- 偵測資料：現場臉＋公開集（如 WIDER FACE）微調 YOLO26n，類別只留 `face`。本系統內建 **YuNet 2023mar ONNX**（OpenCV zoo，Apache-2.0），不必等自訓完成就能用。
- 身份資料：每人 3～8 張，含側臉與燈光變化。門檻先設餘弦 **0.45**，再依誤報／漏報調。
- 法規：人臉是個人資料，要告知目的、保存期限、存取權限，不要把示範庫當正式員工檔。

YOLO 分類頭無法穩定「認人」。人一多、角度一變就垮。

## 車牌：偵測 + 讀字

台灣常見格式：`ABC-1234`、`AB-1234`、綠底電動、紅底營業、黃底租賃。

```
影格 → YOLO26-obb 找車牌（或第一版色彩／輪廓）
     → 透視矯正 → OCR（Tesseract／PaddleOCR／自訓 CRNN）
     → 正規化與正則過濾
```

正則起步：`^[A-Z]{2,3}-?\d{3,4}$`。

資料標註用 OBB 較耐側向。imgsz 車牌可用 640，遠距再試 1024。

本工作台未載入 YOLO 時：HSV 白／黃／綠／紅＋輪廓找矩形，再字元樣板或 Tesseract。這是為了讓 Web 當下能示範，不是量產終態。

## 相片、影片、串流

| 來源 | 做法 |
| --- | --- |
| 相片 | 單張推論，畫框回 JPEG |
| 影片檔 | ffmpeg 抽帧（預設 4 fps）→ 逐張推論。事後分析夠用 |
| 瀏覽器攝影機 | `getUserMedia` 送 JPEG 到 `/api/infer/frame` |
| RTSP | ffmpeg 拉流後接同一條影格管線（可再加） |
| 即時 25 fps＋多路 | GPU（CUDA／TensorRT）＋佇列；CPU 只適合低路數 |

追蹤：第一版 IoU；遮擋多再上 ByteTrack。

## .NET 8 架構

```
瀏覽器 Razor Pages
    ├ 資料集／畫框標註     → data/runtime/datasets（YOLO txt + data.yaml）
    ├ 訓練工作             → python/train_yolo.py（Ultralytics）
    ├ 相片／影片／串流     → VisionPipeline
    ├ 人臉身份庫           → SFace 特徵 JSON
    └ 模型管理             → 上傳 ONNX、啟用 YOLO
VisionPipeline
    ├ YoloOnnxDetector（可選）
    ├ FaceEngine（YuNet + SFace）
    └ PlateEngine（定位 + OCR）
```

- 推論：`Microsoft.ML.OnnxRuntime` + `OpenCvSharp4`
- 影片：系統 `ffmpeg`（slim OpenCV 不一定含可靠 VideoIO）
- 不在 C# 做 YOLO 訓練迴圈

授權：Ultralytics 預訓練權重多為 **AGPL-3.0**。商用產品需企業授權或完全自訓並自行法務確認。YuNet／SFace 來自 OpenCV zoo，授權較友善。

## 落地順序

1. 用內建 YuNet＋車牌定位跑通現場相片與攝影機（本 repo 已做到）。
2. 把漏檢、誤檢圖放進資料集畫框。
3. GPU 主機訓 YOLO26n／obb，50～150 epoch。
4. 匯出 ONNX，在「模型管理」啟用。
5. 身份庫持續補臉；車牌 OCR 不夠再換 PaddleOCR 或自訓讀字模型。
6. 多路即時再上 GPU 與正式追蹤器。

## 訓練指令範例

```bash
pip install ultralytics
python python/train_yolo.py --data data.yaml --model yolo26n --epochs 80 --imgsz 640 --batch 8 --project runs --name faces
# 車牌傾斜：
python python/train_yolo.py --data plates.yaml --model yolo26n-obb --epochs 100 --imgsz 640 --project runs --name plates
```
