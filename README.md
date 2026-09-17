# VisionStudio

.NET 8 影像訓練與辨識工作台：人臉身份、車牌讀字、YOLO 資料集／訓練／ONNX 推論，支援相片、影片與攝影機串流。

## 建議模型（請先看這裡）

| 用途 | 建議 |
| --- | --- |
| 新專案偵測訓練 | **YOLO26n**，不夠再升 **YOLO26s** |
| 傾斜車牌 | **YOLO26n-obb** |
| 人臉「是誰」 | YuNet 或 YOLO 只做人臉框，身份用 **SFace／ArcFace** |
| 車牌幾個字 | 定位後做 **OCR**，不要用 YOLO 分類號碼 |
| 已有 YOLO11 量產 | 可續用 YOLO11n／s |
| 不要用 | YOLO12、YOLO13（官方不建議量產） |

完整理由與流程見 [docs/VISION_PLAN.md](docs/VISION_PLAN.md)，畫面上也有「技術規劃」頁。

## 本機執行

需要 .NET 8 SDK、ffmpeg。人臉模型已放在 `models/`。車牌讀字建議安裝 Tesseract：

```bash
sudo apt-get install -y tesseract-ocr
chmod +x run.sh
./run.sh
```

瀏覽器開啟 http://127.0.0.1:43173

### 訓練（可選，需 GPU 較實用）

```bash
pip install ultralytics
# 在 Web「訓練工作」啟動，或：
python python/train_yolo.py --data <data.yaml> --model yolo26n --epochs 50 --imgsz 640 --project ./runs --name demo
```

YOLO26 匯出：`yolo export model=best.pt format=onnx opset=18`  
YOLO11：`opset=17`，再到「模型管理」上傳 ONNX。

## 功能

- 資料集上傳、畫框標註、匯出 YOLO `data.yaml`
- 訓練工作佇列（有 ultralytics 就真的訓，沒有就留下指令）
- 相片／影片（ffmpeg 抽帧）／攝影機串流辨識
- 人臉身份庫（SFace 特徵比對）
- 車牌定位與讀字、進出紀錄
- 內建 YuNet 人臉偵測；未匯入 YOLO 時車牌走色彩＋輪廓

Ultralytics 權重多為 AGPL-3.0，商用請自行確認授權。
