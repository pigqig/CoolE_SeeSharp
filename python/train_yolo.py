#!/usr/bin/env python3
"""Ultralytics YOLO 訓練與 ONNX 匯出（由 VisionStudio 呼叫）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO and export ONNX for VisionStudio")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--model", default="yolo26n", help="yolo26n / yolo26s / yolo26n-obb / yolo11n ...")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="run")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("尚未安裝 ultralytics。請執行：pip install ultralytics", file=sys.stderr)
        return 2

    weights = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    print(f"載入預訓練權重 {weights}")
    model = YOLO(weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        plots=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        print(f"找不到 {best}", file=sys.stderr)
        return 3

    trained = YOLO(str(best))
    opset = 18 if "26" in args.model else 17
    print(f"匯出 ONNX（opset={opset}）")
    trained.export(format="onnx", opset=opset, simplify=True)
    print("訓練與匯出完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
