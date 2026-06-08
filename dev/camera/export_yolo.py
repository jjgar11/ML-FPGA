"""
Run on dev machine (not Ultra96).
Downloads YOLOv8n and exports to ONNX for ARM64 inference.

Usage:
    pip install ultralytics
    python dev/camera/export_yolo.py
"""

from ultralytics import YOLO
from pathlib import Path

OUT = Path("data/models/yolov8n.onnx")
OUT.parent.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True, opset=12)

import shutil
shutil.move("yolov8n.onnx", OUT)
print(f"Saved: {OUT}")
