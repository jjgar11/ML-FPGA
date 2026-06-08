"""
Runs on Ultra96 (Python 3.7, OpenCV 3.4, headless).
YOLOv3-tiny inference from USB camera via cv2.dnn (Darknet format).

Files needed alongside this script:
  - yolov3-tiny.cfg     (included in shared folder)
  - yolov3-tiny.weights (34 MB — download separately, see README)

Usage:
    python3 infer_yolo.py [--device /dev/video0] [--save]
"""

import argparse
import time
import cv2
import numpy as np

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

CONF_THRESH = 0.4
IOU_THRESH  = 0.4
INPUT_SIZE  = 416  # YOLOv3-tiny native resolution


def load_net(cfg, weights):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    out_layers = [net.getLayerNames()[i[0] - 1]
                  for i in net.getUnconnectedOutLayers()]
    return net, out_layers


def detect(net, out_layers, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0,
                                 (INPUT_SIZE, INPUT_SIZE),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf < CONF_THRESH:
                continue
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            boxes.append([x, y, int(bw * w), int(bh * h)])
            confidences.append(conf)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, IOU_THRESH)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((class_ids[i], confidences[i], boxes[i]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",     default="yolov3-tiny.cfg")
    parser.add_argument("--weights", default="yolov3-tiny.weights")
    parser.add_argument("--device",  default="/dev/video0")
    parser.add_argument("--save",    action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    net, out_layers = load_net(args.cfg, args.weights)
    print("Model loaded. Starting camera...")

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError("Cannot open: " + args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        t0 = time.time()
        detections = detect(net, out_layers, frame)
        ms = (time.time() - t0) * 1000

        print("Frame {:04d} | {:.0f} ms | {} objects".format(
              frame_idx, ms, len(detections)))
        for class_id, conf, box in detections:
            label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
            print("  {}: {:.2f}  box={}".format(label, conf, box))
            x, y, bw, bh = box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, "{} {:.2f}".format(label, conf),
                        (x, max(y - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if args.save:
            cv2.imwrite("frame_{:04d}.jpg".format(frame_idx), frame)

        frame_idx += 1

    cap.release()


if __name__ == "__main__":
    main()
