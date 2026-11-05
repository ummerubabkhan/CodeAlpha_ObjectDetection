"""
Task 3: Object Detection and Tracking

- Video input with OpenCV (video file or webcam)
- Object detection with YOLOv8 (yolov8n.pt)
- Object tracking with Deep SORT, with tracking IDs and labels
- Saves the result as output_tracked.mp4

Install:  pip install ultralytics opencv-python deep-sort-realtime
Run:      python "Object Detection.py"
"""

import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------- Settings (change these to adjust speed) ----------------
RESIZE_WIDTH = 480        
FRAME_SKIP = 3            # process every 3rd frame
CONF_THRESHOLD = 0.4      # Minimum confidence to keep a detection
DETECT_CLASSES = [0, 2, 3, 5, 7]     # For traffic videos: [0, 2, 3, 5, 7]
                          # (person, car, motorcycle, bus, truck)

# ---- 1. Video input ----
# Automatically finds the folder this script is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(BASE_DIR, "video.mp4")   # For webcam use: SOURCE = 0

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("ERROR: Could not open the video. Make sure video.mp4 is in this folder:")
    print(BASE_DIR)
    raise SystemExit

# ---- Work out the resized frame size ----
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = RESIZE_WIDTH / orig_width
width = RESIZE_WIDTH
height = int(orig_height * scale)

# ---- Output video writer ----
fps = cap.get(cv2.CAP_PROP_FPS) or 25
out_path = os.path.join(BASE_DIR, "output_tracked.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps / FRAME_SKIP, (width, height))

# ---- 2. Pre-trained detection model (YOLO) ----
model = YOLO("yolov8n.pt")

# ---- 4. Tracker (Deep SORT) ----
tracker = DeepSort(max_age=30)

print("Running... press 'q' on the video window to stop.")

frame_count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Skip frames to speed things up
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # Shrink the frame before processing
    frame = cv2.resize(frame, (width, height))

    # ---- 3. Detect objects in the frame ----
    results = model(frame, verbose=False, classes=DETECT_CLASSES, imgsz=320)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        # Deep SORT format: ([left, top, width, height], confidence, class)
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

    # ---- Update the tracker with the new detections ----
    tracks = tracker.update_tracks(detections, frame=frame)

    # ---- 5. Draw boxes, labels and tracking IDs ----
    for track in tracks:
        if not track.is_confirmed():
            continue
        left, top, right, bottom = map(int, track.to_ltrb())
        cls_id = track.get_det_class()
        name = model.names[cls_id] if cls_id is not None else "object"
        label = f"ID {track.track_id} | {name}"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, max(top - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame to the output video
    writer.write(frame)

    # Show the frame in real time
    cv2.imshow("Object Detection and Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print("Done! Output video saved at:", out_path)