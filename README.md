# Task 3: Object Detection and Tracking

## What it does
Detects objects in a video using YOLOv8 and tracks them with Deep SORT.
Each object gets a bounding box, a label and a tracking ID
(for example: ID 5 | car).

## Features
- Video file input with OpenCV (webcam also possible)
- Pre-trained YOLOv8 model (yolov8n.pt)
- Detects: person, car, motorcycle, bus and truck
- Deep SORT tracking with a unique ID for each object
- Live display and the result saved as a video
- Speed settings at the top of the code (frame skip, resize, confidence)

## How it works
1. OpenCV reads the video frame by frame.
2. YOLOv8 detects objects in each frame.
3. Deep SORT gives each object a tracking ID across frames.
4. Boxes, labels and IDs are drawn and saved to output_tracked.mp4.

## How to run
1. Install Python 3.
2. Install libraries: `pip install -r requirements.txt`
3. Keep yolov8n.pt and video.mp4 in the same folder as the code.
4. Run: `python "Object Detection.py"`
5. Press q on the video window to stop.

To use a webcam, change SOURCE to 0 in the code.

## Files
- Object Detection.py : main code
- yolov8n.pt : pretrained YOLOv8 model
- video.mp4 : input video
- output_tracked.mp4 : output video with boxes and tracking IDs
- requirements.txt : libraries needed

## Author
Umm e Rubab