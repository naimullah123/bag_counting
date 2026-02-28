import cv2
from ultralytics import YOLO
import supervision as sv

VIDEO_PATH = r"C:\bag_counting_project\Input.mp4\Problem Statement Scenario3.mp4"
OUTPUT_PATH = r"C:\bag_counting_project\output_scenario3.mp4"

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open video")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

tracker = sv.ByteTrack()

# BLUE LINE position (count line)
blue_line_x = int(frame_width * 0.55)

# Store previous positions
prev_x = {}

# Store counted IDs (avoid double counting)
counted_ids = set()

# Final sack count
sack_count = 0

print("Processing...")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Detect persons
    results = model(frame, conf=0.4)[0]
    detections = sv.Detections.from_ultralytics(results)

    if detections.class_id is not None:
        detections = detections[detections.class_id == 0]

    detections = tracker.update_with_detections(detections)

    annotated = frame.copy()

    for box, track_id in zip(detections.xyxy, detections.tracker_id):

        if track_id is None:
            continue

        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2

        if track_id not in prev_x:
            prev_x[track_id] = center_x
            continue

        previous = prev_x[track_id]

        # ONLY count RIGHT → LEFT crossing BLUE line
        if previous > blue_line_x and center_x <= blue_line_x:

            if track_id not in counted_ids:

                sack_count += 1
                counted_ids.add(track_id)

                print(f"Sack counted. Total: {sack_count}")

        prev_x[track_id] = center_x

        # Draw bounding box
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 3)

    # Draw BLUE counting line
    cv2.line(annotated,
             (blue_line_x,0),
             (blue_line_x,frame_height),
             (255,0,0),
             4)

    # Show ONLY total sack count
    cv2.putText(annotated,
                f"TOTAL SACKS: {sack_count}",
                (50,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0,255,0),
                4)

    out.write(annotated)

    cv2.imshow("Sack Counter", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Finished")
print("Final Sack Count:", sack_count)