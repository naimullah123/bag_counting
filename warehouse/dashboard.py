import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(layout="wide")

# Load trained sack model
model = YOLO("models/best.pt")

# Counter
bags_in = 0

# HEADER
st.markdown(
    "<h1 style='text-align:center;background-color:#2a6f84;color:white;padding:10px;'>Warehouse Management System</h1>",
    unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align:center;background-color:#1e4f66;color:white;padding:8px;'>Bag Counting Management System</h3>",
    unsafe_allow_html=True)

st.write("")

# ======================
# SINGLE CAMERA PANEL
# ======================

st.markdown("### Gate Number : 1")

frame_placeholder = st.empty()

video_path = "videos/scenario1.mp4"

def process_video():

    global bags_in

    cap = cv2.VideoCapture(video_path)

    line_x = 500
    counted_ids = set()
    prev_positions = {}

    counter_placeholder = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.cpu()

            for box, track_id in zip(boxes, ids):

                track_id = int(track_id)

                x1,y1,x2,y2 = map(int, box)

                cx = int((x1+x2)/2)

                if track_id not in prev_positions:
                    prev_positions[track_id] = cx

                prev_x = prev_positions[track_id]

                # crossing detection
                if prev_x > line_x and cx <= line_x and track_id not in counted_ids:

                    bags_in += 1
                    counted_ids.add(track_id)

                prev_positions[track_id] = cx

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.line(frame,(line_x,0),(line_x,frame.shape[0]),(255,0,0),3)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame)

        # Update counter LIVE
        counter_placeholder.markdown(f"### Bags In : {bags_in}")

    cap.release()

process_video()




st.write("")
st.write("")

# ======================
# IOT SECTION
# ======================

st.markdown(
    "<h3 style='text-align:center;background-color:#2a6f84;color:white;padding:8px;'>IOT Parameters Monitoring</h3>",
    unsafe_allow_html=True)

c1,c2 = st.columns(2)

with c1:
    st.write("Temperature : 28°C")
    st.write("Humidity : 65%")
    st.write("Phosphine Gas Level : Safe")

with c2:
    st.write("Smoke and Fire Status : Normal")
    st.write("Gate Open/Close Status : Open")