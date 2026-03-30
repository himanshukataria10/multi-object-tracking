import streamlit as st
import cv2
from ultralytics import YOLO

st.title("Multi Object Detection App")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload a video")

if uploaded_file is not None:
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp.mp4")

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for r in results.boxes.data:
            x1, y1, x2, y2, score, cls = r
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0,255,0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()