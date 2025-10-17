import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

st.title("Webcam with Face Blur & Posture Detection")

# Load models
pose_model = YOLO("yolov8n-pose.pt")  # YOLOv8-pose
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # --- Face Detection & Blurring ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_face = mp_face.process(img_rgb)
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            x1, y1 = int(bboxC.xmin * w), int(bboxC.ymin * h)
            x2, y2 = x1 + int(bboxC.width * w), y1 + int(bboxC.height * h)
            
            # Ensure bounding box is within image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Apply Gaussian blur to the face region
            face_roi = img[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_roi, (35, 35), 0)
            img[y1:y2, x1:x2] = blurred_face

    # --- Pose Detection ---
    results_pose = pose_model(img, stream=True)
    for r in results_pose:
        img = r.plot()  # draw skeleton

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="webcam",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)