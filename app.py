import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import plotly.express as px
import pandas as pd
import datetime
import base64
import os
import tempfile

# ==========================================
# ⚙️ SYSTEM CONFIGURATION
# ==========================================
# Assuming these are in the same folder
MODEL_PATH = "weights.pt" 
LOGO_PATH = "logo.png" 

# 🎛️ HARDCODED DETECTION & VISUAL SETTINGS
CONF_THRESH = 0.30       #
IOU_THRESH = 0.90        #
DRAW_LABELS = False      #
DRAW_COUNT_TAG = True    #
ANNOTATION_OPACITY = 0.8 #
# ==========================================

# ==========================================
# 💾 PERSISTENT GLOBAL STATE
# ==========================================
if 'global_total_count' not in st.session_state:
    st.session_state.global_total_count = 0
if 'history_data' not in st.session_state:
    st.session_state.history_data = pd.DataFrame([{"Time": datetime.datetime.now().strftime("%H:%M:%S"), "Total Count": 0}])
if 'current_run_count' not in st.session_state:
    st.session_state.current_run_count = 0

# 1. Initialize Model and Annotators
@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)

model = load_yolo_model()
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=1)

def get_base64_logo(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

def update_global_state(new_count):
    if new_count > 0:
        st.session_state.global_total_count += new_count
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        new_row = pd.DataFrame([{"Time": current_time, "Total Count": st.session_state.global_total_count}])
        st.session_state.history_data = pd.concat([st.session_state.history_data, new_row], ignore_index=True)
    st.session_state.current_run_count = new_count

def annotate_frame(frame, detections):
    annotated = frame.copy()
    if len(detections) > 0:
        overlay = frame.copy()
        overlay = box_annotator.annotate(scene=overlay, detections=detections)
        
        labels = []
        for i, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
            parts = []
            if DRAW_COUNT_TAG: parts.append(f"#{i+1}")
            if DRAW_LABELS: parts.append(f"{model.names[class_id]} {conf:.2f}")
            labels.append(" ".join(parts).strip())
            
        overlay = label_annotator.annotate(scene=overlay, detections=detections, labels=labels)
        cv2.addWeighted(overlay, ANNOTATION_OPACITY, annotated, 1 - ANNOTATION_OPACITY, 0, annotated)
    return annotated

# ==========================================
# 🎨 UI LAYOUT
# ==========================================
st.set_page_config(page_title="Sleeve Counting System", layout="wide")

# Hide Streamlit Branding (Fixed Parameter)
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 32px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# 1. HEADER
logo_b64 = get_base64_logo(LOGO_PATH)
st.markdown(f"""
<div style="display: flex; align-items: center; justify-content: space-between; padding: 25px; background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%); border-radius: 12px; margin-bottom: 20px; border: 1px solid #dcdde1; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
    <div style="flex: 1;"><img src="{logo_b64}" style="height: 70px;"></div>
    <div style="flex: 2; text-align: center;">
        <h1 style="margin: 0; color: #2c3e50; font-family: sans-serif; font-size: 32px; letter-spacing: 1px;">SLEEVE COUNTING SYSTEM</h1>
        <p style="margin: 5px 0 0 0; color: #7f8c8d; font-family: sans-serif; font-size: 14px; font-weight: bold;">INDUSTRIAL VISION ANALYTICS AND COUNTING MODULE</p>
    </div>
    <div style="flex: 1;"></div>
</div>
""", unsafe_allow_html=True)

# 2. STATUS DASHBOARD
col_target, col_total = st.columns(2)
with col_target:
    target_count = st.number_input("🎯 Target Packing Count", min_value=1, value=100)
    
    # Status Message Logic (Dynamic HTML)
    if st.session_state.global_total_count == target_count:
        st.markdown(f"<h3 style='color: #27ae60; margin-top: 10px;'>✅ TARGET COUNT ({target_count}) REACHED : READY FOR PACKAGING</h3>", unsafe_allow_html=True)
    elif st.session_state.global_total_count > target_count:
        st.markdown(f"<h3 style='color: #c0392b; margin-top: 10px;'>⚠️ OVER COUNT: Detected {st.session_state.global_total_count} (Target: {target_count})</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: #f39c12; margin-top: 10px;'>⏳ IN PROGRESS: {st.session_state.global_total_count} / {target_count}</h3>", unsafe_allow_html=True)

with col_total:
    st.metric("📦 OVERALL TOTAL DETECTED (Accumulated)", st.session_state.global_total_count)
    st.markdown(f"<h4 style='color: #e67e22; border-left: 4px solid #e67e22; padding-left: 10px;'>🔍 CURRENT VALUE DETECTED: <b>{st.session_state.current_run_count}</b></h4>", unsafe_allow_html=True)

st.divider()

# 3. WORKSPACE
col_input, col_output = st.columns(2)

with col_input:
    tab_img, tab_cam, tab_vid = st.tabs(["📷 Upload Image", "🔴 Live Camera", "🎥 Video Feed"])
    
    with tab_img:
        img_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
        if img_file and st.button("RUN IMAGE DETECTION", type="primary"):
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH)[0]
            detections = sv.Detections.from_ultralytics(results)
            st.session_state.last_processed = annotate_frame(frame, detections)
            update_global_state(len(detections))
            st.rerun()

    with tab_cam:
        cam_file = st.camera_input("Take a snapshot")
        if cam_file:
            file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH)[0]
            detections = sv.Detections.from_ultralytics(results)
            st.session_state.last_processed = annotate_frame(frame, detections)
            update_global_state(len(detections))
            st.rerun()

    with tab_vid:
        vid_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        if vid_file and st.button("START VIDEO TRACKING"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid_file.read())
            tracker = sv.ByteTrack()
            unique_ids = set()
            frame_generator = sv.get_video_frames_generator(tfile.name)
            
            for frame in frame_generator:
                results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)
                for tid in detections.tracker_id: unique_ids.add(tid)
            
            update_global_state(len(unique_ids))
            st.success(f"Video Tracked! Added {len(unique_ids)} unique items.")
            st.rerun()

with col_output:
    if 'last_processed' in st.session_state:
        st.image(st.session_state.last_processed, channels="BGR", use_container_width=True, caption="Processed Result")
    else:
        st.info("Visual Output will appear here after detection.")

# 4. ANALYTICS FOOTER
st.divider()
footer_graph, footer_big = st.columns([3, 1])

with footer_graph:
    fig = px.line(st.session_state.history_data, x="Time", y="Total Count", title="Production Timeline", markers=True)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with footer_big:
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 300px; background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); border-radius: 12px; border: 2px solid #bdc3c7; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <h2 style="margin: 0; color: #7f8c8d; font-family: sans-serif; text-transform: uppercase; font-size: 16px; letter-spacing: 2px;">Total Counted</h2>
        <h1 style="margin: 10px 0 0 0; color: #2c3e50; font-family: sans-serif; font-size: 80px; font-weight: 900;">{st.session_state.global_total_count}</h1>
    </div>
    """, unsafe_allow_html=True)

# Emergency Reset
if st.button("Emergency System Reset (Clear All Counts)"):
    st.session_state.global_total_count = 0
    st.session_state.history_data = pd.DataFrame([{"Time": datetime.datetime.now().strftime("%H:%M:%S"), "Total Count": 0}])
    st.session_state.current_run_count = 0
    if 'last_processed' in st.session_state: del st.session_state.last_processed
    st.rerun()