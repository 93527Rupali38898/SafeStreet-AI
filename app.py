import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
from transformers import VideoMAEForVideoClassification
from torchvision import transforms

# --- Page Config ---
st.set_page_config(page_title="SafeStreet AI", page_icon="🚨", layout="centered")

st.title("🚨 SafeStreet AI - Autonomous Anomaly Detection")
st.markdown("**Edge-Optimized CCTV Surveillance System**")
st.write("Upload a street CCTV clip (.mp4) and let the VideoMAE Transformer detect violence/anomalies in real-time.")

# --- 1. Load Model (Cached) ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = 'SafeStreet_VideoMAE.pth'
    
    if not os.path.exists(weights_path):
        st.error(f"⚠️ Error: '{weights_path}' not found! Please ensure it is in the same folder as app.py")
        st.stop()

    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True
    )
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval() 
    return model, device

model, device = load_model()

# --- 2. Pure CCTV Preprocessing (Consecutive Frames) ---
def process_video_cctv(video_path, num_frames=16, sz=224):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 16:
        st.error(f"Video is too short! ({total_frames} frames). Need at least 16.")
        cap.release()
        return None
        
    # Extract 3 continuous clips (Start, Middle, End) to cover the whole event
    start_points = [
        0,                                         # Clip 1: Start
        max(0, total_frames // 2 - (num_frames)),  # Clip 2: Middle
        max(0, total_frames - (num_frames * 2))    # Clip 3: End
    ]
    
    all_chunks = []
    for start_idx in start_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        
        # Read 16 CONSECUTIVE frames (Standard for 15-30 FPS CCTV cameras)
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret: 
                break
                
            frame = cv2.resize(frame, (sz, sz))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        # Pad the last frame if the video ended slightly early
        while len(frames) > 0 and len(frames) < num_frames:
            frames.append(frames[-1])
            
        if len(frames) == num_frames:
            all_chunks.append(frames)
            
    cap.release()
    
    if not all_chunks:
        return None
        
    video_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((sz, sz), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert chunks to Tensors
    tensor_batches = []
    for chunk in all_chunks:
        tensor_frames = [video_transforms(f) for f in chunk]
        tensor_batches.append(torch.stack(tensor_frames))
        
    return torch.stack(tensor_batches) # Shape: (3, 16, 3, 224, 224)

# --- 3. UI and Inference ---
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close() 
    
    st.video(uploaded_file)
    
    if st.button("🔍 Analyze Video", use_container_width=True):
        with st.spinner("🧠 AI is scanning CCTV feed (Start, Middle, End sequences)..."):
            input_tensor = process_video_cctv(tfile.name)
            
            if input_tensor is not None:
                input_tensor = input_tensor.to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    logits = outputs.logits 
                    probabilities = torch.softmax(logits, dim=1)
                    
                    # Find the chunk with the HIGHEST probability of violence
                    anomaly_probs = probabilities[:, 1] 
                    max_anomaly_prob, max_idx = torch.max(anomaly_probs, dim=0)
                    
                    confidence = max_anomaly_prob.item() * 100
                    
                    # Threshold: 50%
                    predicted_class = 1 if confidence > 50.0 else 0
                
                st.markdown("---")
                if predicted_class == 1:
                    st.error(f"🔴 **VIOLENCE / ANOMALY DETECTED!** (Confidence: {confidence:.2f}%)")
                    st.warning("Action: Generating alert for authorities...")
                else:
                    normal_conf = (100 - confidence)
                    st.success(f"🟢 **NORMAL CROWD.** (Confidence: {normal_conf:.2f}%)")
                    st.info("Action: Maintaining standard 2 FPS observation.")
                    
    try:
        os.remove(tfile.name)
    except:
        pass