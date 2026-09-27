import streamlit as st
from pathlib import Path
from app import detect, Settings, parse_rectangle

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Red-Light Violation Detector", 
    page_icon="🚦", 
    layout="wide"
)

st.title("🚦 Red-Light Violation Detector")
st.markdown("Configure the YOLO tracking model and run it directly from this interface.")

# ============================================================
# FIND LOCAL VIDEOS
# ============================================================
APP_DIR = Path(__file__).parent
video_files = list(APP_DIR.glob("*.mp4")) + list(APP_DIR.glob("*.avi"))
video_names = [f.name for f in video_files]

# ============================================================
# UI LAYOUT
# ============================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("⚙️ Settings")
    
    if not video_names:
        st.warning("No .mp4 or .avi videos found in this folder. Please add one!")
        input_video = None
    else:
        # Try to default to the specific mixkit video if it exists
        default_index = 0
        target_video = "mixkit-the-streets-of-los-angeles-4243-hd-ready.mp4"
        if target_video in video_names:
            default_index = video_names.index(target_video)
            
        input_video = st.selectbox("🎥 Select Input Video", video_names, index=default_index)
        
        # Get the full absolute path for the selected video
        if input_video:
            input_video_path = APP_DIR / input_video
        
    model_name = st.text_input("🧠 Model Name", value="yolo11n.pt")
    
    st.divider()
    st.subheader("📍 Calibration Coordinates")
    st.caption("Format: X1, Y1, X2, Y2")
    
    stop_line = st.text_input("🛑 Stop Line", value="842, 497, 396, 503")
    light_roi = st.text_input("🚥 Traffic Light ROI", value="673, 281, 683, 306")
    
    st.divider()
    st.subheader("🎛️ Tracking Parameters")
    
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.3)
    min_track_frames = st.number_input("Minimum Track Frames", min_value=1, value=5)
    
    st.write("")
    run_button = st.button("🚀 Start Detection", type="primary", use_container_width=True)

with col2:
    st.header("📺 Original Video")
    if input_video:
        st.video(str(input_video_path))
    else:
        st.info("Select a video to preview it here.")

# ============================================================
# RUN DETECTION
# ============================================================
if run_button:
    if not input_video:
        st.error("Cannot run without an input video.")
    else:
        st.info("Starting OpenCV tracking window... (Look for a new window on your computer!)")
        try:
            # Prepare settings
            settings = Settings(
                input_video=input_video_path,
                model=model_name,
                stop_line=parse_rectangle(stop_line),
                light_roi=parse_rectangle(light_roi),
                confidence=float(confidence),
                min_track_frames=int(min_track_frames),
                display=True,
                output_video=None,
                trajectory_csv=None
            )
            
            # Run the detection algorithm from app.py
            crossed, violations = detect(settings)
            
            st.success("✅ Detection finished!")
            
            st.divider()
            st.subheader("📊 Results")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Total Vehicles Crossed", crossed)
            metric_col2.metric("🔴 Red-Light Violations", violations, delta=f"+{violations}" if violations > 0 else None, delta_color="inverse")
            
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
