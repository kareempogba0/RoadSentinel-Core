import os
import streamlit as st
import time
from PIL import Image
from src.predictor import RoadSentinelPredictor

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="RoadSentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #FF4B4B; color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. LOAD AI ENGINE
# ---------------------------------------------------------
@st.cache_resource
def get_ai_engine():
    return RoadSentinelPredictor('models/best_model_mobilenet.keras')


try:
    predictor = get_ai_engine()
except Exception as e:
    st.error(f"⚠️ Model loading failed. Error details: {e}")
    st.write("Current working directory:", os.getcwd())
    st.write("Files in 'models' folder:", os.listdir('models') if os.path.exists('models') else "Folder not found")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    # FIXED: Removed Markdown syntax []() from the string
    st.image("https://cdn-icons-png.flaticon.com/512/1167/1167993.png", width=80)
    st.title("RoadSentinel")
    st.markdown("**Intelligent Traffic Safety System**")
    st.markdown("---")

    app_mode = st.selectbox("Choose Mode", ["📸 Image Analysis", "🎥 Live Stream Simulation"])

    st.markdown("### ⚙️ Calibration")
    # UPDATED: Slider focused on the high range (0.90 - 1.00)
    threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.5, 0.01)
    # st.caption("Adjust to filter out safe roads appearing as accidents.")
    st.caption(f"Model: MobileNetV2 (Transfer Learning)")

    st.markdown("---")
    st.info("System Status: **ONLINE** 🟢")

# ---------------------------------------------------------
# 4. MAIN APP CONTENT
# ---------------------------------------------------------

if app_mode == "📸 Image Analysis":
    st.subheader("Incident Detection Interface")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            # FIXED: Updated parameter based on warning (use_container_width -> width='stretch')
            st.image(image, caption="Input Feed", width="stretch")

    with col2:
        if uploaded_file:
            st.write("### AI Analysis Results")

            with st.spinner("Analyzing pixels..."):
                time.sleep(0.5)
                result = predictor.predict(uploaded_file, threshold=threshold)

            if "error" in result:
                st.error(f"Analysis Failed: {result['error']}")
            else:
                # DEBUG INFO: Show the raw score
                raw_score = result['raw_score']
                st.info(f"🧠 Raw Score: **{raw_score:.4f}** | Threshold: **{threshold:.3f}**")

                # Result Banner
                if result['is_safe']:
                    st.success(f"## ✅ {result['label']}")
                    st.metric("Safety Confidence", f"{result['confidence'] * 100:.1f}%")
                    st.markdown("Traffic flow appears normal.")
                else:
                    st.error(f"## 🚨 {result['label']} DETECTED")
                    st.metric("Accident Probability", f"{result['confidence'] * 100:.1f}%")

                    st.markdown("#### ⚠️ Recommended Actions:")
                    st.warning("1. 🚑 Dispatch Emergency Services")
                    st.warning("2. 🚓 Alert Traffic Control")

elif app_mode == "🎥 Live Stream Simulation":
    st.subheader("Live Traffic Monitoring (Demo)")
    st.warning("Connect RTSP stream to enable real-time video feed.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Cameras", "12", "+2")
    c2.metric("Incidents Today", "4", "High")
    c3.metric("Avg Response Time", "3 min", "-30s")

    # FIXED: Removed Markdown syntax []() and updated width parameter
    st.image(
        "https://media.istockphoto.com/id/483647334/photo/traffic-jam-on-highway.jpg?s=612x612&w=0&k=20&c=L0kQouK44i-Z6QjC2VwYQJ0w_T0w_T0w.jpg",
        caption="Camera Feed 04 [LIVE]", width="stretch")