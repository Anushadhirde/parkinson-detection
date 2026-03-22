import streamlit as st
import os
import time
import wave
import base64
from predict.svm.predict import run_prediction_detailed

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="Parkinson Detection", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions ---
def get_audio_info(filepath):
    try:
        with wave.open(filepath, 'rb') as audio_file:
            rate = audio_file.getframerate()
            frames = audio_file.getnframes()
            duration = frames / float(rate)
            return duration, rate
    except Exception:
        return 0.0, 0

def render_circular_progress(percentage, is_healthy):
    color = "#198754" if is_healthy else "#FF6B6B"
    deg = int(percentage * 360)
    return f"""
    <div style="display: flex; justify-content: center; align-items: center; margin: 1.5rem 0;">
        <div style="
            position: relative;
            width: 140px;
            height: 140px;
            border-radius: 50%;
            background: conic-gradient({color} {deg}deg, #e2e8f0 0deg);
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        ">
            <div style="
                position: absolute;
                width: 110px;
                height: 110px;
                background-color: white;
                border-radius: 50%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                font-weight: 800;
                font-size: 1.8rem;
                color: {color};
            ">
                {int(percentage * 100)}%
                <span style="font-size: 0.65rem; font-weight: 700; color: #6c757d; text-transform: uppercase; margin-top: -5px;">Confidence</span>
            </div>
        </div>
    </div>
    """

# --- CSS INJECTION ---
st.markdown("""
<style>
    /* 1. Visual Upgrades: Gradient Background & Colors */
    :root {
        --primary: #008080;
        --accent: #FF6B6B;
        --success: #198754;
        --card-bg: #FFFFFF;
        --text-main: #2b2b2b;
        --text-muted: #6c757d;
    }

    .stApp {
        background: linear-gradient(to bottom right, #f0f8f8, #ffffff);
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        color: var(--text-main);
    }

    .block-container {
        max-width: 1100px !important;
        padding-top: 2.5rem;
        padding-bottom: 6rem;
    }

    /* Gradient Header */
    .custom-header {
        background: linear-gradient(135deg, var(--primary), #005c5c);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 128, 128, 0.2);
    }
    
    .custom-header h1 {
        color: white;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .custom-header p {
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-size: 1.15rem;
        font-weight: 400;
    }

    /* Cards with rounded corners and soft shadows */
    .info-box, .metric-card {
        background-color: var(--card-bg);
        border-radius: 20px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .info-box {
        padding: 1.8rem;
        margin-bottom: 1.5rem;
    }

    /* Upload Zone Styling: Circular upload icon logic */
    /* Streamlit uploader customization */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #b2d8d8 !important;
        border-radius: 20px !important;
        background-color: rgba(0, 128, 128, 0.03);
        padding: 3rem 1rem;
        transition: all 0.3s ease;
        position: relative;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(0, 128, 128, 0.08);
        border-color: var(--primary) !important;
        transform: translateY(-2px);
    }

    /* Upload badge container */
    .upload-badge {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 5px solid var(--primary);
        margin-bottom: 1.5rem;
    }

    /* Results section formatting */
    .result-pulse {
        text-align: center;
        padding: 2.5rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        font-weight: 800;
        font-size: 1.6rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        position: relative;
        overflow: hidden;
    }

    .result-healthy {
        background: linear-gradient(145deg, #ffffff, #f0fdf4);
        border: 3px solid var(--success);
        color: var(--success);
        animation: pulse-green 2s infinite;
    }
    
    .result-parkinson {
        background: linear-gradient(145deg, #ffffff, #fef2f2);
        border: 3px solid var(--accent);
        color: var(--accent);
        animation: pulse-red 2s infinite;
    }

    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(25, 135, 84, 0); }
        100% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0); }
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 107, 107, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: -10px;
    }

    /* Audio Player */
    audio {
        width: 100%;
        margin-top: 1rem;
        border-radius: 50px;
        height: 50px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    .audio-specs {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 0.8rem;
        font-size: 0.85rem;
        color: var(--text-muted);
        font-weight: 600;
    }

    /* Footer Details */
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-top: 1px solid #e2e8f0;
        padding: 1rem 0;
        text-align: center;
        z-index: 999;
        font-size: 0.95rem;
        color: var(--text-muted);
    }
    .footer-divider {
        width: 50px;
        height: 3px;
        background-color: var(--primary);
        margin: 0 auto 0.5rem auto;
        border-radius: 5px;
    }

    /* Sidebar Enhancements */
    [data-testid="stSidebar"] {
        background-color: #f7fbfc;
        border-right: 1px solid #e0ebeb;
    }
    .sidebar-stats-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary);
    }
    .sidebar-title {
        color: var(--primary);
        font-weight: 800;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }
    .bullet-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .bullet-list li {
        position: relative;
        padding-left: 1.5rem;
        margin-bottom: 0.5rem;
        color: #555;
        font-size: 0.9rem;
    }
    .bullet-list li::before {
        content: '🔹';
        position: absolute;
        left: 0;
        top: 2px;
        font-size: 0.8rem;
    }

    /* Spin Animation */
    @keyframes spin-grow {
        0% { transform: rotate(0deg) scale(1); }
        50% { transform: rotate(180deg) scale(1.1); }
        100% { transform: rotate(360deg) scale(1); }
    }
    .spinning-element {
        font-size: 4rem;
        animation: spin-grow 1.5s linear infinite;
        display: inline-block;
        color: var(--primary);
    }

</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div style="text-align: center; font-size: 3.5rem; margin-bottom: 1rem; margin-top: 1rem;">🫁</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">🎙️ Recording Guide</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-stats-card">
        <ul class="bullet-list">
            <li>Find a quiet environment.</li>
            <li>Mic 10-15 cm from mouth.</li>
            <li>Take a deep breath.</li>
            <li>Say "Aaaah", "Eeeee", or "Ooooo" steadily for 3-5 seconds.</li>
            <li>Save and upload as .wav.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">⚙️ AI Model Details</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-stats-card">
        <ul class="bullet-list">
            <li><b>Classifier:</b> SVM (RBF Kernel)</li>
            <li><b>Features:</b> Jitter, Shimmer, HNR, MFCCs</li>
            <li><b>Analysis:</b> Overlapping window segmentation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">ℹ️ Disclaimer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-stats-card" style="border-left-color: var(--accent); background-color: #fffaf9;">
        <span style="color: var(--accent); font-weight: bold;">Research Prototype</span><br>
        <span style="font-size: 0.85rem; color: #666;">Not validated for medical diagnosis. See a healthcare professional for clinical evaluation.</span>
    </div>
    """, unsafe_allow_html=True)

# --- HEADER APP ---
st.markdown("""
<div class="custom-header">
    <h1>Parkinson Detection Tool</h1>
    <p>Upload a vocal recording for automated acoustic analysis</p>
</div>
""", unsafe_allow_html=True)

# Main Application Layout
col1, space, col2 = st.columns([1.1, 0.05, 1.25])

with col1:
    st.markdown("<h3 style='color: var(--primary); font-weight: 800;'><span style='background: var(--primary); color: white; border-radius: 50%; width: 28px; height: 28px; display: inline-flex; align-items: center; justify-content: center; font-size: 1rem; margin-right: 8px;'>1</span> Upload Audio</h3>", unsafe_allow_html=True)
    
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    uploaded_file = st.file_uploader(
        "Upload a .wav file (Max 200MB)", 
        type=["wav"], 
        label_visibility="collapsed",
        key=st.session_state["file_uploader_key"],
    )

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        # File Validation
        if not uploaded_file.name.lower().endswith('.wav'):
            st.error("Invalid file format. Please upload a .WAV file.")
            uploaded_file = None
        elif file_size_mb > 200:
            st.error("File is too large. Maximum size is 200MB.")
            uploaded_file = None
        else:
            # Custom Upload Badge with Clear Button
            btn_col1, btn_col2 = st.columns([4, 1.5])
            with btn_col1:
                st.markdown(f"""
                <div class="upload-badge">
                    <div>
                        <div style="font-weight: 700; color: var(--primary);">{uploaded_file.name}</div>
                        <div style="color: white; background: var(--success); padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; display: inline-block; margin-top: 4px; font-weight: bold;">
                            {file_size_mb:.2f} MB
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with btn_col2:
                # Vertical alignment trick
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state["file_uploader_key"] += 1
                    if 'analyzed_file' in st.session_state:
                        del st.session_state['analyzed_file']
                    st.rerun()
            
            # Analyze Action
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🔍 Analyze Acoustic Signatures", type="primary", use_container_width=True)
            
            if analyze_button:
                st.session_state['analyzed_file'] = uploaded_file.name

with col2:
    st.markdown("<h3 style='color: var(--primary); font-weight: 800;'><span style='background: var(--primary); color: white; border-radius: 50%; width: 28px; height: 28px; display: inline-flex; align-items: center; justify-content: center; font-size: 1rem; margin-right: 8px;'>2</span> Preview & Results</h3>", unsafe_allow_html=True)
    
    if not uploaded_file:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 4rem 1.5rem; background: rgba(255,255,255,0.7); border: 2px dashed #cbd5e1; box-shadow: none !important;">
            <div style="font-size: 3.5rem; color: #94a3b8; margin-bottom: 1rem;">🎧</div>
            <h4 style="color: #64748b; margin: 0; font-weight: 600;">Awaiting Audio File</h4>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.5rem;">Upload a recording on the left to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        dur, rate = get_audio_info(filepath)
            
        # Preview Box with Custom Player Style
        st.markdown('<div class="info-box" style="padding: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("<b style='color: var(--primary); font-size: 1.1rem;'>Audio Preview</b>", unsafe_allow_html=True)
        st.audio(filepath, format="audio/wav")
        st.markdown(f"""
        <div class="audio-specs">
            <span>⏱️ Duration: {dur:.2f}s</span>
            <span>🎵 Sample Rate: {rate} Hz</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results Execution
        if st.session_state.get('analyzed_file') == uploaded_file.name:
            result_container = st.empty()
            
            # Fake processing with custom HTML spinner
            if analyze_button:
                result_container.markdown("""
                <div class="info-box" style="text-align: center; padding: 3rem;">
                    <div class="spinning-element">🎙️</div>
                    <h3 style="color: var(--primary); margin-top: 1.5rem;">Extracting Acoustics...</h3>
                    <p style="color: #6c757d;">Running SVM Predictive Model</p>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1.5)
            
            # Fetch Prediction
            try:
                prediction_data = run_prediction_detailed(filepath)
                result = prediction_data['result']
                confidence = prediction_data['confidence']
                details = prediction_data['details']
                
                if analyze_button:
                    st.toast("Analysis complete!", icon="✅")
                
                # Render Results HTML dynamically via result_container
                is_healthy = (result == "healthy")
                
                pulse_class = "result-healthy" if is_healthy else "result-parkinson"
                result_text = "NEGATIVE FOR PARKINSON'S" if is_healthy else "POTENTIAL INDICATIONS DETECTED"
                result_icon = "✓" if is_healthy else "⚠️"
                
                card_html = f"""
                <div class="result-pulse {pulse_class}">
                    <div class="result-icon">{result_icon}</div>
                    <div style="font-size: 1.4rem;">{result_text}</div>
                    {render_circular_progress(confidence, is_healthy)}
                </div>
                """
                
                result_container.markdown(card_html, unsafe_allow_html=True)
                
            except Exception as e:
                result_container.error(f"An error occurred during prediction: {e}")

# --- FOOTER ---
st.markdown("""
<div class="custom-footer">
    <div class="footer-divider"></div>
    <p style="margin: 0; font-weight: 500;">Developed by <span>The Sonics</span></p>
</div>
""", unsafe_allow_html=True)