import streamlit as st
import requests
import os
from pathlib import Path

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Audio to Guitar Tab Converter",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    .stApp {
        background: linear-gradient(135deg, #14000d 0%, #d10404 100%);
    }
    .main-header {
        text-align: center;
        color: white;
        font-size: 3rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .success-box {
        background: rgba(76, 175, 80, 0.2);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #4CAF50;
        margin: 1rem 0;
    }
    .error-box {
        background: rgba(244, 67, 54, 0.2);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f44336;
        margin: 1rem 0;
    }
    .info-box {
        background: rgba(33, 150, 243, 0.2);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2196F3;
        margin: 1rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def send_audio_to_backend(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(
            f"{BACKEND_URL}/generate-tab",
            files=files,
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "detail": response.json().get("detail", "Unknown error")}
    
    except requests.exceptions.Timeout:
        return {"status": "error", "detail": "Timeout - file too long or server not responding"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "detail": "Cannot connect to backend. Check if server is running."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def main():
    st.markdown('<h1 class="main-header">Audio to Guitar Tab Converter</h1>', unsafe_allow_html=True)
    
    backend_status = check_backend_status()
    
    if not backend_status:
        st.markdown("""
        <div class="error-box">
            <strong>⚠️ Attetnion ⚠️</strong><br>
            Backend not available<br>
            Make sure the backend server is running on port 8000.<br>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="info-box">
        Backend connected.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader(
        "Select audio file:",
        type=['wav', 'mp3'],
        help="Supported formats: WAV, MP3"
    )
    
    if uploaded_file is not None:
        file_info = {
            "Name": uploaded_file.name,
            "Size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "Type": uploaded_file.type
        }
        
        st.write("**File Information:**")
        for key, value in file_info.items():
            st.write(f"{key}: `{value}`")
        
        st.audio(uploaded_file, format=uploaded_file.type)
        
        st.markdown("---")
        
        if st.button("Generate Tablature", type="primary", use_container_width=True):
            with st.spinner("Processing audio..."):
                result = send_audio_to_backend(uploaded_file)
                
                if result.get("status") == "success":
                    st.markdown('<div class="success-box">Tablature generated.</div>', 
                                unsafe_allow_html=True)
                    
                    
                    st.subheader("Tablature")
                    tab_text = result.get("tab", "")
                    
                    st.text_area(
                        "Generated tablature",
                        tab_text,
                        height=400,
                        label_visibility="collapsed"
                    )
                    
                    download_filename = f"{Path(uploaded_file.name).stem}_tab.txt"
                    st.download_button(
                        label="Download Tabs",
                        data=tab_text,
                        file_name=download_filename,
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    st.session_state['last_tab'] = tab_text
                    st.session_state['last_filename'] = uploaded_file.name
                    
                else:
                    st.markdown(f'<div class="error-box">{result.get("detail", "Error xD")}</div>', 
                                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
