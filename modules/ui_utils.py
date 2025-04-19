import streamlit as st
import requests
from streamlit_lottie import st_lottie
import time
import base64
from pathlib import Path

def load_lottie_url(url: str):
    """
    Load a Lottie animation from a URL
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def load_lottie_file(filepath: str):
    """
    Load a Lottie animation from a local file
    """
    with open(filepath, "r") as f:
        return json.load(f)

def display_lottie_animation(animation, key=None, height=300, width=None):
    """
    Display a Lottie animation in the Streamlit app
    """
    return st_lottie(
        animation,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",  # medium, high
        height=height,
        width=width,
        key=key,
    )

def add_bg_from_url(url):
    """
    Add a background image from URL
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({url});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def add_bg_from_local(image_file):
    """
    Add a background image from a local file
    """
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def apply_custom_css():
    """
    Apply custom CSS to improve the UI
    """
    st.markdown("""
    <style>
    /* Card effect for sections */
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        padding: 16px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
        border: 2px solid #4CAF50;
    }
    
    div.stButton > button:hover {
        background-color: white;
        color: #4CAF50;
    }
    
    /* Improve header styling */
    h1, h2, h3 {
        color: #1E88E5;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Add box shadow to plots */
    .stPlot {
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        border-radius: 5px;
        padding: 10px;
    }
    
    .stPlot:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    
    /* Improve sidebar styling */
    .css-1d391kg {
        background-color: #f5f5f5;
    }
    
    /* Add animation to expanders */
    .streamlit-expanderHeader {
        transition: background-color 0.3s;
    }
    .streamlit-expanderHeader:hover {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

def animated_progress(label, progress_value):
    """
    Display an animated progress bar
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(101):
        # Update progress bar
        progress_bar.progress(i)
        # Update status text
        status_text.text(f"{label}: {i}%")
        # Add a small delay
        time.sleep(0.01)
    
    progress_bar.progress(progress_value)
    status_text.text(f"{label}: {progress_value:.1f}%")

def create_animated_counter(start_val, end_val, prefix="", suffix="", duration=2.0):
    """
    Create an animated counter that counts from start_val to end_val
    """
    placeholder = st.empty()
    step = (end_val - start_val) / (duration * 10)  # 10 updates per second
    
    current = start_val
    for i in range(int(duration * 10)):
        current += step
        if current >= end_val:
            current = end_val
            placeholder.markdown(f"<h1 style='text-align: center;'>{prefix}{current:.1f}{suffix}</h1>", unsafe_allow_html=True)
            break
        
        placeholder.markdown(f"<h1 style='text-align: center;'>{prefix}{current:.1f}{suffix}</h1>", unsafe_allow_html=True)
        time.sleep(0.1)
    
    return placeholder

def create_animated_card(title, content, icon=None):
    """
    Create an animated card with title and content
    """
    card_html = f"""
    <div class="card" style="
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        padding: 20px;
        margin: 10px 0;
        animation: fadeIn 1s;
    ">
        <h3 style="color: #1E88E5;">{icon} {title}</h3>
        <p>{content}</p>
    </div>
    
    <style>
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    .card:hover {{
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        transform: translateY(-5px);
        transition: 0.3s;
    }}
    </style>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def create_tabs_with_animation(tab_names):
    """
    Create tabs with animation effect
    """
    tabs_html = """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
        transform: translateY(-5px);
    }
    </style>
    """
    
    st.markdown(tabs_html, unsafe_allow_html=True)
    return st.tabs(tab_names)