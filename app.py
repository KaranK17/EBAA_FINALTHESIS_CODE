import streamlit as st
from liveFeed import showLiveFeed
from pre_process import showPreProcessedVideo
import os
import numpy as np

def remove_outliers_iqr(data, threshold=3.5):
    data = np.array(data)
    median = np.median(data)
    diff = np.abs(data - median)
    mad = np.median(diff)
    if mad == 0:
        return data.tolist()  # If MAD is 0, return original data

    # Calculate modified z-scores
    modified_z_scores = 0.6745 * diff / mad
    
    # Filter data based on threshold
    return data[modified_z_scores <= threshold].tolist()

with st.sidebar:
    st.header("About")
    justified_text = """
    <div style='text-align: justify'>
    This project is aimed at developing a robust system for analyzing
    the impact of advertisement holdings in real-time. The system
    records the total number of humans detected within a
    specified area, while specifically tracking those who are
    actively looking towards the advertisement. Utilizing advanced
    computer vision techniques like YOLO for human detection and
    face tracking, combined with head pose estimation, the system
    ensures accurate identification of individuals focused on the
    advertisement. This technology provides invaluable insights
    into the advertisement’s effectiveness, allowing advertisers
    to quantify the reach and engagement of their campaigns by
    distinguishing between casual passersby and those whose
    attention is genuinely captured by the advertisement.
    </div>
    """
    st.markdown(justified_text, unsafe_allow_html=True)
    st.image('people-crowd.jpg')

# Create the main tabs
tabs = st.tabs(["Live Feed", "Pre Process"])

# Tab 1: LiveFeed
with tabs[0]:
    st.header("Live Feed")
    run_live_feed = st.checkbox("Run Live Feed")
    
    if run_live_feed:
        st.write("Live Feed started...")
        frame_placeholder = st.empty()
        showLiveFeed(frame_placeholder, run_live_feed)

# Tab 2: PreProcess
with tabs[1]:
    st.header("Pre Process")
    uploaded_file = st.file_uploader("Upload your file here")
    
    if st.button("Show Results"):
        frame_placeholder = st.empty()
        if uploaded_file:
            file_path = os.path.join("videos", uploaded_file.name)
            
            # Write the file to the designated location
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            results = showPreProcessedVideo(file_path, frame_placeholder)
            total_humans = 0
            humans_looking = 0
            
            for face_ids in results:
                yaw = remove_outliers_iqr(results[face_ids]['yaw'])
                pitch = remove_outliers_iqr(results[face_ids]['pitch'])
                roll = remove_outliers_iqr(results[face_ids]['roll'])
                
                if len(yaw) > 10:
                    total_humans += 1
                    print(np.mean(yaw), face_ids)
                
                if np.mean(yaw) > 139 and np.mean(yaw) < 175:
                    humans_looking += 1
            
            st.write(f'Total unique faces in video are {total_humans}')
            st.write(f'Total faces looking towards camera are {humans_looking}')
        else:
            st.warning("Please upload a file first.")
