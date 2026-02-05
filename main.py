import streamlit as st
import tempfile
import cv2
import detector
import yaml

def main():
    st.title("Object Tracking Dashboard YOLO")
    st.sidebar.title("Settings")
    # st.markdown("""<style>
    #             [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width: 488px;}
    #             [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{width: 488px; margin-left: -488px;}
    #             </style>""",
    #             unsafe_allow_html=True)
    
    st.sidebar.markdown('---')
    confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.25)
    st.sidebar.markdown('---')

    # Checkboxers
    save_video = st.sidebar.checkbox("Save Video")
    enable_GPU = st.sidebar.checkbox("Enable GPU")
    custom_classes = st.sidebar.checkbox("Use Custom Classes")
    assigned_class_id = []
    st.sidebar.markdown('---')

    # Custom classes
    with open("coco.yaml", "r") as file:
        classes_dict = yaml.safe_load(file)["classes"]

    if custom_classes:
        assigned_class = st.sidebar.multiselect("Select The Custom Classes", classes_dict, default="person")
        for each in assigned_class:
            assigned_class_id.append(classes_dict.index(each))

    # Caso Custom classes não seja selecionada, utilizar classe "padrão" de pessoas
    else:
        assigned_class_id = [0]

    # Uploading out video
    DEMO_VIDEO = "demo_videos/demo.mp4"
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])
    tfflie = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    # Get out input video here
    if not video_file_buffer:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tfflie.name = DEMO_VIDEO
        dem_vid = open(tfflie.name, "rb")
        demo_bytes = dem_vid.read()

        st.sidebar.text("Input Video")
        st.sidebar.video(demo_bytes)

    else:
        tfflie.write(video_file_buffer.read())
        dem_vid = open(tfflie.name, "rb")
        demo_bytes = dem_vid.read()

        st.sidebar.text("Input Video")
        st.sidebar.video(demo_bytes)

    print(tfflie.name)

    stFrame = st.empty()
    st.sidebar.markdown('---')

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Tracked Objects**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Width**")
        kpi3_text = st.markdown("0")

    # # Carrengando função do video que foi estudado (não implementado nesse código)
    # load_yolor_and_process_each_frame = (tfflie.name, enable_GPU, confidence, assigned_class_id, kpi1_text, kpi2_text, kpi3_text, stframe)

    detector.detection(tfflie.name, confidence, save_video, enable_GPU, assigned_class_id, kpi1_text, kpi2_text, kpi3_text, stFrame)

    st.text("Video is Processed")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
