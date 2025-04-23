import streamlit as st
from PIL import Image
import time

st.title("📁 File Uploader Demo")

st.write("Upload an image or audio file, and we'll display or play it for you!")

uploaded_file = st.file_uploader("Choose an image or audio file", type=["png", "jpg", "jpeg", "mp3", "wav"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    with st.spinner("Processing..."):
        time.sleep(2)  # simulate processing time

    if "image" in file_type:
        st.write("### Uploaded Image:")
        image = Image.open(uploaded_file)
        st.image(image, caption="Your image", use_column_width=True)

    elif "audio" in file_type:
        st.write("### Uploaded Audio:")
        st.audio(uploaded_file, format=file_type)

    else:
        st.write("Unsupported file type.")
