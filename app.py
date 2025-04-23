import streamlit as st
from PIL import Image
import time

# App title
st.title("Streamlit Demo on Hugging Face")

# Write some text
st.write("Welcome to a demo app showcasing basic Streamlit components!")

# File uploader for image and audio
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

# Display image with spinner
if uploaded_image is not None:
    with st.spinner("Loading image..."):
        time.sleep(1)  # Simulate a delay
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Play audio with spinner
if uploaded_audio is not None:
    with st.spinner("Loading audio..."):
        time.sleep(1)  # Simulate a delay
        st.audio(uploaded_audio)

# Button interaction
if st.button("Click Me"):
    st.write("🎉 You clicked the button!")
