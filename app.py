# import part
import streamlit as st
from PIL import Image
import time

# function part
def generate_image_caption(image_path):
    """Generates a caption for the given image using a pre-trained model."""
    img2caption = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    result = img2caption(image_path)
    return result[0]['generated_text']

# text2story
def text2story(text):
    pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story_text = pipe(text)[0]['generated_text']
    return story_text




# main part
# App title
st.title("Assignment")

# Write some text
st.write("Image to Story")

# File uploader for image and audio
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# Display image with spinner
if uploaded_image is not None:
    with st.spinner("Loading image..."):
        time.sleep(1)  # Simulate a delay
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        caption = generate_image_caption(uploaded_image)
        st.write("Generated Caption: {caption}")



