# import part
import streamlit as st
from PIL import Image
import time
from transformers import pipeline

# Load models (once globally)
img2caption = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
story_gen = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

# function part
def generate_image_caption(image):
    """Generates a caption for the given image using a pre-trained model."""
    result = img2caption(image)
    return result[0]['generated_text']

def text2story(text):
    """Generates a story from the given caption using a story generation model."""
    story_text = story_gen(text)[0]['generated_text']
    return story_text

# main part
# App title
st.title("Assignment")

# Write some text
st.write("Image to Story")

# File uploader for image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display image with spinner
if uploaded_image is not None:
    with st.spinner("Processing..."):
        time.sleep(1)  # Simulate a delay
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Generate caption
        caption = generate_image_caption(uploaded_image.name)
        st.write(f"**Generated Caption:** {caption}")
        
        # Button to generate story
        if st.button("Generate Story from Caption"):
            story = text2story(caption)
            st.markdown("**Generated Story:**")
            st.write(story)
