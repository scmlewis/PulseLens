# import part
import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# function part
def generate_image_caption(image):
    """Generates a caption for the given image using a pre-trained model."""
    img2caption = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
       
    # Generate caption
    result = img2caption(image)
    return result[0]['generated_text']

# text2story
def text2story(text):
    pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story_text = pipe(text)[0]['generated_text']
    return story_text

def main():
    # App title
    st.title("Streamlit Demo on Hugging Face")

    # Write some text
    st.write("Welcome to a demo app showcasing basic Streamlit components!")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Stage 1: Image to Text
        st.text('Processing img2text...')
        image_caption = generate_image_caption(image)
        st.write(image_caption)

if __name__ == "__main__":
    main()