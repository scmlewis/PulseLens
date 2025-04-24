# import part
import streamlit as st
from transformers import pipeline

# function part
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




def main():
    # App title
    st.title("Streamlit Demo on Hugging Face")

    # Write some text
    st.write("Welcome to a demo app showcasing basic Streamlit components!")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        print(uploaded_file)

        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)


        #Stage 1: Image to Text
        st.text('Processing img2text...')
        image_caption = generate_image_caption(uploaded_image.name)
        st.write(image_caption)

if __name__ == "__main__":
    main()