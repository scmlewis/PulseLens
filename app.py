# import part
import streamlit as st
from transformers import pipeline

# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story_text = pipe(text)[0]['generated_text']
    return story_text

# text2audio
def text2audio(story_text):
    pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = pipe(story_text)
    return audio_data


def main():
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    uploaded_file = st.file_uploader("Select an Image...")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


        #Stage 1: Image to Text
        st.text('Processing img2text...')
        scenario = img2text(uploaded_file.name)
        st.write(scenario)

if __name__ == "__main__":
    main()