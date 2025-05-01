import streamlit as st
from transformers import pipeline

def main():
    sentiment_pipeline = pipeline(model="isom5240/2025SpringL2")

    st.title("Sentiment Analysis with HuggingFace Spaces")
    st.write("Enter a sentence to analyze its sentiment:")

    user_input = st.text_input("")
    if user_input:
        result = sentiment_pipeline(user_input)
        sentiment = result[0]["label"]
        confidence = result[0]["score"]

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()