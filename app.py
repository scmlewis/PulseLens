import streamlit as st
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import numpy as np

def main():


    st.title("yelp2024fall Test")
    st.write("Enter a sentence for analysis:")

    user_input = st.text_input("")
    if user_input:
        # Approach: AutoModel
        model2 = AutoModelForSequenceClassification.from_pretrained("isom5240/CustomModel_yelp2025L1",
                                                                    num_labels=5)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        inputs = tokenizer(user_input,
                        padding=True,
                        truncation=True,
                        return_tensors='pt')

        outputs = model2(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = predictions.cpu().detach().numpy()
        # Get the index of the largest output value
        max_index = np.argmax(predictions)
        st.write(f"result (AutoModel) - Label: {max_index}")


if __name__ == "__main__":
    main()