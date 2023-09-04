import streamlit as st
from transformers import pipeline

st.title("Text Paraphrasing App")

# Load the paraphrasing pipeline
paraphrase_pipeline = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")

input_text = st.text_input("Enter a sentence:")

if input_text:
    # Paraphrase the input text using the pipeline
    paraphrased_text = paraphrase_pipeline(input_text, max_length=100)[0]['generated_text']

    st.subheader("Input Text:")
    st.write(input_text)

    st.subheader("Paraphrased Text:")
    st.write(paraphrased_text)