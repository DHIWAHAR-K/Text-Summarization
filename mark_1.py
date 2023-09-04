import streamlit as st
from transformers import BertForMaskedLM, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load BERT for masked language modeling
bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForMaskedLM.from_pretrained(bert_model_name)

# Load GPT-2 for text generation
gpt2_model_name = "gpt2-medium"  # You can choose a different GPT-2 model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

def main():
    st.title("Text Reformulation App")

    # User input for text reformulation
    input_text = st.text_input("Enter a sentence:")

    if input_text:
        # Use BERT to understand the context
        input_ids = bert_tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
        masked_index = torch.where(input_ids == bert_tokenizer.mask_token_id)[1]
        with torch.no_grad():
            bert_output = bert_model(input_ids).logits
            predicted_tokens = torch.argmax(bert_output, dim=2)

        # Convert predicted BERT tokens to text
        predicted_words = bert_tokenizer.convert_ids_to_tokens(predicted_tokens[0].tolist())

        # Generate text using GPT-2 based on the predicted words
        prompt_text = " ".join(predicted_words)
        generated_text = gpt2_model.generate(
            gpt2_tokenizer.encode(prompt_text, return_tensors="pt"),
            max_length=50,  # Adjust as needed
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

        # Convert generated IDs to text
        generated_text = gpt2_tokenizer.decode(generated_text[0], skip_special_tokens=True)

        st.subheader("Input Text:")
        st.write(input_text)

        st.subheader("Reformed Text:")
        st.write(generated_text)

if __name__ == "__main__":
    main()