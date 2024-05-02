import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize the pretrained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "About"])

    if app_mode == "Home":
        st.title("AI Text Summarization")
        text = st.text_area("Enter the text you want to summarize:")
        if st.button("Summarize"):
            if text.strip() != "":
                summarize(text)
            else:
                st.warning("Please enter some text to summarize.")
    elif app_mode == "About":
        st.title("About")
        st.write("This is a simple web app for text summarization using AI.")
        st.write("It uses a pretrained T5 model to generate summaries.")
        st.write("Enter your text in the text area on the Home page and click the 'Summarize' button to generate a summary.")

def summarize(text):
    preprocessed_text = text.strip().replace('\n','')
    t5_input_text = 'summarize: ' + preprocessed_text
    tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512).to(device)
    # Summarize
    summary_ids = model.generate(tokenized_text, min_length=30, max_length=120)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    st.subheader("Summary:")
    st.write(summary)

if __name__ == "__main__":
    main()
