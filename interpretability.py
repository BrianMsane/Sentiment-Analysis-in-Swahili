import streamlit as st
from streamlit import components
from transformers import (BertTokenizer, BertForSequenceClassification)
from transformer_intepret import SequenceClassificationExplainer


def load_model():
    model_name = 'drive/MyDrive/HuggingFace/SwahBERT'
    tokenizer = BertTokenizer.from_pretrain(model_name)
    model = BertForSequenceClassification.from_pretrain(model)

    return model, tokenizer


def model_interpret(sentence, model, tokenizer):
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = cls_explainer(sentence)

    return cls_explainer, word_attributions


def transformer_visual(doc):
    model, tokenizer = load_model()
    cls_explanier, word_attributions = model_interpret(doc, model, tokenizer)

    return components.v1.html(cls_explanier.visualize()._repr_html_(), scrolling=True, height=350)


def main():
    st.title("Visualize Transformer Model")
    doc = st.text_area("Enter your sentence to visualize", "")
    if doc:
        transformer_visual(doc)


# calling the main function
if __name__ == "__main__":
    main()
