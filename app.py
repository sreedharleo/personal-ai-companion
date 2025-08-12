import os
import fitz  # PyMuPDF
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# Load API key
load_dotenv()
genai.configure(api_key="AIzaSyBOf5-JO7JJaOE8obRbTXA5b7H2scQI-Zo")

# PDF text extractor
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Create embeddings & vector store
def create_vector_store(text):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    return FAISS.from_texts(chunks, embeddings)

# Query Gemini
def query_gemini(context, question):
    prompt = f"""
    You are a helpful assistant. 
    Context: {context}
    Question: {question}
    Answer clearly and concisely.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI setup
st.set_page_config(page_title="Gemini PDF Q&A", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f4f6f8;
    }
    .main-container {
        background: white;
        padding: 30px;
        border-radius: 15px;
        max-width: 800px;
        margin: auto;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    }
    h1 {
        text-align: center;
        color: #2c3e50;
    }
    .uploaded {
        color: #27ae60;
        font-weight: bold;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border: none;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    .answer-box {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 1.1em;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("📄 Gemini PDF Q&A Assistant")
st.write("Upload a PDF and ask questions to get instant answers powered by **Gemini AI**.")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    st.markdown('<p class="uploaded">✅ PDF uploaded successfully!</p>', unsafe_allow_html=True)
    text = extract_text_from_pdf(uploaded_file)
    vector_store = create_vector_store(text)
    
    st.subheader("Ask a question about your PDF:")
    question = st.text_input("Type your question here...")
    
    if st.button("Get Answer") and question:
        with st.spinner("🤔 Thinking..."):
            docs = vector_store.similarity_search(question, k=3)
            context = " ".join([d.page_content for d in docs])
            answer = query_gemini(context, question)
        st.markdown(f'<div class="answer-box"><b>Answer:</b><br>{answer}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
