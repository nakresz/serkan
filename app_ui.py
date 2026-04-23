import os

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found in .env file.")
    st.stop()

# Streamlit page setup
st.set_page_config(page_title="First-Principles Academic RAG Assistant")
st.title("First-Principles Academic RAG Assistant")
st.write("Ask questions about your academic PDF and get first-principles explanations.")

# PDF path
pdf_path = "data/raw/sample.pdf"

# Load and process PDF
@st.cache_resource
def build_vector_store():
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

vector_store = build_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(api_key=api_key, temperature=0)

# User input
query = st.text_input("Ask a question:")

if st.button("Ask") and query:
    relevant_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    concept_prompt = f"""
You are helping build a first-principles academic assistant.

From the context and question below, extract the 3 to 6 most important CORE PHYSICS concepts
needed to explain the answer from first principles.

Focus on:
- foundational physical ideas
- main mathematical objects
- core dynamical concepts

Do NOT focus on:
- minor details
- secondary consequences
- descriptive phrases unless they are central

Rules:
1. Return only the concept names.
2. Use short concept phrases.
3. No explanations.
4. One concept per line.
5. Prefer fundamental concepts over derived quantities.

Context:
{context}

Question:
{query}

Concepts:
"""

    concept_response = llm.invoke(concept_prompt)
    concepts = concept_response.content.strip()

    answer_prompt = f"""
You are a physics professor teaching a university student.

Your task is to answer the question from FIRST PRINCIPLES.

Use the following extracted concepts as the backbone of your explanation:
{concepts}

Rules:
1. Start from the most basic idea.
2. Explain why the concept is needed in physics.
3. Build the explanation step by step.
4. Use simple but scientifically correct language.
5. Do not just summarize the context.
6. Do not say "the text says" or "according to the context".
7. End with a short intuitive takeaway.

Context:
{context}

Question:
{query}

Answer in this structure:

Core idea:
...

Key concepts used:
- ...
- ...

Step-by-step explanation:
1. ...
2. ...
3. ...

Intuition:
...

Short takeaway:
...
"""

    response = llm.invoke(answer_prompt)

    st.subheader("Key Concepts")
    st.text(concepts)

    st.subheader("Answer")
    st.write(response.content)