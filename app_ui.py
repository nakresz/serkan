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


@st.cache_resource
def build_vector_store() -> FAISS:
    """Load the PDF, split it into chunks, and build a FAISS vector store."""
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


# Build app resources
vector_store = build_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(api_key=api_key, temperature=0)

# User input
query = st.text_input("Ask a question:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        # Step 1: Retrieve relevant chunks
        relevant_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Step 2: Extract core concepts
        concept_prompt = f"""
You are helping build a first-principles academic assistant.

From the context and question below, extract the 3 to 6 most important concepts that are
DIRECTLY necessary to explain the question from first principles.

Prioritize:
- the main physical object
- the main dynamical idea
- the main governing equation
- the most central mathematical structure

Avoid:
- secondary quantities
- consequences
- measurement-related quantities unless the question directly asks for them
- introductory wording from the text

Rules:
- Return only concept names
- No numbering
- No bullet points
- No explanations
- One concept per line

Context:
{context}

Question:
{query}

Concepts:
"""
        concept_response = llm.invoke(concept_prompt)
        concepts = concept_response.content.strip()

        # Step 3: Build reasoning plan
        plan_prompt = f"""
You are a physics professor preparing a teaching plan.

Question:
{query}

Core concepts:
{concepts}

Create a short reasoning plan for how to explain this question from first principles.

Rules:
- Start from the most basic idea
- Build logically toward the answer
- Do not include derivations unless the question explicitly asks for one
- Do not include examples unless they help the explanation
- Keep it concise
- Output only the plan steps

Format:
1. ...
2. ...
3. ...
4. ...
"""
        plan_response = llm.invoke(plan_prompt)
        reasoning_plan = plan_response.content.strip()

        # Step 4: Generate final answer
        answer_prompt = f"""
You are a physics professor teaching a university student.

Use the concepts and reasoning plan below to build a clear first-principles explanation.

Concepts:
{concepts}

Reasoning plan:
{reasoning_plan}

Rules:
1. Follow the reasoning plan step by step.
2. Start from the most basic physical idea.
3. Explain why the concept is needed.
4. Use intuitive but scientifically correct language.
5. Do not summarize the text mechanically.
6. Do not mention the plan explicitly.
7. Make the explanation pedagogical and clear.
8. End with a short intuition and a short takeaway.
9. Be precise: a wavefunction is a complex-valued function, not just a complex number.
10. Prefer explanation over formula dumping unless the formula is essential.

Context:
{context}

Question:
{query}

Answer in exactly this structure:

Core idea:
...

Step-by-step explanation:
1. ...
2. ...
3. ...
4. ...

Intuition:
...

Short takeaway:
...
"""
        response = llm.invoke(answer_prompt)

    # UI output
    st.subheader("Key Concepts")
    st.text(concepts)

    st.subheader("Reasoning Plan")
    st.text(reasoning_plan)

    st.subheader("Answer")
    st.write(response.content)