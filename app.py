from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found.")

# Load PDF
pdf_path = "data/raw/sample.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings(api_key=api_key)

# Create FAISS index
vector_store = FAISS.from_documents(chunks, embeddings)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Create LLM
llm = ChatOpenAI(api_key=api_key, temperature=0)

# Ask a question
query = "What is the Schrödinger equation?"

# Retrieve relevant chunks
relevant_docs = retriever.invoke(query)

# Combine retrieved context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Step 1: Extract key concepts
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

print("Key Concepts:")
print(concepts)
print("\n" + "=" * 50)

# Step 2: Generate first-principles explanation
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

print("Final Answer:")
print(response.content)