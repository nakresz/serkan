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

print("Retrieved chunks:")
print("-" * 50)

for i, doc in enumerate(relevant_docs):
    print(f"\nChunk {i + 1}:")
    print(doc.page_content[:500])

print("\n" + "=" * 50)

# Combine context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Final answer prompt
prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

response = llm.invoke(prompt)

print("\nFinal Answer:")
print(response.content)