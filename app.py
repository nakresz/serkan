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

print("Key Concepts:")
print(concepts)
print("\n" + "=" * 50)

# Step 2: Build reasoning plan
plan_prompt = f"""
You are a physics professor preparing a lesson.

Using the concepts below, create a logical teaching plan.

Concepts:
{concepts}

Rules:
- Start from basics
- Build step by step
- Only outline steps (no explanations)
- Keep it short

Format:
1. ...
2. ...
3. ...
"""

plan_response = llm.invoke(plan_prompt)
reasoning_plan = plan_response.content.strip()

print("Reasoning Plan:")
print(reasoning_plan)
print("\n" + "=" * 50)

# Step 3: Generate final answer using plan
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

print("Final Answer:")
print(response.content)