# First-Principles Academic RAG Assistant

This project is an academic assistant designed to explain complex topics from first principles rather than providing surface-level summaries.

It combines retrieval-augmented generation with structured reasoning to produce clear and grounded explanations based on source documents.

---

## Overview

The system allows users to upload a PDF (such as lecture notes or academic papers) and ask questions about its content. Instead of returning a direct summary, the system:

- identifies the core concepts relevant to the question
- builds a structured reasoning plan
- generates a step-by-step explanation
- grounds the answer in the original source

---

## Architecture

The pipeline follows these steps:

PDF Upload  
→ Text Chunking  
→ Embedding Generation  
→ Vector Search (FAISS)  
→ Relevant Context Retrieval  
→ Concept Extraction  
→ Reasoning Plan Generation  
→ First-Principles Explanation  
→ Source Grounding  

---

## Technologies

- Python  
- LangChain  
- OpenAI API  
- FAISS  
- Streamlit  

---

## Setup

### Clone the repository

```bash
git clone https://github.com/nakresz/serkan.git
cd serkan