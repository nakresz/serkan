# First-Principles Academic RAG Assistant

This project is an academic assistant designed to explain complex topics from first principles rather than providing surface-level summaries.

It combines retrieval-augmented generation with structured reasoning to produce clear, step-by-step explanations grounded in source documents.

---

## Overview

The system allows users to upload a PDF (lecture notes, research papers, technical documents) and ask questions about its content.

Instead of returning a simple answer, the system:

- identifies the core concepts behind the question  
- extracts a structured conceptual representation  
- builds a reasoning plan  
- generates a step-by-step explanation  
- produces process-based intuition (how the system evolves or works)  

The goal is not just to answer, but to teach.

---

## Key Features

- Structured concept extraction (main object, governing relation, process, structure)
- First-principles reasoning pipeline
- Process-based intuition generation (no generic explanations)
- Multi-domain support (physics, machine learning, etc.)
- Adjustable explanation levels (Beginner, Undergraduate, Technical)
- PDF-based contextual grounding using vector search

---

## Architecture

The pipeline:

PDF Upload  
→ Text Chunking  
→ Embedding Generation  
→ Vector Search (FAISS)  
→ Context Retrieval  
→ Structured Concept Extraction  
→ Reasoning Plan Generation  
→ First-Principles Explanation  
→ Process-Based Intuition  

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
```

### Install dependencies

```
pip install -r requirements.txt
```

### Set your API key

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

### Run the application

```
streamlit run app_ui.py
```


