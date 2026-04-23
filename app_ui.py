import json
import os
import tempfile

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
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

explanation_level = st.selectbox(
    "Explanation level",
    ["Beginner", "Undergraduate", "Technical"],
    index=1
)


@st.cache_resource
def build_vector_store(pdf_bytes: bytes):
    """Build a FAISS vector store from uploaded PDF bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        temp_pdf_path = tmp_file.name

    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def get_level_instructions(level: str) -> str:
    """Return prompt instructions based on the selected explanation level."""
    if level == "Beginner":
        return """
- Assume the user is new to the topic.
- Use simple language.
- Avoid heavy formalism unless absolutely necessary.
- Focus on intuition, clarity, and concrete meaning.
- Define unfamiliar terms before using them.
- Prefer conceptual understanding over technical detail.
- Keep the explanation accessible and easy to follow.
"""
    if level == "Undergraduate":
        return """
- Assume the user has university-level background.
- Use proper academic terminology.
- Balance intuition with technical accuracy.
- Include formal details only when they help the explanation.
- Keep the explanation structured and pedagogical.
- Emphasize both meaning and mechanism.
"""
    return """
- Assume the user is technically strong.
- Use precise, discipline-appropriate language.
- Do not oversimplify.
- Emphasize formal structure, assumptions, relations, and implications.
- If the topic involves an equation, explain what its terms do.
- If the topic involves a mechanism, explain the causal structure.
- If the topic involves a principle, explain assumptions and consequences.
- If the topic involves a representation or formalism, explain what is represented and how.
- Prefer structural understanding over motivational language.
- Make the explanation compact but rigorous.
"""


def extract_structured_concepts(llm, context: str, query: str) -> dict:
    """Ask the LLM for structured concepts in JSON format."""
    concept_prompt = f"""
You are helping build a first-principles academic assistant.

From the context and question below, extract the most important conceptual structure needed
to explain the answer from first principles.

Return ONLY valid JSON with exactly this structure:

{{
  "main_object": "...",
  "governing_relation": "...",
  "core_process_or_idea": "...",
  "formal_structure": "...",
  "secondary_concepts": ["...", "..."]
}}

Rules:
- Use short concept names
- Do not include explanations
- Do not include markdown
- Do not include any text before or after the JSON
- Prefer concepts explicitly supported by the context
- Use discipline-neutral naming
- If something is unclear, still return the same JSON structure with your best guess

Context:
{context}

Question:
{query}
"""
    concept_response = llm.invoke(concept_prompt)
    raw_text = concept_response.content.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "main_object": "Unknown",
            "governing_relation": "Unknown",
            "core_process_or_idea": "Unknown",
            "formal_structure": "Unknown",
            "secondary_concepts": []
        }


def capitalize_value(value: str) -> str:
    """Capitalize the first character of a value safely."""
    if not value:
        return "Unknown"
    return value[0].upper() + value[1:]


if uploaded_file:
    pdf_bytes = uploaded_file.read()
    vector_store = build_vector_store(pdf_bytes)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 6,
            "lambda_mult": 0.7
        }
    )

    llm = ChatOpenAI(api_key=api_key, temperature=0)

    query = st.text_input("Ask a question:")

    if st.button("Ask") and query:
        with st.spinner("Thinking..."):
            relevant_docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            level_instructions = get_level_instructions(explanation_level)

            concept_data = extract_structured_concepts(llm, context, query)

            concept_block = f"""
Main object: {concept_data.get("main_object", "Unknown")}
Governing relation: {concept_data.get("governing_relation", "Unknown")}
Core process or idea: {concept_data.get("core_process_or_idea", "Unknown")}
Formal structure: {concept_data.get("formal_structure", "Unknown")}
Secondary concepts: {", ".join(concept_data.get("secondary_concepts", []))}
""".strip()

            plan_prompt = f"""
You are an expert teacher preparing a teaching plan.

Question:
{query}

Explanation level:
{explanation_level}

Structured concepts:
{concept_block}

Create a short reasoning plan for how to explain this question from first principles.

Rules:
- Start from the most basic idea
- Build logically toward the answer
- Prefer ideas explicitly present in the context
- Go beyond "what it is" and include "why it is needed"
- If a mechanism, equation, or formal structure is present, include a step about how it works
- Keep it concise
- Make the plan suitable for the selected explanation level
- Output only the plan steps

Format:
1. ...
2. ...
3. ...
4. ...
5. ...
"""
            plan_response = llm.invoke(plan_prompt)
            reasoning_plan = plan_response.content.strip()

            answer_prompt = f"""
You are an expert academic instructor.

Use the structured concepts and reasoning plan below to build a clear first-principles explanation.

Explanation level:
{explanation_level}

Level instructions:
{level_instructions}

Structured concepts:
{concept_block}

Reasoning plan:
{reasoning_plan}

IMPORTANT:
- Base your explanation primarily on the provided context.
- Prefer ideas explicitly present in the context over generic background knowledge.
- If a concept appears in the context and is central to the question, include it.
- Do not drift into unrelated domain knowledge.
- Be precise: describe objects according to their role in the context, not with careless shorthand.

Rules:
1. Follow the reasoning plan step by step.
2. Start from the most basic conceptual foundation.
3. Explain not only what the concept is, but also why this structure, relation, or mechanism is needed.
4. If the topic includes formal structure, explain what that structure does.
5. If the topic includes an equation, explain what its terms or parts represent and how they interact.
6. If the topic includes a mechanism, explain how the mechanism works step by step.
7. If the topic includes a principle, explain its assumptions and implications.
8. When possible, explain what would be missing or impossible without this structure.
9. Use intuitive but academically correct language.
10. Do not summarize the text mechanically.
11. Do not mention the plan explicitly.
12. Make the explanation structured, clear, and teachable.
13. The intuition section must use a concrete mental model, not vague phrases like "guiding principle", "framework", or "roadmap".
14. The intuition must answer: "what is actually happening?" in simple but accurate terms.
15. Use analogies only if they map clearly to the real mechanism.
16. If possible, include a minimal mental simulation of the process: what happens step by step in reality.
17. Connect abstract ideas to something operational: what can be computed, predicted, explained, or observed because of this?
18. End with a short takeaway.
19. In Technical mode, emphasize formal structure, assumptions, relations, and implications.
20. After writing the intuition, check if it contains vague phrases such as "guiding principle", "framework", or "roadmap".
21. If it does, rewrite the intuition using a concrete, physically or operationally grounded explanation.
22. The intuition must describe a process that could be mentally simulated step by step.
23. The intuition MUST describe a concrete physical or computational process.
24. It must explain what is changing, why it changes, and how it changes step by step.
25. Avoid metaphors like "roadmap", "framework", or "guiding principle".
26. If such phrases appear, rewrite the intuition using a process-based explanation.
27. The intuition should allow the reader to mentally simulate the system evolving over time.
28. The intuition MUST include phrases like:
- "at time t"
- "after a small time step dt"
- "this change is caused by"

29. If these are missing, rewrite the intuition.

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

Intuition (must follow this structure):

1. Start from an initial state of the system.
2. Describe what changes in a very small step.
3. Explain what causes that change.
4. Describe how repeating this process determines the full behavior.

Write this as a continuous explanation, not bullet points.

Short takeaway:
...
"""
            response = llm.invoke(answer_prompt)

        st.subheader("Explanation Level")
        st.write(explanation_level)

        st.subheader("Core Concepts")
        secondary_concepts = concept_data.get("secondary_concepts", [])
        secondary_text = ", ".join(secondary_concepts) if secondary_concepts else "None"

        st.markdown(f"""
### Main object
{capitalize_value(concept_data.get("main_object", "Unknown"))}

### Governing relation
{capitalize_value(concept_data.get("governing_relation", "Unknown"))}

### Core process or idea
{capitalize_value(concept_data.get("core_process_or_idea", "Unknown"))}

### Formal structure
{capitalize_value(concept_data.get("formal_structure", "Unknown"))}

### Secondary concepts
{capitalize_value(secondary_text)}
""")

        st.subheader("Reasoning Plan")
        st.text(reasoning_plan)

        st.subheader("Answer")
        st.write(response.content)