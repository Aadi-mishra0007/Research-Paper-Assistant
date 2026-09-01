"""
Research Assistant "agent" layer.
 
This module is intentionally simple and explicit rather than using a heavy
agent framework — for this use case the agent's job is well-defined:
  1. Summarize a paper (structured output)
  2. Answer follow-up questions about a paper, grounded in RAG context
  3. Compare multiple papers' summaries
 
This keeps the system easy to explain, debug, and grade, while still
satisfying the "Agent" requirement (task-specific behavior + tool use = RAG retrieval).
"""
 
from typing import List, Tuple
 
from langchain_groq import ChatGroq
 
from .rag import PaperIndex, retrieve_context, retrieve_context_for_summary
from .prompts import SUMMARY_PROMPT, QA_SYSTEM_PROMPT, COMPARE_PROMPT
 
# Groq model choice: llama-3.3-70b-versatile was deprecated by Groq (June 2026).
# openai/gpt-oss-120b is Groq's current recommended replacement for
# general-purpose, high-quality responses. Swap to "openai/gpt-oss-20b" for
# faster/cheaper responses if needed.
GROQ_MODEL = "openai/gpt-oss-120b"
 
 
def get_llm(temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, temperature=temperature)
 
 
def summarize_paper(index: PaperIndex) -> str:
    """Generate the structured Summary / Contributions / Limitations / Future Work."""
    context = retrieve_context_for_summary(index)
    llm = get_llm()
    prompt = SUMMARY_PROMPT.format(context=context)
    response = llm.invoke(prompt)
    return response.content
 
 
def answer_question(
    index: PaperIndex,
    question: str,
    chat_history: List[Tuple[str, str]],
) -> str:
    """
    Answer a question about the paper using RAG context.
    chat_history is a list of (role, message) tuples for simple conversational memory.
    """
    context = retrieve_context(index, question, k=6)
 
    history_text = "\n".join(f"{role}: {msg}" for role, msg in chat_history[-6:]) or "(no prior turns)"
 
    prompt = QA_SYSTEM_PROMPT.format(
        context=context,
        chat_history=history_text,
        question=question,
    )
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content
 
 
def compare_papers(summaries: dict) -> str:
    """
    summaries: dict of {paper_name: summary_text}
    Produces a structured comparison across 2+ papers.
    """
    formatted = "\n\n".join(
        f"### {name}\n{summary}" for name, summary in summaries.items()
    )
    prompt = COMPARE_PROMPT.format(paper_summaries=formatted)
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content
 