"""Step C - Ask an LLM (Llama on Groq) to answer ONLY from the retrieved chunks."""
from groq import Groq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

DEFAULT_MODEL = "llama-3.3-70b-versatile"
NOT_FOUND = "I could not find that in the document."

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions about the user's documents.
Use ONLY the numbered context passages below. After each fact, cite its passage like [1] or [2].
If the answer is not in the context, reply exactly: "{not_found}"

Recent conversation (for understanding follow-ups only, not a source of facts):
{history}

Context:
{context}

Question: {question}

Answer:"""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the follow-up question as one standalone question that makes sense without the
conversation. Keep names and numbers. Reply with the question only.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""
)


def list_chat_models(api_key):
    """Ask Groq which models this key can use (so we never hard-code a retired one)."""
    ids = [m.id for m in Groq(api_key=api_key).models.list().data]
    skip = ("whisper", "tts", "guard", "embed")  # not chat models
    return sorted(i for i in ids if not any(s in i.lower() for s in skip))


def _llm(model):
    return ChatGroq(model=model, temperature=0)  # 0 = factual, no creativity


def format_history(history, turns=3):
    """Last few (question, answer) pairs as plain text."""
    lines = []
    for q, a, *_ in history[-turns:]:
        lines += [f"User: {q}", f"Assistant: {a}"]
    return "\n".join(lines) or "(none)"


def make_standalone(question, history, model=DEFAULT_MODEL):
    """Chat memory: turn 'what about part-time staff?' into a full question before searching."""
    if not history:
        return question
    chain = REWRITE_PROMPT | _llm(model) | StrOutputParser()
    return chain.invoke({"history": format_history(history), "question": question}).strip()


def stream_answer(question, chunks, history=(), model=DEFAULT_MODEL):
    """Yield the answer piece by piece, so it appears on screen as it's written."""
    context = "\n\n".join(f"[{i}] {c.page_content}" for i, c in enumerate(chunks, 1))
    chain = ANSWER_PROMPT | _llm(model) | StrOutputParser()
    yield from chain.stream({
        "context": context,
        "question": question,
        "history": format_history(list(history)),
        "not_found": NOT_FOUND,
    })
