"""Step C - Ask an LLM (Llama on Groq) to answer ONLY from the retrieved chunks."""
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

DEFAULT_MODEL = "llama-3.3-70b-versatile"

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I could not find that in the document."

Context:
{context}

Question: {question}

Answer:"""
)


def list_chat_models(api_key):
    """Ask Groq which models this key can use (so we never hard-code a retired one)."""
    ids = [m.id for m in Groq(api_key=api_key).models.list().data]
    skip = ("whisper", "tts", "guard", "embed")  # not chat models
    return sorted(i for i in ids if not any(s in i.lower() for s in skip))


def answer_question(question, chunks, model=DEFAULT_MODEL):
    """Glue the chunks into one context block and let the LLM write the answer."""
    context = "\n\n".join(c.page_content for c in chunks)
    llm = ChatGroq(model=model, temperature=0)  # 0 = factual, no creativity
    reply = (PROMPT | llm).invoke({"context": context, "question": question})
    return reply.content
