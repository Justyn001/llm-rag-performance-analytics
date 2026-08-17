"""
Prompt templates for the RAG system.

These templates tell the LLM HOW to use the retrieved context.
Without a good prompt, even the best LLM will give bad answers.

We have separate templates for EN and PL to test multilingual performance.
"""

# =============================================================================
# English prompts
# =============================================================================

RAG_SYSTEM_PROMPT_EN = """You are a helpful question-answering assistant. \
Your task is to answer questions based ONLY on the provided context. \
If the context does not contain enough information to answer the question, \
say "I don't have enough information to answer this question."

Rules:
- Answer based ONLY on the provided context
- Be concise and direct
- If the answer is not in the context, say so clearly
- Do not make up information"""

RAG_USER_PROMPT_EN = """Context:
{context}

Question: {question}

Answer:"""

# =============================================================================
# Polish prompts
# =============================================================================

RAG_SYSTEM_PROMPT_PL = """Jesteś pomocnym asystentem odpowiadającym na pytania. \
Twoim zadaniem jest odpowiadanie na pytania WYŁĄCZNIE na podstawie dostarczonego kontekstu. \
Jeśli kontekst nie zawiera wystarczających informacji, \
powiedz "Nie mam wystarczających informacji, aby odpowiedzieć na to pytanie."

Zasady:
- Odpowiadaj WYŁĄCZNIE na podstawie dostarczonego kontekstu
- Bądź zwięzły i konkretny
- Jeśli odpowiedzi nie ma w kontekście, powiedz to jasno
- Nie wymyślaj informacji"""

RAG_USER_PROMPT_PL = """Kontekst:
{context}

Pytanie: {question}

Odpowiedź:"""


def get_prompts(language: str = "en") -> tuple[str, str]:
    """
    Get system and user prompts for the specified language.

    Args:
        language: "en" or "pl"

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    if language == "pl":
        return RAG_SYSTEM_PROMPT_PL, RAG_USER_PROMPT_PL
    return RAG_SYSTEM_PROMPT_EN, RAG_USER_PROMPT_EN
