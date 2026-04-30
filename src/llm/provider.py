"""
CONCEPT: LLM Provider (Interface to LLMs) + Prompt Assmembly

1. Build the final prompt by combining:
- System prompt (agent's personality and behavior guidelines)
- Context (retrieved facts from ChromeDB) -> RAG
- History (last N messages from the conversation) -> Short-term memory
- User (current question)

2. Call the LLM with streaming enabled and yield tokens one by one.

KEY INSIGHT - Why facts go in the system prompt:
- By putting facts in system promot instead of user prompt, we keep the conversation history cleaner 
and prevent the LLM from forgetting the context when history gets too long.

"""
from openai import OpenAI
from typing import Generator

client = OpenAI()

CHAT_MODEL = "gpt-4o-mini" #swao to  "gpt-4o" for higher quality but more expensive responses

#----------------------------------
# Prompt assembly - RAG prompt for context injection.
#----------------------------------

CONTEXT_SECTION = """\
--- RELEVANT CONTEXT ---
{context}
--- END OF CONTEXT ---

Use the context above to answer the question. If the context does not contain enough information, 
answer to the best of your ability based on your general knowledge, but clearly state that you are doing so.

"""

def stream_answer(
        agent_config: dict, 
        facts: list[str], 
        history: list[dict], 
        question: str
) -> Generator[str, None, None]:
    """
    Main entry point. Builds the final prompt and streams the LLM response token by token.
    Args:
        agent_config (dict): The configuration for the selected agent, including system prompt and collection.
        facts (list[str]): The retrieved facts from ChromeDB to be injected into the system prompt.
        history (list[dict]): The conversation history to be included in the user prompt.
        question (str): The current user question to be answered by the LLM.
    Yields:
        str: The next token in the LLM's response (for SSE streaming).
    """
    messages = _build_messages(agent_config, facts, history, question)

    print(f"[Provider] Sending to LLM:")
    print(f" model: {CHAT_MODEL}")
    print(f" messages: {len(messages)} total"
          f" system + {len(history)} + 1 user question")
    print(f" facts: {len(facts)} chunks injected")

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=1024, 
    )

    # Yield each token as it arrives
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def _build_messages(
        agent_config: dict, 
        facts: list[str], 
        history: list[dict], 
        question: str) -> list[dict]:
    """
    Assemble the messages array in the correct order
    1. System: agent identity + retrived facts
    2. History: last N messages from the conversation (if any)
    3. User: the current question

    The LLM reads top to bottom, so order matters.
    """
    messages = []
    system_prompt = agent_config["system_prompt"]
    if facts:
        format_facts = "\n\n".join([f"Fact {i+1}: {fact}" for i, fact in enumerate(facts)])
        system_prompt += CONTEXT_SECTION.format(context=format_facts)
    else:
        system_prompt += CONTEXT_SECTION.format(context="No relevant facts found in the knowledge base.")
    
    messages.append({"role": "system", "content": system_prompt})
    messages.extend(history) # History is already in the correct format of {"role": ..., "content": ...}
    messages.append({"role": "user", "content": question})

    return messages