"""
CONCEPT: LLM Provider  (updated: TIER 2 — Token Management + Tool Calling)
============================================================================
Production equivalents:
  - InternalPromptService    (prompt assembly)
  - TokenManagementService   (token counting + budget trimming)
  - LlmGatewayOpenAiStreamingProvider  (streaming call)
  - McpToOpenAiConverter + MCPRegistryLiteService (tool calling loop)

NEW IN THIS VERSION:

  1. TOKEN BUDGET MANAGEMENT
     Before sending to the LLM, we count tokens in every message and trim
     history from oldest-first until the total fits within BUDGET_HISTORY.
     Uses tiktoken (same library as jtokkit conceptually).
     Production: TokenManagementService.calculateGPTTokens() with
     JTokkitTokenCountEstimator(EncodingType.CL100K_BASE / O200K_BASE)

  2. TOOL CALLING LOOP
     - First call: non-streaming + tools → detect if LLM wants to call a tool
     - If tool calls: execute them, add results to messages
     - Final call: streaming → yield tokens to client
     Production: FactCompilationService.getMcpFacts() pre-executes MCP tools;
     here the LLM itself decides when to call tools (OpenAI function calling).

  3. MEMORIES INJECTED INTO SYSTEM PROMPT
     Long-term user memories are now a separate section in the system message,
     distinct from the RAG facts. This mirrors how MemoryService results are
     injected in StreamingChatService.handlePrompt().
"""

import json
from typing import Generator
from openai import OpenAI
import tiktoken

from llm.tools import TOOLS, execute_tool
from utils.logger import get_logger
from config import (
    CHAT_MODEL,
    BUDGET_SYSTEM_PROMPT,
    BUDGET_FACTS,
    BUDGET_MEMORY,
    BUDGET_HISTORY,
)

logger = get_logger("provider")
client = OpenAI()

# tiktoken encoder — cl100k_base covers gpt-4o-mini, gpt-4o, gpt-4
# Production: JTokkitTokenCountEstimator with EncodingType.CL100K_BASE
try:
    _encoder = tiktoken.encoding_for_model(CHAT_MODEL)
except KeyError:
    _encoder = tiktoken.get_encoding("cl100k_base")


# ── Token utilities ───────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    """
    Count the approximate number of tokens in a string.
    Production: TokenManagementService calls JTokkitTokenCountEstimator.estimate()
    Falls back to character/4 heuristic if encoder fails (same as estimateTokens()).
    """
    try:
        return len(_encoder.encode(text))
    except Exception:
        return max(1, len(text) // 4)  # fallback: 4 chars ≈ 1 token


def _trim_history_to_budget(
    history: list[dict],
    budget: int = BUDGET_HISTORY,
) -> list[dict]:
    """
    Remove oldest messages until history fits within the token budget.

    Strategy: drop from the front (oldest first), keep the most recent exchanges.
    Production: TokenManagementService trims based on exact token counts,
    this mirrors that logic using tiktoken.

    Args:
        history: List of {"role": ..., "content": ...} dicts
        budget:  Max tokens allowed for history

    Returns:
        Trimmed history list
    """
    trimmed = list(history)
    while trimmed:
        total = sum(count_tokens(m["content"]) for m in trimmed)
        if total <= budget:
            break
        trimmed.pop(0)  # drop oldest message

    if len(trimmed) < len(history):
        logger.info(
            "History trimmed to fit token budget",
            extra={"original": len(history), "trimmed": len(trimmed), "budget": budget},
        )
    return trimmed


# ── Prompt templates ──────────────────────────────────────────────────────────

_FACTS_SECTION = """
--- RETRIEVED KNOWLEDGE ---
{facts}
--- END KNOWLEDGE ---"""

_MEMORY_SECTION = """
--- WHAT I KNOW ABOUT YOU ---
{memories}
--- END USER CONTEXT ---"""


# ── Main entry point ──────────────────────────────────────────────────────────


def stream_answer(
    agent: dict,
    facts: list[str],
    memories: list[str],
    history: list[dict],
    question: str,
) -> Generator[str, None, None]:
    """
    Build the prompt, handle tool calls, then stream the final LLM response.

    Flow:
      1. Assemble messages (system + history + user)
      2. Trim history to token budget
      3. First LLM call (non-streaming) to detect tool calls
      4. Execute any tool calls → add results to messages
      5. Stream the final answer

    Args:
        agent    : Agent config dict from registry
        facts    : RAG chunks from retriever (Retrieved Knowledge)
        memories : Long-term user memories from memory.py
        history  : Prior conversation turns (trimmed to budget)
        question : Current user question

    Yields:
        String tokens as they arrive from the LLM
    """
    messages = _build_messages(agent, facts, memories, history, question)

    # ── Log token usage per component ─────────────────────────────────────────
    system_tokens = count_tokens(messages[0]["content"])
    history_tokens = sum(count_tokens(m["content"]) for m in messages[1:-1])
    question_tokens = count_tokens(question)
    logger.info(
        "Prompt assembled",
        extra={
            "agent": agent["name"],
            "system_tokens": system_tokens,
            "history_tokens": history_tokens,
            "question_tokens": question_tokens,
            "facts_count": len(facts),
            "memories_count": len(memories),
        },
    )

    # ── Step 1: Non-streaming call with tools to detect tool calls ────────────
    # Production: FactCompilationService pre-executes MCP tools before streaming.
    # Here: the LLM itself decides whether to call a tool (OpenAI function calling).
    #
    # IMPORTANT: OpenAI rejects tool_choice="auto" when tools=[] (empty list).
    # Only include tools/tool_choice params when there are tools registered.
    create_kwargs: dict = dict(
        model=CHAT_MODEL,
        messages=messages,
        stream=False,
        temperature=0.7,
    )

    # -----
    # When you pass tools=TOOLS, OpenAI’s API automatically injects a hidden system-level instruction telling the LLM:
    # “These tools are available to you. If the user’s question needs one, respond with a tool call instead of plain text.”
    if TOOLS:
        create_kwargs["tools"] = TOOLS
        create_kwargs["tool_choice"] = "auto"  # LLM decides whether to call a tool

    first_response = client.chat.completions.create(**create_kwargs)

    first_msg = first_response.choices[0].message

    # ── Step 2: Execute tool calls if the LLM requested any ───────────────────
    if first_msg.tool_calls:
        logger.info(
            "Tool calls requested",
            extra={"tools": [tc.function.name for tc in first_msg.tool_calls]},
        )
        # Add the assistant's tool-call request to message history
        messages.append(first_msg)

        # Execute each tool and add the result back
        for tool_call in first_msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = execute_tool(tool_call.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    elif first_msg.content:
        # LLM answered directly without tool calls — stream it immediately
        # (Reconstruct as a streaming-like generator for API consistency)
        yield first_msg.content
        return

    # ── Step 3: Final streaming call (with tool results now in messages) ──────
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=2048,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ── Private helpers ───────────────────────────────────────────────────────────


def _build_messages(
    agent: dict,
    facts: list[str],
    memories: list[str],
    history: list[dict],
    question: str,
) -> list[dict]:
    """
    Assemble the final messages array for the LLM call.

    Order:
      [0] system  → agent identity + facts + memories
      [1..N-1]    → trimmed conversation history
      [N]         → current user question

    Production: InternalPromptService.buildPrompt() does the same assembly,
    then TokenManagementService.calculateTokens() audits the final sizes.
    """
    # ── System message ────────────────────────────────────────────────────────
    system_parts = [agent["system_prompt"]]

    if facts:
        formatted = "\n\n".join(f"[{i+1}] {f}" for i, f in enumerate(facts))
        system_parts.append(_FACTS_SECTION.format(facts=formatted))

    if memories:
        formatted = "\n".join(f"- {m}" for m in memories)
        system_parts.append(_MEMORY_SECTION.format(memories=formatted))

    system_parts.append(
        "\nAlways ground your answer in the RETRIEVED KNOWLEDGE above when relevant."
    )

    messages = [{"role": "system", "content": "\n".join(system_parts)}]

    # ── Trimmed history ───────────────────────────────────────────────────────
    trimmed_history = _trim_history_to_budget(history)
    messages.extend(trimmed_history)

    # ── Current question ──────────────────────────────────────────────────────
    messages.append({"role": "user", "content": question})

    return messages
