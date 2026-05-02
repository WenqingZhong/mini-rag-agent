"""
TIER 2: Tool Use / Function Calling
====================================
Production equivalent:
  MCPRegistryLiteService
  + FactCompilationService.getMcpFacts()
  + McpToOpenAiConverter  (converts MCP schema → OpenAI format)

HOW TOOL CALLING WORKS (OpenAI API):
  1. We send the LLM a list of tool definitions (name + description + JSON schema)
  2. The LLM reads them and decides whether to call one
  3. If yes: it returns a tool_calls response INSTEAD of a text reply
  4. We execute the tool locally and get the result
  5. We append the result to the conversation as a "tool" message
  6. We call the LLM again — now it generates a final text response with the result as context

PRODUCTION PARALLEL:
  getMcpFacts() calls MCP tools BEFORE the LLM call (pre-execution pattern).
  OpenAI function calling uses a mid-conversation loop (the LLM triggers tools).
  Both achieve the same goal: giving the LLM fresh, dynamic data.

TOOLS DEFINED:
  - get_current_datetime : returns the current UTC timestamp
  - calculate            : evaluates safe arithmetic expressions
  - count_words          : counts words in a string
"""

from datetime import datetime, timezone
from utils.logger import get_logger

logger = get_logger("tools")


# ── Tool definitions in OpenAI function-calling format ──────────────────────
# Production: McpToOpenAiConverter builds this list from MCP tool JSON schemas.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": (
                "Returns the current UTC date and time. "
                "Use this when the user asks what date or time it is."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluates a safe arithmetic expression and returns the result. "
                "Use for any math calculation the user asks about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '(100 + 50) * 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_words",
            "description": "Counts the number of words in a given piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text whose words should be counted.",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """
    Execute a tool call and return the result as a plain string.

    The result is appended to the conversation as:
        {"role": "tool", "content": result}
    before the final LLM call.

    Production equivalent: MCPRegistryLiteService.invokeToolFromCoordinatesAsync()

    Args:
        name: Tool name (must match a name in TOOLS)
        args: Parsed argument dict from the LLM's tool_call.function.arguments JSON
    Returns:
        String result to feed back to the LLM
    """
    logger.info("Tool called", extra={"tool": name, "tool_args": args})

    if name == "get_current_datetime":
        result = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    elif name == "calculate":
        expression = args.get("expression", "")
        try:
            # Restrict eval to arithmetic only — no builtins, no imports
            result = str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            result = f"Error evaluating '{expression}': {e}"

    elif name == "count_words":
        text = args.get("text", "")
        result = str(len(text.split()))

    else:
        result = f"Unknown tool: '{name}'"

    logger.info("Tool result", extra={"tool": name, "result": result[:120]})
    return result
