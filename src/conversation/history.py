"""
CONCEPT: Conversation History Manager

By Injecting the last N messages into every LLM call, we give the agent a "short term memory" 
that allows it to maintain context and continuity across multiple interactions.

HOW IT WORKS:
  We store messages as a list of {"role": "user"/"agent"/"system", "content": "the message text"} dicts in memory.}

TRIMING:
    Production: calculates exact token counts and trims history to fit wirhin the model's context window
    Here we simply cap at MAX_HISTORY messages (pairs of user and agent messages) for simplicity.

STORAGE:
    In-memory dict (lost on restart). For persistence:
    - SQLite: store messages in a local database file.
    - Redis: use an in-memory data store for fast access and persistence.
    Production uses Cosmos DB
"""
# Max number of message pairs to keep in history
# eg. MAX_HISTORY=10 means we keep the last 10 user messages and 10 agent messages (20 total) in the history context.
MAX_HISTORY = 10  

# In-memory storage for conversation history
_store: dict[str, list[dict]] = {}

def add_message(conversation_id: str, role: str, content: str):
    """
    Add a message to the conversation history.
    """
    if conversation_id not in _store:
        _store[conversation_id] = []
    
    _store[conversation_id].append({"role": role, "content": content})
    
    # Trim the oldest history to keep only the last MAX_HISTORY messages
    if len(_store[conversation_id]) > MAX_HISTORY: 
        trimmed = len(_store[conversation_id]) - MAX_HISTORY 
        _store[conversation_id] =_store[conversation_id][-MAX_HISTORY:]
        print(f"[History] Trimmed {trimmed} messages from conversation {conversation_id} history.")

def get_history(conversation_id: str) -> list[dict]:
    """
    Retrieve the conversation history for a given conversation ID.
    This list is passed directly to provider.py as the LLM prior context
    """
    return list(_store.get(conversation_id, []))

def clear_history(conversation_id: str):
    """
    Clear the conversation history for a given conversation ID.
    Useful for resetting context after a long conversation or when starting a new topic.
    """
    removed = _store.pop(conversation_id, None)
    if removed:
        print(f"[History] Cleared {len(removed)} messages for conversation {conversation_id}.")

def conversation_count() -> int:
    """
    Utility function to get the number of active conversations in memory (Debugging utility).
    """
    return len(_store)
