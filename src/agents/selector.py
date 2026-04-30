"""
CONCEPT: Agent Selector (Router)

In production, AgentSelectorService runs 6 match strategies in priority order:
1. CRITICAL -> @mention syntax (e.g. "Hey @python-agent, how do I use pandas?")
2. HIGH     -> name matching (e.g. "I have a question about the python-agent")
3. MEDIUM   -> propoerty/regex/keyword matching (e.g. "I have a question about pandas")
4. LOW      -> LLM matching (ask the LLM to choose the best agent based on the question)
5. FALLBACK -> default agent (e.g. "general-agent")


Here in this repo, we only implement a simplified version.
STEP 1: Keyword Matching (fast, zero LLM cost)
STEP 2: LLM Matching (smart, costs one LLM call)

kEY INSIGHT: The LLM is NOT the first choice, it is the fallback. 
Keyword rules are the first becuase they are instant and free.
"""

import os
from openai import OpenAI
from agents.registry import AGENT_REGISTRY, get_agent, list_agents

client = OpenAI()

FALLBACK_AGENT = "general-agent"
ROUNTING_MODE = "gpt-4o-mini" # cheap small model - rounting doesn't need GPT-4

#----------------------------------
# Rounting prompt - The LLM is given afent descriptions and ask to return ONE agent name.
# Temprature is set to 0 to make the output deterministic.
#----------------------------------
ROUTING_PROMPT = """\
You are an agent rounter. Your only job is to pick the best agent for a given question.

Avilable agents:
{agent_list}

Rules:
 - Read the name and description of each agent carefully.
 - Return ONLY the exact agent name - no explanation, no punctuatioin
 - If no agent fits the question, return {fallback_agent}

 User question: {question}
"""

def select_agent(question: str) -> dict:
    """
    Main entry point. Returns the full agent config dict for the best match.
    """
    print(f"[Selector] Rounting question: {question[:80]}" if len(question) > 80 else f"[Selector] Rounting question: {question}")
    # STEP 1: Keyword Matching
    matched_name = _keyword_match(question)
    if matched_name:
        print(f"[Selector] Keyword matched agent: {matched_name}")
        return get_agent(matched_name)
    
    # STEP 2: LLM Matching
    matched_name = _llm_match(question)
    print(f"[Selector] LLM matched agent: {matched_name}")
    return get_agent(matched_name)


def _keyword_match(question: str) -> str | None:
    """
    Check if any agent's keywords appear in the question. Return the first matched agent name.
    """
    question_lower = question.lower()
    for agent in list_agents():
        for keyword in agent.get("keywords", []):
            if keyword in question_lower:
                return agent["name"]
    return None


def _llm_match(question: str) -> str:
    """
    Use the LLM to select the best agent based on the question and agent descriptions.
    """
    agent_list_str = "\n".join([f"- {agent['name']}: {agent['description']}" for agent in list_agents()])
    prompt = ROUTING_PROMPT.format(agent_list=agent_list_str, fallback_agent=FALLBACK_AGENT, question=question)
    
    try: 
        response = client.chat.completions.create(
            model=ROUNTING_MODE,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        chosen = response.choices[0].message.content.strip()

        if chosen in AGENT_REGISTRY:
            return chosen
        
        print(f"[Selector] LLM returned invalid agent name: {chosen}. Falling back to {FALLBACK_AGENT}")
        return FALLBACK_AGENT
    
    except Exception as e:
        print(f"[Selector] LLM matching failed with error: {e}. Falling back to {FALLBACK_AGENT}")
        return FALLBACK_AGENT