from typing import TypedDict


class AgentState(TypedDict):
    input: str            # The user's question
    output: str           # The agent's generated answer
    system_prompt: str    # The prompt being optimized by APO
