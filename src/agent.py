from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .schema import AgentState
from dotenv import load_dotenv
import os

load_dotenv()


# Define the node that calls the LLM using LangChain
async def call_model(state: AgentState):
    # Initialize the LangChain ChatOpenAI model
    llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            temperature=0)
    
    # Construct LangChain messages
    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=state["input"])
    ]
    
    # Invoke model
    response = await llm.ainvoke(messages)
    
    return {"output": response.content}


# Build the LangGraph
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", call_model)
    workflow.set_entry_point("assistant")
    workflow.add_edge("assistant", END)
    return workflow.compile()

# This is the compiled graph instance
langgraph_agent = build_graph()