# ai_agent.py
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from typing import List

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langgraph.prebuilt import create_react_agent


@tool
def reasoning_tool(problem: str, steps: List[str]) -> str:
    """
    A reasoning tool that helps break down complex problems into logical steps.
    
    Args:
        problem: The problem or question to reason through
        steps: A list of reasoning steps to work through the problem
    
    Returns:
        A structured reasoning analysis
    """
    reasoning_output = f"🧠 **Reasoning Analysis**\n\n"
    reasoning_output += f"**Problem:** {problem}\n\n"
    reasoning_output += f"**Step-by-step reasoning:**\n"
    
    for i, step in enumerate(steps, 1):
        reasoning_output += f"{i}. {step}\n"
    
    reasoning_output += f"\n**Conclusion:** Based on the above reasoning steps, "
    reasoning_output += f"this analysis provides a logical framework for understanding '{problem}'."
    
    return reasoning_output


@tool
def logical_analysis_tool(premise: str, conclusion: str) -> str:
    """
    Analyzes the logical connection between a premise and conclusion.
    
    Args:
        premise: The starting assumption or fact
        conclusion: The proposed conclusion
    
    Returns:
        Analysis of the logical validity
    """
    analysis = f"🔍 **Logical Analysis**\n\n"
    analysis += f"**Premise:** {premise}\n"
    analysis += f"**Conclusion:** {conclusion}\n\n"
    analysis += f"**Analysis:** Examining the logical connection between the premise and conclusion...\n"
    analysis += f"This tool helps evaluate whether the conclusion logically follows from the given premise."
    
    return analysis


def get_response_from_ai_agent(llm_id, messages, allow_search, allow_reasoning, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)
    else:
        raise ValueError("Unsupported provider")

    tools = []
    
    if allow_search:
        tools.append(TavilySearchResults(max_results=2))
    
    if allow_reasoning:
        tools.extend([reasoning_tool, logical_analysis_tool])

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    state = {
        "messages": [
            SystemMessage(content=system_prompt),
            *[HumanMessage(content=msg) for msg in messages]
        ]
    }

    response = agent.invoke(state)
    messages = response.get("messages", [])
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1] if ai_messages else "No AI response received."