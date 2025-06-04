import os
from dotenv import load_dotenv

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


def get_response_from_ai_agent(llm_id, messages, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)
    else:
        raise ValueError("Unsupported provider")

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

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
