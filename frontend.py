import streamlit as st

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

system_prompt=st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider=st.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

# Tool selection section
st.subheader("Tool Options")
col1, col2 = st.columns(2)

with col1:
    allow_web_search = st.checkbox("Allow Web Search", help="Enable web search for real-time information")

with col2:
    allow_reasoning = st.checkbox("Enable Reasoning Tools", help="Enable logical reasoning and analysis tools")

user_query=st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

API_URL="http://127.0.0.1:9999/chat"

if st.button("Ask Agent!", type="primary"):
    if user_query.strip():
        #Step2: Connect with backend via URL
        import requests

        # Show tools being used
        tools_info = []
        if allow_web_search:
            tools_info.append("Web Search")
        if allow_reasoning:
            tools_info.append("Reasoning Tools")
        
        if tools_info:
            st.info(f"Tools enabled: {', '.join(tools_info)}")

        payload={
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search,
            "allow_reasoning": allow_reasoning
        }

        with st.spinner("Agent is thinking..."):
            response=requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:**\n\n{response_data.get('response', '')}")
        else:
            st.error(f"API Error: {response.status_code}")
    else:
        st.warning("Please enter a query to get a response from the agent.")