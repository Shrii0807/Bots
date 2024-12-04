import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize LLM and Chat Template
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model='llama3')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit App with Chat
def handle_conversation():
    st.title("🤖 Interactive AI Chatbot")
    st.write("Welcome to the AI chatbot! Type 'exit' to quit.")
    
    # Initialize session states to store chat history
    if "context" not in st.session_state:
        st.session_state.context = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Get user input
    user_input = st.text_input("You:", "", key="user_input")
    if user_input:
        # Append new user input to context and get response
        result = chain.invoke({"context": st.session_state.context, "question": user_input})
        
        # Store conversation history
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("AI", result))
        st.session_state.context += f"\nUser: {user_input}\nAI: {result}"
        
        # Clear the input field by setting an empty string
        st.session_state.user_input = ""  # Triggers rerun with input cleared

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            st.chat_message("User").markdown(f"**🙂 You:** {message}")
        else:
            st.chat_message("AI").markdown(f"**🤖 AI:** {message}")

# CSS Styling for better UI
st.markdown("""
    <style>
    /* Chatbox styling */
    .stChatMessage { font-size: 16px; line-height: 1.6; padding: 8px 12px; margin: 8px 0; border-radius: 8px; }
    .stChatMessage.User { background-color: #d1e7ff; }
    .stChatMessage.AI { background-color: #f8f9fa; }
    
    /* Input box styling */
    .stTextInput input { padding: 10px; font-size: 16px; border-radius: 8px; border: 1px solid #ccc; width: 100%; }
    
    /* General styling */
    h1 { color: #0073e6; text-align: center; }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    handle_conversation()
