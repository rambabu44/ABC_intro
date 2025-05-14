import streamlit as st
import datetime
import pandas as pd
from PIL import Image
import os
import time
import re

# Import the chatbot class
from TNT_NZ.chatbot import NZTravelChatbot

# Page configuration
st.set_page_config(
    page_title="NZ Travel Assistant",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E6B52;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #E8F4F0;
    }
    .chat-message.bot {
        background-color: #F0F0F0;
    }
    .chat-message .avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .chat-message .timestamp {
        font-size: 0.8rem;
        color: #999;
        margin-top: 0.5rem;
    }
    .sidebar-content {
        background-color: #F6F6F6;
        padding: 1rem;
        border-radius: 10px;
    }
    .info-box {
        background-color: #E8F4F0;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'chatbot' not in st.session_state:
    # Initialize the chatbot
    with st.spinner('Initializing NZ Travel Assistant...'):
        st.session_state.chatbot = NZTravelChatbot()
        st.session_state.messages = []
        st.session_state.debug_mode = False

# Sidebar
with st.sidebar:
    st.markdown("<h2>🏔️ NZ Travel Assistant</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    st.markdown("### Options")
    
    # Debug mode toggle
    debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    st.session_state.debug_mode = debug_mode
    
    if st.button("Clear Chat History"):
        confirmation = st.session_state.chatbot.clear_history()
        st.session_state.messages = []
        st.success(confirmation)
    
    # Search knowledgebase directly
    st.markdown("### Search Knowledge Base")
    search_query = st.text_input("Search query")
    num_results = st.slider("Number of results", 1, 10, 3)
    
    if st.button("Search") and search_query:
        with st.spinner('Searching...'):
            results = st.session_state.chatbot.search_knowledge_base(search_query, k=num_results)
            
            if results:
                st.markdown("#### Search Results")
                for i, doc in enumerate(results):
                    with st.expander(f"Result {i+1} - {doc.metadata.get('category', 'Unknown')}"):
                        st.markdown(doc.page_content)
                        st.markdown("---")
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
            else:
                st.warning("No results found.")
    
    st.markdown("### About")
    st.markdown("""
    This chatbot uses RAG (Retrieval Augmented Generation) to provide helpful information about traveling in New Zealand.
    
    You can ask about:
    - Flight bookings and changes
    - Travel destinations in NZ
    - Trip planning advice
    - And more!
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Main content area
st.markdown("<h1 class='main-header'>New Zealand Travel Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Your AI guide to exploring Aotearoa New Zealand</p>", unsafe_allow_html=True)

# Display messages
for message in st.session_state.messages:
    avatar_img = "👤" if message["role"] == "user" else "🤖"
    message_class = "user" if message["role"] == "user" else "bot"
    
    col1, col2 = st.columns([1, 9])
    
    with col1:
        st.markdown(f"<div style='font-size:30px; text-align:center;'>{avatar_img}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div class='chat-message {message_class}'>", unsafe_allow_html=True)
        st.markdown(f"<div class='message'>{message['content']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='timestamp'>{message['timestamp']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show debug info if enabled
    if st.session_state.debug_mode and message["role"] == "user" and "intent" in message:
        with st.expander("Debug Info"):
            st.markdown(f"**Detected Intent:** {message['intent']}")
            if "rag_results" in message:
                st.markdown("**RAG Results:**")
                for i, doc in enumerate(message["rag_results"]):
                    st.markdown(f"Result {i+1}: {doc.page_content[:150]}...")

# User input
with st.container():
    user_input = st.chat_input("Ask about New Zealand travel...", key="user_input")
    
    if user_input:
        # Add user message to chat
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        user_message = {"role": "user", "content": user_input, "timestamp": timestamp}
        st.session_state.messages.append(user_message)
        
        # Process with chatbot
        with st.spinner('Thinking...'):
            # Simulate some processing time to make the interaction feel more natural
            time.sleep(0.5)
            
            # Get response from chatbot
            response = st.session_state.chatbot.process_message(user_input)
            
            # If debug mode is on, get intent information
            if st.session_state.debug_mode:
                # Get the intent that was classified (this would need to be added to your chatbot)
                # This is a placeholder - you'll need to modify the chatbot to return this info
                intent = st.session_state.chatbot.intent_classifier.classify(user_input)
                user_message["intent"] = intent
                
                # Get the RAG results (again, this would need to be added to your chatbot)
                # This is a placeholder - you'll need to modify the chatbot to return this info
                rag_results = st.session_state.chatbot.search_knowledge_base(user_input, k=3)
                user_message["rag_results"] = rag_results
            
            # Add assistant message to chat
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": timestamp})
            
            # Force a rerun to show the new messages
            st.rerun()

# Footer
st.markdown("<div class='footer'>© 2025 NZ Travel Assistant | Powered by LangChain, Groq and Streamlit</div>", unsafe_allow_html=True)