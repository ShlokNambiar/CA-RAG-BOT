"""
Streamlit entry point for UDCPR RAG Chatbot

This file serves as the entry point for Streamlit Cloud deployment.
It includes error handling for the main application.
"""

import streamlit as st
import os
import sys
import traceback

# IMPORTANT: Set page config first before any other Streamlit commands
st.set_page_config(
    page_title="UDCPR RAG Chatbot",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a minimalist design
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .chat-message.assistant {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
    .chat-message .message-content {
        display: flex;
        margin-top: 0;
    }
    .avatar {
        min-width: 20px;
        margin-right: 10px;
        font-size: 20px;
    }
    .message {
        flex-grow: 1;
    }
    h1, h2, h3 {
        color: #333;
    }
    .stMarkdown a {
        color: #1890ff;
        text-decoration: none;
    }
    .stMarkdown a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📚 UDCPR Document Assistant")

# Now load environment variables
with st.expander("Environment Setup", expanded=False):
    # Try to load from Streamlit secrets first, then fall back to environment variables
    env_vars = {}

    # Load from Streamlit secrets
    try:
        # Check if secrets exist
        if hasattr(st, "secrets") and "general" in st.secrets:
            st.write("Loading secrets from Streamlit...")

            # Load all secrets from the general section
            env_vars["OPENAI_API_KEY"] = st.secrets["general"].get("OPENAI_API_KEY")
            env_vars["PINECONE_API_KEY"] = st.secrets["general"].get("PINECONE_API_KEY")
            env_vars["PINECONE_ENVIRONMENT"] = st.secrets["general"].get("PINECONE_ENVIRONMENT")
            env_vars["SUPABASE_URL"] = st.secrets["general"].get("SUPABASE_URL")
            env_vars["SUPABASE_API_KEY"] = st.secrets["general"].get("SUPABASE_API_KEY")
            env_vars["ENABLE_WEB_SEARCH"] = st.secrets["general"].get("ENABLE_WEB_SEARCH")

            # Set environment variables for other modules that use os.getenv()
            if env_vars["OPENAI_API_KEY"]:
                os.environ["OPENAI_API_KEY"] = env_vars["OPENAI_API_KEY"]
            if env_vars["PINECONE_API_KEY"]:
                os.environ["PINECONE_API_KEY"] = env_vars["PINECONE_API_KEY"]
            if env_vars["PINECONE_ENVIRONMENT"]:
                os.environ["PINECONE_ENVIRONMENT"] = env_vars["PINECONE_ENVIRONMENT"]
            if env_vars["SUPABASE_URL"]:
                os.environ["SUPABASE_URL"] = env_vars["SUPABASE_URL"]
            if env_vars["SUPABASE_API_KEY"]:
                os.environ["SUPABASE_API_KEY"] = env_vars["SUPABASE_API_KEY"]
            if env_vars["ENABLE_WEB_SEARCH"]:
                os.environ["ENABLE_WEB_SEARCH"] = env_vars["ENABLE_WEB_SEARCH"]

            st.success("Successfully loaded secrets from Streamlit!")
        else:
            st.warning("No secrets found in Streamlit or 'general' section missing.")

            # Show what sections exist if any
            if hasattr(st, "secrets"):
                st.write("Available sections in secrets:")
                for section in st.secrets:
                    st.write(f"- {section}")
    except Exception as e:
        st.error(f"Error loading secrets from Streamlit: {str(e)}")

    # Fall back to environment variables for any missing values
    if not env_vars.get("OPENAI_API_KEY"):
        env_vars["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    if not env_vars.get("PINECONE_API_KEY"):
        env_vars["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    if not env_vars.get("PINECONE_ENVIRONMENT"):
        env_vars["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_ENVIRONMENT")
    if not env_vars.get("SUPABASE_URL"):
        env_vars["SUPABASE_URL"] = os.getenv("SUPABASE_URL")
    if not env_vars.get("SUPABASE_API_KEY"):
        env_vars["SUPABASE_API_KEY"] = os.getenv("SUPABASE_API_KEY")
    if not env_vars.get("ENABLE_WEB_SEARCH"):
        env_vars["ENABLE_WEB_SEARCH"] = os.getenv("ENABLE_WEB_SEARCH")

    # Display the keys that were loaded (without showing the actual values)
    st.write("### Environment Variables Status:")
    for key in env_vars:
        if env_vars[key]:
            st.write(f"- {key}: ✅ Present")
        else:
            st.write(f"- {key}: ❌ Missing")

# Check for required API keys
if not env_vars.get("OPENAI_API_KEY"):
    st.error("OpenAI API key is missing. Please add it to your Streamlit secrets.")
    st.stop()

if not env_vars.get("PINECONE_API_KEY"):
    st.error("Pinecone API key is missing. Please add it to your Streamlit secrets.")
    st.stop()

# Initialize main components
try:
    import openai
    import pinecone

    # Set up OpenAI API key
    openai.api_key = env_vars["OPENAI_API_KEY"]

    # Initialize Pinecone
    try:
        # Use the newer Pinecone initialization method
        pinecone.Pinecone(api_key=env_vars["PINECONE_API_KEY"])
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

    # Import necessary modules from the main app
    try:
        from rag_chatbot import (
            generate_response, create_chat_prompt, format_context_from_results,
            MODEL, MAX_HISTORY_MESSAGES, TOP_K_RESULTS, WEB_SEARCH_ENABLED, WEB_SEARCH_AVAILABLE
        )
        from query_interface import search_pinecone

        # Try to import Supabase functions, but provide fallbacks if not available
        try:
            from supabase_config import initialize_supabase, get_chat_history, format_chat_history_for_openai, save_message
            SUPABASE_AVAILABLE = True
        except ImportError:
            # Fallback to basic functionality without Supabase
            SUPABASE_AVAILABLE = False

            # Define dummy functions
            def initialize_supabase():
                st.warning("Supabase package not installed. Using in-memory chat history only.")
                return None

            def get_chat_history(supabase, session_id, limit=10):
                return []

            def format_chat_history_for_openai(messages):
                return messages

            def save_message(supabase, session_id, role, content):
                return {}
    except Exception as e:
        st.error(f"Failed to import required modules: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if "use_supabase" not in st.session_state:
        # Default to using Supabase if available and credentials exist
        st.session_state.use_supabase = SUPABASE_AVAILABLE and bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_API_KEY"))

    if "use_web_search" not in st.session_state:
        # Default to using web search if enabled in environment
        st.session_state.use_web_search = WEB_SEARCH_ENABLED and WEB_SEARCH_AVAILABLE

    # Try to initialize Supabase and load existing chat if we have a session ID
    if st.session_state.use_supabase and not st.session_state.messages:
        try:
            supabase = initialize_supabase()

            # Create a new session if we don't have one
            if not st.session_state.session_id:
                # Check URL parameters for session_id
                if "session_id" in st.query_params:
                    st.session_state.session_id = st.query_params["session_id"]

                    # Load chat history from Supabase
                    db_messages = get_chat_history(supabase, st.session_state.session_id)
                    if db_messages:
                        st.session_state.messages = db_messages
                        st.session_state.chat_history = format_chat_history_for_openai(db_messages)
                else:
                    # Generate a new session ID and create the session in Supabase
                    from supabase_config import create_chat_session
                    try:
                        session_id = create_chat_session(supabase)
                        st.session_state.session_id = session_id
                        st.info(f"Created new chat session: {session_id}")
                    except Exception as e:
                        st.error(f"Failed to create chat session: {str(e)}")
                        st.session_state.use_supabase = False
        except Exception as e:
            st.warning(f"Failed to connect to Supabase: {str(e)}")
            st.session_state.use_supabase = False

    # App description
    st.markdown("""
    This chatbot uses Retrieval Augmented Generation (RAG) to provide accurate information from the
    Unified Development Control and Promotion Regulations (UDCPR) for Maharashtra State.
    """)

    # Function to display chat messages
    def display_chat_message(role, content, avatar=None):
        with st.container():
            col1, col2 = st.columns([1, 12])
            with col1:
                if avatar:
                    st.markdown(f"<div class='avatar'>{avatar}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='message'>{content}</div>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])

    # Import and use our chat handler
    from chat_handler import handle_chat_input, add_conversation_buttons, add_footer

    # Handle chat input
    handle_chat_input()

    # Add conversation management buttons
    add_conversation_buttons()

    # Add footer
    add_footer()

except Exception as e:
    st.error("### Error Starting Application")
    st.write("An error occurred while starting the application:")
    st.code(str(e))

    st.write("### Detailed Error Information:")
    st.code(traceback.format_exc())

    st.write("### Environment Variables Status:")
    for var, value in env_vars.items():
        status = "Present" if value else "Missing"
        st.write(f"- {var}: {status}")

    st.write("### Python Information:")
    st.write(f"Python Version: {sys.version}")

    # Override the chat input function to prevent crashes
    def safe_chat_input():
        prompt = st.chat_input("Application is in error state. Please refresh the page.")
        if prompt:
            st.error("The application is currently in an error state. Please check the logs above and fix the configuration issues.")
        return None

    # Replace the standard chat_input with our safe version
    st.chat_input = safe_chat_input
