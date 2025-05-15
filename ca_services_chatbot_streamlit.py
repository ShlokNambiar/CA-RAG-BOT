"""
CA Services Chatbot Streamlit App

A completely standalone Streamlit app for the CA Services RAG Chatbot.
This file doesn't import any code from other files to avoid set_page_config conflicts.
"""

import streamlit as st
import os
import sys
import traceback
import time
import openai
import uuid
from datetime import datetime
from supabase import create_client, Client

# IMPORTANT: Set page config first before any other Streamlit commands
st.set_page_config(
    page_title="CA Services RAG Chatbot",
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
st.title("📚 CA Services Document Assistant")

# Load environment variables
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
        pc = pinecone.Pinecone(api_key=env_vars["PINECONE_API_KEY"])

        # Constants
        INDEX_NAME = "udcpr-rag-index"  # Using the existing index which contains both UDCPR and CA Services data
        EMBEDDING_MODEL = "text-embedding-3-small"
        EMBEDDING_DIMENSIONS = 1024
        MODEL = "gpt-4o"
        MAX_HISTORY_MESSAGES = 10
        TOP_K_RESULTS = 5

        # Check if the index exists
        index_list = [index.name for index in pc.list_indexes()]
        if INDEX_NAME not in index_list:
            st.error(f"Pinecone index '{INDEX_NAME}' does not exist. Please create it first.")
            st.stop()

        # Connect to the index
        index = pc.Index(INDEX_NAME)

    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

    # Initialize Supabase
    try:
        # Check if Supabase credentials are available
        if not env_vars.get("SUPABASE_URL") or not env_vars.get("SUPABASE_API_KEY"):
            st.warning("Supabase credentials are missing. Chat history will not be saved to the database.")
            supabase_client = None
        else:
            # Initialize Supabase client
            supabase_url = env_vars["SUPABASE_URL"]
            supabase_key = env_vars["SUPABASE_API_KEY"]

            try:
                # Try the standard initialization first
                supabase_client = create_client(supabase_url, supabase_key)
                st.success("Successfully connected to Supabase!")
            except Exception as e:
                st.warning(f"Standard Supabase initialization failed: {str(e)}")

                # Set up a simple dictionary-based mock client for storing messages
                # This will store messages in memory during the session
                st.warning("Using in-memory storage for chat history instead of Supabase")

                class MockSupabaseClient:
                    def __init__(self):
                        self.tables = {
                            "chat_messages": [],
                            "chat_memories": []
                        }

                    def table(self, table_name):
                        return MockSupabaseTable(self.tables.get(table_name, []))

                class MockSupabaseTable:
                    def __init__(self, data):
                        self.data = data
                        self.current_record = None

                    def insert(self, record):
                        self.current_record = record
                        return self

                    def execute(self):
                        if self.current_record:
                            self.data.append(self.current_record)
                            return {"data": self.current_record}
                        return {"data": None}

                supabase_client = MockSupabaseClient()
                st.info("Using in-memory storage for chat history. Data will not persist between sessions.")

            # Initialize session ID if not already set
            if "session_id" not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())

            # Function to save message to Supabase
            def save_message_to_supabase(role, content):
                """Save a message to the Supabase chat_messages table."""
                if supabase_client:
                    try:
                        message_data = {
                            "session_id": st.session_state.session_id,
                            "role": role,
                            "content": content,
                            "created_at": datetime.now().isoformat()
                        }
                        supabase_client.table("chat_messages").insert(message_data).execute()
                    except Exception as e:
                        st.error(f"Failed to save message to Supabase: {str(e)}")

            # Function to save chat memory to Supabase
            def save_memory_to_supabase(query, response):
                """Save a chat memory to the Supabase chat_memories table."""
                if supabase_client:
                    try:
                        memory_data = {
                            "session_id": st.session_state.session_id,
                            "query": query,
                            "response": response,
                            "created_at": datetime.now().isoformat()
                        }
                        supabase_client.table("chat_memories").insert(memory_data).execute()
                    except Exception as e:
                        st.error(f"Failed to save memory to Supabase: {str(e)}")

    except Exception as e:
        st.error(f"Failed to initialize Supabase: {str(e)}")
        st.code(traceback.format_exc())
        supabase_client = None

    # App description
    st.markdown("""
    This chatbot uses Retrieval Augmented Generation (RAG) to provide accurate information about
    CA Services and related accounting, tax, and financial advisory topics.
    """)

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])

    # Helper functions
    def get_query_embedding(query):
        """Get embedding for a query string."""
        response = openai.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS
        )
        return response.data[0].embedding

    def search_pinecone(query, top_k=5, include_metadata=True):
        """Search Pinecone index with a query string."""
        # Get query embedding
        query_embedding = get_query_embedding(query)

        # Search Pinecone
        search_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata
        )

        return search_response["matches"]

    def format_context_from_results(results):
        """Format search results into a context string for the LLM."""
        if not results:
            return "Use your general knowledge to answer this question."

        context_parts = []

        for result in results:
            score = result.get("score", 0)
            text = result.get("metadata", {}).get("text", "")
            source = result.get("metadata", {}).get("source", "Unknown")
            page = result.get("metadata", {}).get("page_num", "Unknown")

            if text and score > 0.3:  # Lower threshold to include more results
                context_parts.append(f"[Source: {source}, Page: {page}]\n{text}\n")

        if not context_parts:
            return "Use your general knowledge to answer this question."

        return "\n".join(context_parts)

    def create_chat_prompt(query, context, web_search_context=None, chat_history=None):
        """Create a chat prompt for the LLM."""
        messages = []

        # System message with instructions
        system_message = """You are an expert assistant for CA Services, specializing in accounting, tax, and financial advisory services.
Your task is to provide accurate, helpful information about accounting principles, tax regulations, financial planning, and advisory services based on the CA Services documents.

When answering:
1. Base your answers primarily on the provided context from the CA Services documents
2. If the context contains the information, provide detailed, accurate answers
3. If the context doesn't contain enough information, provide a helpful response based on your general knowledge without explicitly stating that the documents don't contain the information
4. Be concise but thorough
5. Use bullet points or numbered lists for clarity when appropriate
6. If asked about something outside the scope of CA Services, still provide helpful information related to accounting, tax, and financial advisory topics
7. If web search results are provided, you may use them to supplement your answer, but clearly indicate when information comes from external sources
8. Never start your response with phrases like "It seems that the CA Services documents do not contain..." or similar disclaimers

Remember, your goal is to help users understand accounting, tax, and financial advisory services accurately and provide valuable information regardless of what's in the context."""

        messages.append({"role": "system", "content": system_message})

        # Add chat history if available
        if chat_history:
            # Only include a limited number of recent messages
            recent_history = chat_history[-MAX_HISTORY_MESSAGES:]
            messages.extend(recent_history)

        # Construct the user message with context
        if context == "Use your general knowledge to answer this question.":
            user_message = f"Question: {query}\n\nPlease provide a helpful and accurate response based on your knowledge of accounting, tax, and financial advisory services."
        else:
            user_message = f"Question: {query}\n\nHere is some relevant information that may help:\n{context}"

        # Add web search context if available
        if web_search_context:
            user_message += f"\n\nAdditional information from web search:\n{web_search_context}"

        messages.append({"role": "user", "content": user_message})

        return messages

    # Chat input
    if prompt := st.chat_input("Ask a question about CA Services..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)

        # Display assistant response with streaming
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            try:
                # Get relevant context from Pinecone
                results = search_pinecone(prompt, top_k=TOP_K_RESULTS)
                context = format_context_from_results(results)

                # Create chat prompt
                messages = create_chat_prompt(prompt, context, None, st.session_state.chat_history)

                # Stream the response
                message_placeholder.empty()
                full_response = ""

                stream = openai.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=800,
                    stream=True
                )

                # Display the streaming response
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)  # Small delay for smoother streaming

                # Display final response without cursor
                message_placeholder.markdown(full_response)

                # Update in-memory chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

                # Limit chat history
                if len(st.session_state.chat_history) > MAX_HISTORY_MESSAGES:
                    st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_MESSAGES:]

                # Add assistant message to display history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Save to Supabase if client is initialized
                if 'supabase_client' in locals() and supabase_client:
                    # Save individual messages
                    save_message_to_supabase("user", prompt)
                    save_message_to_supabase("assistant", full_response)

                    # Save as a memory pair
                    save_memory_to_supabase(prompt, full_response)

            except Exception as e:
                message_placeholder.markdown(f"Error: {str(e)}")
                st.error(traceback.format_exc())

    # Add buttons for conversation management
    col1, col2 = st.columns(2)

    # Clear conversation button
    if col1.button("Clear Conversation"):
        # Reset all conversation state
        st.session_state.chat_history = []
        st.session_state.messages = []

        # Generate a new session ID for Supabase
        if "session_id" in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        st.experimental_rerun()

    # Footer
    st.markdown("""
    ---
    *Powered by OpenAI GPT-4o and Pinecone*
    """)

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
