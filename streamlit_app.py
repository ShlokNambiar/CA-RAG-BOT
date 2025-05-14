"""
Streamlit entry point for UDCPR RAG Chatbot

This file serves as the entry point for Streamlit Cloud deployment.
It includes error handling for the main application.
"""

import streamlit as st
import os
import traceback
import sys

# First, check if environment variables are properly set
# Try to load from Streamlit secrets first, then fall back to environment variables
env_vars = {}

# Load from Streamlit secrets
try:
    # First, let's see what's in the secrets
    st.write("### Streamlit Secrets Structure:")
    if hasattr(st, "secrets"):
        st.write("Secrets object exists")
        if "general" in st.secrets:
            st.write("'general' section exists in secrets")
            # List all keys in the general section (without values)
            st.write("Keys in 'general' section:")
            for key in st.secrets["general"]:
                st.write(f"- {key}")
        else:
            st.write("'general' section does not exist in secrets")
            # Show what sections do exist
            st.write("Available sections:")
            for section in st.secrets:
                st.write(f"- {section}")
    else:
        st.write("No secrets object found in Streamlit")

    # Now try to load the secrets
    env_vars["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
    env_vars["PINECONE_API_KEY"] = st.secrets["general"]["PINECONE_API_KEY"]
    env_vars["PINECONE_ENVIRONMENT"] = st.secrets["general"]["PINECONE_ENVIRONMENT"]
    env_vars["SUPABASE_URL"] = st.secrets["general"]["SUPABASE_URL"]
    env_vars["SUPABASE_API_KEY"] = st.secrets["general"]["SUPABASE_API_KEY"]
    env_vars["ENABLE_WEB_SEARCH"] = st.secrets["general"]["ENABLE_WEB_SEARCH"]

    # Set environment variables for other modules that use os.getenv()
    os.environ["OPENAI_API_KEY"] = env_vars["OPENAI_API_KEY"]
    os.environ["PINECONE_API_KEY"] = env_vars["PINECONE_API_KEY"]
    os.environ["PINECONE_ENVIRONMENT"] = env_vars["PINECONE_ENVIRONMENT"]
    os.environ["SUPABASE_URL"] = env_vars["SUPABASE_URL"]
    os.environ["SUPABASE_API_KEY"] = env_vars["SUPABASE_API_KEY"]
    os.environ["ENABLE_WEB_SEARCH"] = env_vars["ENABLE_WEB_SEARCH"]

    st.success("Successfully loaded secrets from Streamlit!")

    # Display the keys that were loaded (without showing the actual values)
    st.write("### Loaded Keys:")
    for key in env_vars:
        if env_vars[key]:
            st.write(f"- {key}: ✅ Present")
        else:
            st.write(f"- {key}: ❌ Missing")
except Exception as e:
    st.error(f"Error loading secrets from Streamlit: {str(e)}")
    st.write("Falling back to environment variables...")

    # Fall back to environment variables
    env_vars["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    env_vars["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    env_vars["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_ENVIRONMENT")
    env_vars["SUPABASE_URL"] = os.getenv("SUPABASE_URL")
    env_vars["SUPABASE_API_KEY"] = os.getenv("SUPABASE_API_KEY")
    env_vars["ENABLE_WEB_SEARCH"] = os.getenv("ENABLE_WEB_SEARCH")

# Import the main app with error handling
try:
    # Import necessary modules with error handling
    import openai
    import pinecone

    # Set up OpenAI API key
    if env_vars["OPENAI_API_KEY"]:
        openai.api_key = env_vars["OPENAI_API_KEY"]
    else:
        st.error("OpenAI API key is missing. Please add it to your Streamlit secrets.")
        st.stop()

    # Initialize Pinecone
    if env_vars["PINECONE_API_KEY"]:
        try:
            # Use the newer Pinecone initialization method
            pinecone.Pinecone(api_key=env_vars["PINECONE_API_KEY"])
            # No need to store the client as it will be initialized again in query_interface.py
        except Exception as e:
            st.error(f"Failed to initialize Pinecone: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    else:
        st.error("Pinecone API key is missing. Please add it to your Streamlit secrets.")
        st.stop()

    # Now import the main app
    from chatbot_web import *

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
