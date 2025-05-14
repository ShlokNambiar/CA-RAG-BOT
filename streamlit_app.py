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
env_vars = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT"),
    "SUPABASE_URL": os.getenv("SUPABASE_URL"),
    "SUPABASE_API_KEY": os.getenv("SUPABASE_API_KEY"),
    "ENABLE_WEB_SEARCH": os.getenv("ENABLE_WEB_SEARCH")
}

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
    if env_vars["PINECONE_API_KEY"] and env_vars["PINECONE_ENVIRONMENT"]:
        try:
            pinecone.init(
                api_key=env_vars["PINECONE_API_KEY"],
                environment=env_vars["PINECONE_ENVIRONMENT"]
            )
        except Exception as e:
            st.error(f"Failed to initialize Pinecone: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()
    else:
        st.error("Pinecone API key or environment is missing. Please add them to your Streamlit secrets.")
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
