"""
Streamlit entry point for UDCPR RAG Chatbot

This file serves as the entry point for Streamlit Cloud deployment.
It includes error handling and debugging information.
"""

import streamlit as st
import traceback
import os
import sys

# Set up page for error reporting
st.set_page_config(
    page_title="UDCPR RAG Chatbot",
    page_icon="📚",
    layout="centered"
)

try:
    # Display environment variables (without sensitive values)
    st.write("### Environment Check")
    env_vars = {
        "OPENAI_API_KEY": "Present" if os.getenv("OPENAI_API_KEY") else "Missing",
        "PINECONE_API_KEY": "Present" if os.getenv("PINECONE_API_KEY") else "Missing",
        "PINECONE_ENVIRONMENT": "Present" if os.getenv("PINECONE_ENVIRONMENT") else "Missing",
        "SUPABASE_URL": "Present" if os.getenv("SUPABASE_URL") else "Missing",
        "SUPABASE_API_KEY": "Present" if os.getenv("SUPABASE_API_KEY") else "Missing",
        "ENABLE_WEB_SEARCH": "Present" if os.getenv("ENABLE_WEB_SEARCH") else "Missing"
    }

    st.write("Environment Variables Status:")
    for var, status in env_vars.items():
        st.write(f"- {var}: {status}")

    # Import the main app
    st.write("### Importing main application...")
    from chatbot_web import *
    st.write("Import successful!")

except Exception as e:
    st.error("### Error Starting Application")
    st.write("An error occurred while starting the application:")
    st.code(str(e))

    st.write("### Detailed Error Information:")
    st.code(traceback.format_exc())

    st.write("### Python Information:")
    st.write(f"Python Version: {sys.version}")
    st.write(f"Python Path: {sys.executable}")

    # List installed packages
    st.write("### Installed Packages:")
    try:
        import pkg_resources
        packages = sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
        st.code("\n".join(packages))
    except:
        st.write("Could not retrieve package information")
