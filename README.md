# UDCPR RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot for the Unified Development Control and Promotion Regulations (UDCPR) document for Maharashtra State, India. The pipeline extracts text from the PDF, chunks it into manageable pieces, generates embeddings, and stores them in a vector database for semantic search. A Streamlit web interface allows users to ask questions about the UDCPR regulations and get accurate answers.

## Features

- **PDF Text Extraction**: Uses PyMuPDF for efficient extraction from large PDFs
- **Smart Text Chunking**: Implements 512-token chunks with 15% overlap for context preservation
- **Rate-Limited Embedding Generation**: Handles OpenAI API rate limits with batch processing
- **Vector Database Storage**: Uses Pinecone for efficient vector search
- **Checkpointing**: Supports resuming from checkpoints for long-running processes
- **Query Interface**: Simple interface for semantic search
- **Streamlit Web App**: User-friendly web interface for asking questions about UDCPR
- **OpenAI GPT-4o Integration**: Provides accurate, context-aware responses
- **Responsive Design**: Works on desktop and mobile devices

## Requirements

- Python 3.8+
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys (see `.env.example`)

## Usage

### Streamlit Web App

The easiest way to use this project is through the Streamlit web app:

1. Visit the deployed app at: [UDCPR Chatbot on Streamlit](https://udcpr-rag.streamlit.app/)
2. Type your question about UDCPR regulations in the chat input
3. Get an AI-generated response based on the UDCPR document

### Running the Full Pipeline Locally

If you want to run the data processing pipeline locally:

```bash
python main.py --pdf "UDCPR Updated 30.01.25 with earlier provisions & corrections.pdf"
```

### Running Individual Steps Locally

```bash
# Extract text from PDF
python pdf_extractor.py "UDCPR Updated 30.01.25 with earlier provisions & corrections.pdf" -o output/udcpr_extracted.json

# Chunk the text
python text_chunker.py output/udcpr_extracted.json -o output/udcpr_chunked.json

# Generate embeddings
python embeddings_generator.py output/udcpr_chunked.json -o output/udcpr_embeddings.json -c output/embeddings_checkpoint.json

# Upload to Pinecone
python pinecone_uploader.py output/udcpr_embeddings.json -c output/upload_checkpoint.json
```

### Querying the RAG System Locally

```bash
# Interactive query mode
python main.py --query

# Single query
python query_interface.py "What are the building height regulations?"
```

### Running the Streamlit App Locally

```bash
streamlit run udcpr_chatbot_streamlit.py
```

## Pipeline Components

1. **PDF Extraction** (`pdf_extractor.py`): Extracts text with page numbers and metadata
2. **Text Chunking** (`text_chunker.py`): Splits text into semantic chunks with overlap
3. **Embeddings Generation** (`embeddings_generator.py`): Creates vector embeddings with rate limit handling
4. **Pinecone Upload** (`pinecone_uploader.py`): Uploads vectors to Pinecone with metadata
5. **Query Interface** (`query_interface.py`): Provides semantic search functionality

## Optimization Features

- **Rate Limit Handling**: Implements delays and retries to avoid API rate limits
- **Batch Processing**: Processes data in batches to optimize API calls
- **Token Calculation**: Validates token counts before API calls
- **Checkpointing**: Saves progress to resume long-running processes
- **Error Recovery**: Handles errors gracefully with progress saving

## Streamlit Deployment

This project is deployed on Streamlit Cloud. The deployment uses:

- **Streamlit Secrets**: For securely storing API keys
- **Pinecone Vector Database**: For storing and retrieving document embeddings
- **OpenAI API**: For generating embeddings and chat completions
- **Responsive UI**: For a good user experience on all devices

To deploy your own version:

1. Fork this repository
2. Create a Streamlit account at [streamlit.io](https://streamlit.io)
3. Create a new app and connect it to your forked repository
4. Add your API keys to Streamlit secrets with the following structure:
   ```
   [general]
   OPENAI_API_KEY = "your-openai-api-key"
   PINECONE_API_KEY = "your-pinecone-api-key"
   PINECONE_ENVIRONMENT = "your-pinecone-environment"
   SUPABASE_URL = "your-supabase-url"
   SUPABASE_API_KEY = "your-supabase-api-key"
   ENABLE_WEB_SEARCH = "true"
   ```

## License

MIT
