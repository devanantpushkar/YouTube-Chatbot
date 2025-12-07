
# YouTube Video Chatbot

A powerful AI-powered chatbot that allows you to ask questions about YouTube videos by analyzing their transcripts using RAG (Retrieval-Augmented Generation) technology.

## Overview

This application fetches YouTube video transcripts, processes them using advanced NLP techniques, and enables users to ask questions about the video content. The system uses:

- **LangChain** for orchestrating the RAG pipeline
- **FAISS** for efficient vector storage and similarity search
- **HuggingFace Embeddings** for text embeddings
- **Groq's LLaMA 3** for natural language understanding and response generation
- **Streamlit** for an interactive web interface
- **FastAPI** for REST API endpoints

## Features

- **Automatic Transcript Extraction**: Fetches transcripts from YouTube videos
- **Intelligent Q&A**: Ask questions about video content and get accurate answers
- **Context-Aware Responses**: Uses RAG to provide answers based only on video content
- **Fast Retrieval**: FAISS vector store for efficient similarity search
- **Dual Interface**: Both Streamlit UI and REST API available
- **Structured Answers**: Responses formatted in clear bullet points

## Technologies Used

- **Python 3.x**
- **LangChain** - Framework for LLM applications
- **Streamlit** - Web UI framework
- **FastAPI** - REST API framework
- **FAISS** - Vector similarity search
- **HuggingFace Transformers** - Sentence embeddings
- **Groq API** - LLaMA 3 language model
- **YouTube Transcript API** - Transcript extraction

## Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key (get it from [Groq Console](https://console.groq.com))

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd FinalAssignment
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### Running the Streamlit App

```bash
streamlit run app1.py
```

Or using Python module syntax:
```bash
python -m streamlit run app1.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Streamlit Interface

1. Enter a YouTube video URL or video ID (e.g., `Gfr50f6ZBvo`)
2. Type your question about the video
3. Click **Submit**
4. Wait for the AI to process and generate an answer

### Using the REST API

The application also includes a FastAPI endpoint for programmatic access:

**Endpoint**: `POST /ask`

**Request Body**:
```json
{
  "video_id": "Gfr50f6ZBvo",
  "question": "Who is Demis?"
}
```

**Response**:
```json
{
  "answer": "• Demis Hassabis is the CEO and co-founder of DeepMind..."
}
```

## Architecture

### RAG Pipeline

1. **Transcript Extraction**: Fetches video transcript using YouTube Transcript API
2. **Text Chunking**: Splits transcript into manageable chunks (1000 chars with 200 overlap)
3. **Embedding**: Converts chunks to vectors using `sentence-transformers/all-MiniLM-L6-v2`
4. **Vector Storage**: Stores embeddings in FAISS for fast retrieval
5. **Query Processing**: Retrieves top 4 most relevant chunks for user questions
6. **Answer Generation**: LLaMA 3 generates contextual answers from retrieved chunks

### Key Components

- **`extract_video_id()`**: Parses YouTube URLs to extract video IDs
- **`get_transcript()`**: Fetches and caches video transcripts
- **`prepare_chain()`**: Builds the RAG chain with retriever and LLM
- **Streamlit UI**: Interactive web interface
- **FastAPI Endpoints**: REST API for integration

## Project Structure

```
FinalAssignment/
├── app1.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
└── README.md           # This file
```

## Configuration

### Model Settings

The application uses the following default settings:

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `llama3-8b-8192` (via Groq)
- **Temperature**: 0.2 (for more deterministic responses)
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Retrieval**: Top 4 most similar chunks

You can modify these settings in `app1.py` as needed.

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**2. Transcript Not Available**
- Some videos have disabled transcripts
- Try a different video or check if captions are available

**3. API Key Error**
- Ensure your `.env` file contains a valid `GROQ_API_KEY`
- Check that the `.env` file is in the project root directory

**4. FAISS Installation Issues**
- On Windows, use `faiss-cpu` (already in requirements.txt)
- On Linux/Mac, you may need to install system dependencies

## Example Questions

Try these example questions with a video:

- "What is the main topic of this video?"
- "Who are the key people mentioned?"
- "What are the main points discussed?"
- "Can you summarize the conclusion?"

## Contributing

Feel free to fork this project and submit pull requests for improvements!

## License

This project is open source and available for educational purposes.

## Acknowledgments

- **LangChain** for the RAG framework
- **Groq** for fast LLM inference
- **HuggingFace** for embedding models
- **Streamlit** for the amazing UI framework

## Support

For issues or questions, please create an issue in the project repository.

---

**Made with LangChain and Streamlit**
