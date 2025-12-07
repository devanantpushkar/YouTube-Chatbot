import os
from dotenv import load_dotenv
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Functions ---

def extract_video_id(url_or_id):
    """Extracts video ID from full YouTube URL or returns the input if already a valid ID"""
    if len(url_or_id) == 11 and " " not in url_or_id:
        return url_or_id  # already a valid video ID
    try:
        parsed_url = urlparse(url_or_id)
        if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.hostname == "youtu.be":
            return parsed_url.path.lstrip("/")
    except:
        return None
    return None

@st.cache_data
def get_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en"])
        raw_transcript = fetched.to_raw_data()
        return " ".join(entry["text"] for entry in raw_transcript)
    except TranscriptsDisabled:
        return "Transcript not available."
    except Exception as e:
        return f"Error: {str(e)}"


@st.cache_resource
def prepare_chain(transcript):
    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Prompt template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Present the answer in well-structured bullet points for clarity. Do not say things like 'Here is the answer:'.

        Transcript: {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    # LLM setup
    llm = ChatOpenAI(
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        model_name="llama3-8b-8192",
        temperature=0.2,
    )

    # Formatting chain
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }) | prompt | llm | StrOutputParser()
    )

    return chain

# --- Streamlit App UI ---
st.title("🎥 YouTube Video Chatbot")

video_url = st.text_input("Enter YouTube video URL (or ID)", value="Gfr50f6ZBvo")
question = st.text_input("Ask a question about the video", value="Who is Demis?")

if st.button("Submit"):
    video_id = extract_video_id(video_url)
    
    if not video_id:
        st.error("❌ Invalid YouTube video URL or ID.")
    else:
        with st.spinner("📽️ Fetching transcript and building system..."):
            transcript = get_transcript(video_id)
            
            if transcript.startswith("Error") or transcript.startswith("Transcript not"):
                st.error(transcript)
            else:
                chain = prepare_chain(transcript)
                with st.spinner("💬 Generating answer..."):
                    response = chain.invoke(question)
                    st.markdown("### 💡 Answer")
                    st.success(response)


import json
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    video_id = extract_video_id(data.get("video_id", ""))
    question = data.get("question", "")

    transcript = get_transcript(video_id)

    if not transcript:
        return JSONResponse({"answer": "Transcript could not be retrieved (possibly unavailable or disabled)."})

    if isinstance(transcript, str) and (transcript.startswith("Error") or transcript.startswith("Transcript not")):
        return JSONResponse({"answer": transcript})

    try:
        chain = prepare_chain(transcript)
        response = chain.invoke(question)
        return JSONResponse({"answer": response})
    except Exception as e:
        return JSONResponse({"answer": f"An error occurred while processing your question: {str(e)}"})

