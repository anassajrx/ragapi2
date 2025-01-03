# Directory structure:
# /your-project
# ├── app
# │   └── main.py
# ├── requirements.txt
# └── Dockerfile


# main.py (place in /app directory)
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from typing import List

# Configure the API key for Gemini
GEMINI_API_KEY = os.getenv("AIzaSyCgH9GXtpKmt8fue04DkzFrmmYV11XDxGI")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="RAG PDF API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vectorstore and embeddings
def initialize_vectorstore():
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}
    )
    # Use environment variable for persistent storage path
    persist_directory = os.getenv("STORAGE_PATH", "./chroma_db_nccn")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    return vectorstore

def process_pdf(file_path, vectorstore):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)
    vectorstore.add_documents(docs)
    return len(docs)

def generate_rag_prompt(query, context):
    escaped = context.replace("'","").replace('"', "").replace("\n"," ")
    prompt = f"""
    You are a helpful and informative bot that answers questions using text from the reference context included below.
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
    strike a friendly and conversational tone.
    If the context is irrelevant to the answer, you may ignore it.
    
    QUESTION: '{query}'
    CONTEXT: '{context}'
    
    ANSWER:
    """
    return prompt

def get_relevant_context(query, vectorstore):
    context = ""
    search_results = vectorstore.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generate_answer(prompt):
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

@app.get("/")
async def root():
    return {"message": "RAG PDF API is running"}

@app.post("/upload/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    vectorstore = initialize_vectorstore()
    total_chunks = 0
    processed_files = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
            
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            num_chunks = process_pdf(tmp_file.name, vectorstore)
            total_chunks += num_chunks
            processed_files.append({
                "filename": file.filename,
                "chunks_created": num_chunks
            })
            os.unlink(tmp_file.name)
    
    return {
        "status": "success",
        "processed_files": processed_files,
        "total_chunks": total_chunks
    }

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    vectorstore = initialize_vectorstore()
    context = get_relevant_context(question, vectorstore)
    prompt = generate_rag_prompt(question, context)
    answer = generate_answer(prompt)
    
    return {
        "question": question,
        "answer": answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
