from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Configuration constants
GEMEINI_API_KEY = "AIzaSyCgH9GXtpKmt8fue04DkzFrmmYV11XDxGI"
genai.configure(api_key=GEMEINI_API_KEY)

# Initialize vectorstore and embeddings
def initialize_vectorstore():
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectorstore = Chroma(
        persist_directory="./chroma_db_nccn",  # Directory where the vector store is saved
        embedding_function=embedding_function
    )
    return vectorstore

# Function to add documents to the vector store
def add_documents_to_vectorstore(docs, vectorstore):
    vectorstore.add_documents(docs)

# Process all uploaded PDFs
def process_all_pdfs(uploaded_files, vectorstore):
    docs = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file)  # Write the uploaded binary data directly
            temp_file_path = tmp_file.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        pdf_docs = loader.load()

        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        pdf_docs = text_splitter.split_documents(pdf_docs)
        docs.extend(pdf_docs)

        # Optionally, delete the temporary file
        os.remove(temp_file_path)

    # Add all documents to the vector store at once
    add_documents_to_vectorstore(docs, vectorstore)

# Generate RAG prompt
def generate_rag_prompt(query, context):
    escaped = context.replace("'","").replace('"', "").replace("\n"," ")
    prompt = ("""
You are a helpful and informative bot that answers questions using text from the reference context included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and conversational tone. \
  If the context is irrelevant to the answer, you may ignore it.
                QUESTION: '{query}'
                CONTEXT: '{context}'
              
              ANSWER:
              """).format(query=query, context=context)
    return prompt

# Get relevant context from the vectorstore
def get_relevant_context_from_db(query):
    vectorstore = initialize_vectorstore()
    context = ""
    search_results = vectorstore.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

# Generate an answer from the prompt
def generate_answer(prompt):
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

# Endpoint to upload PDFs and vectorize them
@app.post("/upload_pdfs/")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        uploaded_files.append(await file.read())
    
    # Initialize vectorstore
    vectorstore = initialize_vectorstore()

    # Process and vectorize the uploaded PDFs
    process_all_pdfs(uploaded_files, vectorstore)

    return {"message": f"{len(files)} PDF(s) uploaded and vectorized successfully."}

# Endpoint to answer a question based on the vectorized PDFs
@app.post("/ask_question/")  
async def ask_question(question: str = Form(...)):
    vectorstore = initialize_vectorstore()
    context = get_relevant_context_from_db(question)
    prompt = generate_rag_prompt(query=question, context=context)
    answer = generate_answer(prompt)
    return {"question": question, "answer": answer}

# Homepage route displaying "RAG PORT NADIA"
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG Port Nadia</title>
        </head>
        <body>
            <h1>RAG PORT NADIA</h1>
        </body>
    </html>
    """
