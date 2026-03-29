import os
import uuid
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage: Maps doc_id to extracted text
# Note: For production, replace this with Redis or a database.
document_store = {}

# ENV variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_URL = os.getenv("NVIDIA_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama3-70b-instruct")

class QuestionRequest(BaseModel):
    doc_id: str
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    contents = await file.read()
    reader = PyPDF2.PdfReader(BytesIO(contents))

    if len(reader.pages) > 10:
        raise HTTPException(status_code=400, detail="PDF exceeds 10 pages limit")

    text = "".join([page.extract_text() or "" for page in reader.pages]).strip()

    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF")

    # Generate a unique ID for this document and store it
    doc_id = str(uuid.uuid4())
    document_store[doc_id] = text

    return {"message": "PDF processed successfully", "doc_id": doc_id}

@app.post("/chat")
async def chat(request: QuestionRequest):
    # Retrieve the specific document text
    pdf_text = document_store.get(request.doc_id)
    
    if not pdf_text:
        raise HTTPException(status_code=404, detail="Document not found or expired. Please upload again.")

    if not NVIDIA_API_KEY or not NVIDIA_URL:
        raise HTTPException(status_code=500, detail="Server misconfiguration: missing API credentials.")

    prompt = f"""
    Answer based only on the PDF content below.

    PDF Content:
    {pdf_text[:8000]}  # Still limiting characters to avoid breaking token limits

    Question:
    {request.question}
    """

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 500
    }

    # Use httpx for asynchronous HTTP requests
    async with httpx.AsyncClient() as client:
        try:
            # timeout=30.0 gives the LLM up to 30 seconds to reply
            response = await client.post(NVIDIA_URL, headers=headers, json=payload, timeout=30.0)
            
            # This will raise an exception for 4xx and 5xx status codes
            response.raise_for_status()
            
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            return {"answer": answer}

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"LLM API Error: {e.response.text}")
        except (KeyError, IndexError):
            raise HTTPException(status_code=500, detail="Unexpected response structure from LLM API.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")