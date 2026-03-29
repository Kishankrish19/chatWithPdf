# import os
# import uuid
# import httpx
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import PyPDF2
# from io import BytesIO
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# # CORS
# FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # In-memory storage: Maps doc_id to extracted text
# # Note: For production, replace this with Redis or a database.
# document_store = {}

# # ENV variables
# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
# NVIDIA_URL = os.getenv("NVIDIA_URL")
# MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama3-70b-instruct")

# class QuestionRequest(BaseModel):
#     doc_id: str
#     question: str

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     # Validate file type
#     if file.content_type != "application/pdf":
#         raise HTTPException(status_code=400, detail="File must be a PDF")

#     contents = await file.read()
#     reader = PyPDF2.PdfReader(BytesIO(contents))

#     if len(reader.pages) > 10:
#         raise HTTPException(status_code=400, detail="PDF exceeds 10 pages limit")

#     text = "".join([page.extract_text() or "" for page in reader.pages]).strip()

#     if not text:
#         raise HTTPException(status_code=400, detail="Could not extract text from the PDF")

#     # Generate a unique ID for this document and store it
#     doc_id = str(uuid.uuid4())
#     document_store[doc_id] = text

#     return {"message": "PDF processed successfully", "doc_id": doc_id}

# @app.post("/chat")
# async def chat(request: QuestionRequest):
#     # Retrieve the specific document text
#     pdf_text = document_store.get(request.doc_id)
    
#     if not pdf_text:
#         raise HTTPException(status_code=404, detail="Document not found or expired. Please upload again.")

#     if not NVIDIA_API_KEY or not NVIDIA_URL:
#         raise HTTPException(status_code=500, detail="Server misconfiguration: missing API credentials.")

#     prompt = f"""
#     Answer based only on the PDF content below.

#     PDF Content:
#     {pdf_text[:8000]}  # Still limiting characters to avoid breaking token limits

#     Question:
#     {request.question}
#     """

#     headers = {
#         "Authorization": f"Bearer {NVIDIA_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.5,
#         "max_tokens": 500
#     }

#     # Use httpx for asynchronous HTTP requests
#     async with httpx.AsyncClient() as client:
#         try:
#             # timeout=30.0 gives the LLM up to 30 seconds to reply
#             response = await client.post(NVIDIA_URL, headers=headers, json=payload, timeout=30.0)
            
#             # This will raise an exception for 4xx and 5xx status codes
#             response.raise_for_status()
            
#             data = response.json()
#             answer = data["choices"][0]["message"]["content"]
#             return {"answer": answer}

#         except httpx.HTTPStatusError as e:
#             raise HTTPException(status_code=e.response.status_code, detail=f"LLM API Error: {e.response.text}")
#         except (KeyError, IndexError):
#             raise HTTPException(status_code=500, detail="Unexpected response structure from LLM API.")
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


import os
import uuid
import time
import logging
import httpx
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_logger")

app = FastAPI(title="Ultimate AI Backend", description="Multi-purpose API Hub", version="2.0")

# CORS setup for frontend
FRONTEND_URLS = os.getenv("FRONTEND_URLS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. ADVANCED MIDDLEWARE & ERROR HANDLING
# ==========================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Feature: Logs how long every request takes. Great for monitoring LLM speed."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Time: {process_time:.2f}s | Status: {response.status_code}")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Feature: Stops the server from completely crashing on unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "message": str(exc)})


# ==========================================
# 3. SECURITY & MEMORY MANAGEMENT
# ==========================================
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "super-secret-key")

def require_api_key(request: Request):
    """Feature: Plug this into any route to instantly require a password header."""
    key = request.headers.get("X-API-Key")
    if key != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized access")
    return key

document_store = {}
EXPIRATION_SECONDS = 3600  # 1 hour

def cleanup_memory():
    """Feature: Keeps Render RAM clean by deleting old data."""
    now = time.time()
    expired = [k for k, v in document_store.items() if now - v["timestamp"] > EXPIRATION_SECONDS]
    for k in expired:
        del document_store[k]
        logger.info(f"Cleaned up expired document: {k}")


# ==========================================
# 4. MODULE 1: SYSTEM ADMIN ROUTER
# ==========================================
sys_router = APIRouter(prefix="/system", tags=["System"])

@sys_router.get("/health")
def health_check():
    """Use this for your 5-minute pinging robot!"""
    return {"status": "Online and healthy", "uptime_ping": time.time()}

@sys_router.get("/stats", dependencies=[Depends(require_api_key)])
def server_stats():
    """Feature: A locked route only YOU can access to see how much memory is used."""
    return {"active_documents": len(document_store)}


# ==========================================
# 5. MODULE 2: PDF AI ROUTER
# ==========================================
pdf_router = APIRouter(prefix="/pdf", tags=["PDF Engine"])

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_URL = os.getenv("NVIDIA_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama3-70b-instruct")

class QuestionRequest(BaseModel):
    doc_id: str
    question: str

@pdf_router.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    background_tasks.add_task(cleanup_memory) # Trigger cleanup

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDFs are allowed.")

    contents = await file.read()
    if len(contents) > 8 * 1024 * 1024: # Feature: 8MB limit to save CPU
        raise HTTPException(status_code=400, detail="File exceeds 8MB limit.")

    reader = PyPDF2.PdfReader(BytesIO(contents))
    text = "".join([page.extract_text() or "" for page in reader.pages]).strip()

    if not text:
        raise HTTPException(status_code=400, detail="No readable text found.")

    doc_id = str(uuid.uuid4())
    document_store[doc_id] = {"text": text, "timestamp": time.time(), "name": file.filename}

    return {"message": "Success", "doc_id": doc_id, "pages": len(reader.pages)}

@pdf_router.post("/chat")
async def chat_pdf(request: QuestionRequest):
    doc_data = document_store.get(request.doc_id)
    if not doc_data:
        raise HTTPException(status_code=404, detail="Document expired. Please re-upload.")

    prompt = f"Based on this text from '{doc_data['name']}':\n{doc_data['text'][:8000]}\n\nAnswer: {request.question}"

    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 500}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(NVIDIA_URL, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            return {"answer": response.json()["choices"][0]["message"]["content"]}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="LLM Provider Error")


# ==========================================
# 6. MODULE 3: AI TOOLBELT (Ready for expansion)
# ==========================================
tools_router = APIRouter(prefix="/tools", tags=["AI Tools"])

@tools_router.post("/summarize-text")
async def summarize_raw_text(text: str):
    """Feature: A quick endpoint for text summarization without needing a PDF."""
    if not text: return {"error": "Provide text"}
    # You can hook this up to the NVIDIA API just like the chat endpoint!
    return {"status": "Ready to connect to LLM", "preview": text[:50] + "..."}


# ==========================================
# 7. ASSEMBLE THE APP
# ==========================================
app.include_router(sys_router)
app.include_router(pdf_router)
app.include_router(tools_router)
