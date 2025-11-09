"""
FastAPI Backend for RADA Chatbot
Provides REST API endpoints for frontend integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from chat_groq import COLLECTION_NAME

# Import your chatbot modules
from chat_with_compute_gpt import (
    init_chroma,
    init_embedding_model,
    init_reranker,
    init_openai,
    query_chatbot_with_compute as query_openai
)

from chat_groq import (
    init_groq,
    query_chatbot_with_compute as query_groq
)

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="RADA Petroleum Engineering Assistant API",
    description="AI-powered chatbot for petroleum production data analysis",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = "openai"  # "openai" or "groq"
    debug: Optional[bool] = False

class QueryResponse(BaseModel):
    answer: str
    model_used: str
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str
    models_available: list

# Global variables for cached models
collection = None
embedding_model = None
reranker = None
openai_client = None
groq_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize all models on startup (cached)"""
    global collection, embedding_model, reranker, openai_client, groq_client
    
    print("🚀 Initializing RADA Assistant API...")
    
    try:
        # Load vector database and models (shared by both)
        collection = init_chroma()
        embedding_model = init_embedding_model()
        reranker = init_reranker()
        
        # Load LLM clients
        try:
            openai_client = init_openai()
        except Exception as e:
            print(f"⚠️ OpenAI initialization failed: {e}")
            openai_client = None
        
        try:
            groq_client = init_groq()
        except Exception as e:
            print(f"⚠️ Groq initialization failed: {e}")
            groq_client = None
        
        print("✅ API Server ready!")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    models_available = []
    if openai_client:
        models_available.append("openai")
    if groq_client:
        models_available.append("groq")
    
    return HealthResponse(
        status="online",
        message="RADA Assistant API is running",
        models_available=models_available
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    models_available = []
    if openai_client:
        models_available.append("openai")
    if groq_client:
        models_available.append("groq")
    
    if not models_available:
        raise HTTPException(status_code=503, detail="No LLM models available")
    
    return HealthResponse(
        status="healthy",
        message=f"Vector DB: {collection.count()} records loaded",
        models_available=models_available
    )

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Main query endpoint
    
    Parameters:
    - question: User's question
    - model: "openai" or "groq" (default: "openai")
    - debug: Enable debug mode (default: False)
    
    Returns:
    - answer: Generated response
    - model_used: Which model was used
    - status: "success" or "error"
    """
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Route to appropriate model
        if request.model.lower() == "groq":
            if not groq_client:
                raise HTTPException(status_code=503, detail="Groq model not available")
            
            answer = query_groq(
                collection=collection,
                embedding_model=embedding_model,
                reranker=reranker,
                groq_client=groq_client,
                user_question=request.question,
                debug=request.debug
            )
            model_used = "groq"
            
        else:  # default to OpenAI
            if not openai_client:
                raise HTTPException(status_code=503, detail="OpenAI model not available")
            
            answer = query_openai(
                collection=collection,
                embedding_model=embedding_model,
                reranker=reranker,
                openai_client=openai_client,
                user_question=request.question,
                debug=request.debug
            )
            model_used = "openai"
        
        return QueryResponse(
            answer=answer,
            model_used=model_used,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        return {
            "total_records": collection.count(),
            "collection_name": COLLECTION_NAME,
            "models_available": [
                "openai" if openai_client else None,
                "groq" if groq_client else None
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )
