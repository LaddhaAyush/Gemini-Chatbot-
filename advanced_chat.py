import os
from typing import List, Optional, Dict
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import json
import re

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Gemini Chat API",
    description="A more feature-rich chat application using FastAPI and Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

# Initialize Gemini model with more parameters
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_output_tokens=2048,
)

# Store conversation chains
conversation_chains: Dict[str, ConversationChain] = {}

# Custom prompt template for better context awareness
template = """The following is a friendly conversation between a human and an AI. The AI is helpful, creative, clever, and very friendly.

Current conversation:
{history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

def get_or_create_conversation(conversation_id: str) -> ConversationChain:
    """Get existing conversation chain or create a new one"""
    if conversation_id not in conversation_chains:
        memory = ConversationBufferMemory()
        conversation_chains[conversation_id] = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )
    return conversation_chains[conversation_id]

# Pydantic models for request/response validation
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    context: Optional[List[Dict]] = None

def format_response(text: str) -> str:
    """
    Format the response text to make it more readable by:
    1. Removing markdown formatting
    2. Adding proper spacing
    3. Converting lists to readable format
    """
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Convert markdown lists to readable format
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    
    # Add proper spacing between sections
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Format numbered lists
    text = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Add line breaks for better readability
    text = text.replace('. ', '.\n')
    
    return text.strip()

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("advanced_index.html", {
        "request": request,
        "api_key": GOOGLE_API_KEY,
        "current_time": datetime.now().strftime('%H:%M')
    })

@app.post("/chat", response_model=ChatResponse)
async def chat(
    chat_data: ChatMessage,
    background_tasks: BackgroundTasks,
):
    try:
        # Validate input
        if not chat_data.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Generate or use conversation ID
        conversation_id = chat_data.conversation_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get or create conversation chain
        conversation = get_or_create_conversation(conversation_id)
        
        # Get response from conversation chain
        response = conversation.predict(input=chat_data.message)
        
        # Format the response
        formatted_response = format_response(response)
        
        # Get conversation history
        memory = conversation.memory
        history = memory.chat_memory.messages
        
        # Convert history to list of dicts for response
        context = []
        for msg in history:
            context.append({
                "type": "user" if msg.type == "human" else "assistant",
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            })

        return ChatResponse(
            response=formatted_response,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            context=context
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    if conversation_id not in conversation_chains:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = conversation_chains[conversation_id]
    memory = conversation.memory
    history = memory.chat_memory.messages
    
    context = []
    for msg in history:
        context.append({
            "type": "user" if msg.type == "human" else "assistant",
            "content": msg.content,
            "timestamp": datetime.now().isoformat()
        })
    
    return context

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 