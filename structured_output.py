import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
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

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_output_tokens=2048,
)

# Create the prompt template with structured response instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, creative, and friendly AI assistant. 
    When asked to provide structured information, format your response as a JSON object.
    For regular conversations, respond naturally.
    When providing structured data, ensure it's valid JSON and include it at the end of your response.
    Example structured response format:
    {{
        "summary": "Brief summary of the information",
        "key_points": ["point1", "point2"],
        "details": {{
            "field1": "value1",
            "field2": "value2"
        }}
    }}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Store conversation histories
conversation_histories: Dict[str, List[Dict[str, Any]]] = {}

def get_conversation_history(conversation_id: str) -> List[Dict[str, Any]]:
    if conversation_id not in conversation_histories:
        conversation_histories[conversation_id] = []
    return conversation_histories[conversation_id]

def add_to_history(conversation_id: str, role: str, content: str):
    history = get_conversation_history(conversation_id)
    history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

def format_messages(history: List[Dict[str, Any]]) -> List[Any]:
    messages = []
    for msg in history:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    structured_output: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    context: Optional[List[Dict]] = None
    structured_data: Optional[Dict] = None

def format_response(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('. ', '.\n')
    return text.strip()

def extract_structured_data(text: str) -> Optional[Dict]:
    try:
        # Look for JSON object in the text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
    except Exception as e:
        print(f"Error extracting structured data: {e}")
    return None

# Create the chain
chain = (
    RunnablePassthrough.assign(
        history=lambda x: format_messages(get_conversation_history(x["conversation_id"]))
    )
    | prompt
    | llm
)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("advanced_index.html", {
        "request": request,
        "api_key": GOOGLE_API_KEY,
        "current_time": datetime.now().strftime('%H:%M')
    })

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_data: ChatMessage):
    try:
        if not chat_data.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        conversation_id = chat_data.conversation_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add user message to history
        add_to_history(conversation_id, "human", chat_data.message)

        # Prepare input for the chain
        chain_input = {
            "input": chat_data.message + ("\nPlease provide a structured response in JSON format." if chat_data.structured_output else ""),
            "conversation_id": conversation_id
        }

        # Get response from the chain
        response = chain.invoke(chain_input)
        response_text = response.content if hasattr(response, "content") else str(response)
        formatted_response = format_response(response_text)

        # Extract structured data if present
        structured_data = extract_structured_data(response_text)

        # Add AI response to history
        add_to_history(conversation_id, "assistant", formatted_response)

        return ChatResponse(
            response=formatted_response,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            context=get_conversation_history(conversation_id),
            structured_data=structured_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    if conversation_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return get_conversation_history(conversation_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
