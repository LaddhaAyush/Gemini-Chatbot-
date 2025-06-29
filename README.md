# Advanced Gemini Chat Application

A feature-rich chat application built with FastAPI and Google's Gemini AI model, featuring context-aware conversations and a modern user interface.

# Overview

This repository contains sample code demonstrating how to use Google's Gemini AI models with structured outputs, TypedDict, and Pydantic. It serves as a learning resource for those interested in implementing AI chatbots with structured data handling.

## Features

- 🤖 Powered by Google's Gemini AI model
- 💬 Context-aware conversations using LangChain
- 🎨 Modern and responsive user interface
- 📱 Mobile-friendly design
- 🔄 Real-time conversation updates
- 📝 Conversation history tracking
- 📤 Export chat functionality
- 🎯 Customizable prompt templates
- 🔍 Conversation context panel

# Examples Included
🤖 Basic Gemini model integration
📊 Structured output handling with TypedDict
🔍 JSON schema generation with Pydantic
💬 Chat-based conversations with Gemini
🏗️ Examples of prompt engineering for structured responses

## Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Project Structure

```
chatbot/
├── advanced_chat.py          # Main FastAPI application
├── requirements.txt          # Project dependencies
├── static/
│   └── advanced_style.css    # CSS styles
├── templates/
│   └── advanced_index.html   # HTML template
└── README.md                 # This file
```

## Usage

1. Start the server:
```bash
uvicorn advanced_chat:app --reload
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

## Features in Detail

### Context-Aware Conversations
- Uses LangChain's ConversationBufferMemory for maintaining conversation context
- Custom prompt template for better AI responses
- Automatic history tracking and management

### User Interface
- Clean and modern design
- Responsive layout for all devices
- Real-time message updates
- Context panel for viewing conversation history
- Export functionality for chat logs

### API Endpoints

- `GET /`: Main chat interface
- `POST /chat`: Send and receive messages
- `GET /conversations/{conversation_id}`: Retrieve conversation history
- `GET /health`: Health check endpoint

## Technical Implementation

### LangChain Integration
The application uses LangChain for conversation management:
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Custom prompt template
template = """The following is a friendly conversation between a human and an AI. 
The AI is helpful, creative, clever, and very friendly.

Current conversation:
{history}
Human: {input}
Assistant:"""

# Conversation chain setup
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    prompt=PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
)
```

### Memory Management
- Each conversation has its own memory instance
- Messages are automatically stored and formatted
- Context is maintained throughout the conversation
