from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os

# Set your Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") # Replace with your actual key

# Initialize FastAPI app
app = FastAPI()

# Input schema
class PromptInput(BaseModel):
    type: str  # 'zero' or 'few'
    text: str

# Load Groq model
llm = ChatGroq(model_name="gemma2-9b-it")

# Define prompt templates
zero_shot = PromptTemplate.from_template("Translate to French: {text}")

few_shot = PromptTemplate.from_template(
    """Translate the following:
Hello → Bonjour
Good night → Bonne nuit
{text} →"""
)

# POST endpoint
@app.post("/prompt")
def get_response(data: PromptInput):
    prompt = few_shot if data.type == "few" else zero_shot
    text = prompt.format(text=data.text)
    output = llm.predict(text)
    return {"output": output.strip()}
