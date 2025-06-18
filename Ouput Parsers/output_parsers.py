from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint   
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Load the model
llm = HuggingFaceEndpoint(
    model="mistralai/Magistral-Small-2506",
    task="text-generation"
)
model = ChatHuggingFace(
    llm=llm,
    model_kwargs={"temperature": 0.4}
)

# Prompt templates using proper message structure
template1 = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("Write a detailed report on {topic}")
])

template2 = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("Write a 5 line summary on the following text:\n{text}")
])

# Invoke prompt 1
prompt1 = template1.invoke({"topic": "Docker"})
result = model.invoke(prompt1)
print(result.content)

# Invoke prompt 2 on the result of prompt 1
prompt2 = template2.invoke({"text": result.content})
print("\n\n\nSummary of the report:")
print(model.invoke(prompt2).content)
