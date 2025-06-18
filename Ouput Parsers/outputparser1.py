from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint   
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage

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
    HumanMessagePromptTemplate.from_template("Write a 5 line bullet point summary on the following text:\n{text}")
])

parser= StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result=chain.invoke({
    "topic": "Artificial Intelligence"
})

print(result)
