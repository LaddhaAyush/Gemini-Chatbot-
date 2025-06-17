import os
from typing import TypedDict,Annotated,Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field

# Load your API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define the output structure
# class Review(TypedDict):
#     summary: Annotated[str, "A concise summary of given review." ]
#     sentiment: Annotated[Literal["positive","negative"], "The sentiment of the review, either positive, negative, or neutral."]

class Review(BaseModel):
    summary: str = Field(description="A concise summary of given review.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the review, either positive, negative, or neutral."  )
    key_themes: list[str] = Field(default_factory=list, description="Key themes or topics discussed in the review.")

# Create the ChatOpenAI model (make sure to use gpt-4o or gpt-4-turbo)
model = ChatGroq(model="qwen-qwq-32b")

# Wrap it with structured output
structured_model = model.with_structured_output(Review)

# Input text to analyze
input_text = """
The evolution of transformers has revolutionized the field of artificial intelligence, marking a remarkably positive shift from traditional sequential models like RNNs and LSTMs. Introduced by Vaswani et al. in 2017, the "Attention is All You Need" paper laid the foundation for a model that could understand long-range dependencies with parallel processing — a breakthrough that drastically reduced training time while improving performance. Transformers became the backbone of state-of-the-art models such as BERT, GPT, and T5, driving massive improvements in NLP, vision, and multi-modal tasks. Their scalability, transfer learning capability, and adaptability across domains have empowered a new era of intelligent systems, making them a cornerstone of modern AI innovation.
"""

# Run the model and get structured response
result = structured_model.invoke(input_text)

# Output the result
# print(result)
print(result.summary)
print(result.sentiment)
print(result.key_themes)
