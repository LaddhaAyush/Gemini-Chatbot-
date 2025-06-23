import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# --- Config ---
CHROMA_DIR = "chroma_store"
EMBED_MODEL = "intfloat/e5-large-v2"

# --- Fetch transcript from YouTube ---
def fetch_youtube_transcript(video_id):
    """
    Fetch transcript from YouTube video using video ID
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

# --- Main ---
if __name__ == "__main__":
    # 1. Ask user for YouTube video ID
    video_id = input("Enter the YouTube video ID: ").strip()
    
    print("Fetching transcript...")
    transcript_text = fetch_youtube_transcript(video_id)
    if not transcript_text:
        print("No transcript found. Exiting.")
        exit(1)
    
    print(f"Transcript fetched successfully. Length: {len(transcript_text)} characters")

    # 2. Wrap transcript in LangChain Document
    docs = [Document(page_content=transcript_text, metadata={"source": f"YouTube:{video_id}"})]

    # 3. Split into chunks
    print("Splitting transcript into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size for better context
        chunk_overlap=200,  # Increased overlap
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # 4. Create Embeddings
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    # 5. Store embeddings in Chroma vector store
    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=CHROMA_DIR
    )

    # 6. Create prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful and knowledgeable assistant. Based on the YouTube video transcript provided below, please answer the user's question accurately and comprehensively.

Instructions:
- Use only the information from the transcript to answer the question
- If the answer isn't available in the transcript, clearly state "I couldn't find that information in the video transcript"
- Provide specific details when available
- Be concise but thorough

Transcript Context:
{context}

Question: {question}

Answer:"""
    )

    # 7. Load Groq LLM with Gemma model
    print("Initializing Groq LLM...")
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="gemma2-9b-it",  # You can also use "gemma2-9b-it" or "mixtral-8x7b-32768"
        temperature=0.1,
        max_tokens=1024
    )

    # 8. Build RetrievalQA chain
    print("Building QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
        ),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("Setup complete! You can now ask questions about the video.")
    print("Type 'quit' or 'exit' to stop.\n")

    # 9. Interactive question-answer loop
    while True:
        query = input("Ask a question about the YouTube transcript: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not query:
            continue
            
        try:
            print("Searching for answer...")
            response = qa_chain.invoke({"query": query})

            print("\n" + "="*50)
            print("ANSWER:")
            print("="*50)
            print(response["result"])

            print("\n" + "="*30)
            print("RELEVANT SOURCES:")
            print("="*30)
            for i, doc in enumerate(response["source_documents"], 1):
                print(f"{i}. Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"   Content preview: {doc.page_content[:150]}...")
                print()

        except Exception as e:
            print(f"Error processing question: {e}")
        
        print("\n" + "-"*50 + "\n")


# IMPROVEMENTS REQUIRED:-
# Query rewriting using llmm 
# multi query generation
# MMR , Reranking , Contextual COmpression
# Context window optimization
# context awared 
# multimodal 
# agentic ai 
# guard railing

