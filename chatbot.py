# import streamlit as st
# from langchain.vectorstores import Qdrant
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain.chains import ConversationalRetrievalChain
# from qdrant_client import QdrantClient
# import os
# from dotenv import load_dotenv
# import tempfile

# # Load environment variables
# load_dotenv()

# # Initialize session state for chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Initialize session state for the vector store
# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = None

# # Configure embeddings
# model_name = "BAAI/bge-large-en"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

# # Initialize Qdrant client
# url = "http://localhost:6333"
# client = QdrantClient(url=url, prefer_grpc=False)

# def process_pdf(uploaded_file):
#     # Save uploaded file temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_path = tmp_file.name

#     # Load and process the PDF
#     loader = PyPDFLoader(tmp_path)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     # Create vector store
#     vector_store = Qdrant.from_documents(
#         texts,
#         embeddings,
#         url=url,
#         prefer_grpc=False,
#         collection_name="pdf_store"
#     )

#     # Clean up temporary file
#     os.unlink(tmp_path)
#     return vector_store

# def get_conversation_chain(vector_store):
#     llm = ChatGroq(
#         temperature=0.1,
#         groq_api_key=os.getenv('GROQ_API_KEY'),
#         model_name="meta-llama/llama-4-scout-17b-16e-instruct"
#     )
    
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
#         return_source_documents=True,
#         verbose=True
#     )
    
#     return conversation_chain

# # Streamlit UI
# st.title("📚 PDF Chat with Groq")
# st.write("Upload a PDF and ask questions about its content!")

# # File upload
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# if uploaded_file is not None and st.session_state.vector_store is None:
#     with st.spinner("Processing PDF..."):
#         st.session_state.vector_store = process_pdf(uploaded_file)
#     st.success("PDF processed successfully!")

# # Chat interface
# if st.session_state.vector_store is not None:
#     # Initialize conversation chain
#     conversation_chain = get_conversation_chain(st.session_state.vector_store)
    
#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Chat input
#     if prompt := st.chat_input("Ask a question about your PDF"):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 # Get chat history for context
#                 chat_history = [(m["content"], "") for m in st.session_state.messages[:-1] if m["role"] == "user"]
                
#                 # Get response from conversation chain
#                 response = conversation_chain({"question": prompt, "chat_history": chat_history})
                
#                 # Display response
#                 st.markdown(response["answer"])
                
#                 # Add assistant message to chat history
#                 st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# # Add a clear button to reset the chat
# if st.sidebar.button("Clear Chat"):
#     st.session_state.messages = []
#     st.session_state.vector_store = None
#     st.rerun()

import streamlit as st
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Chat",
    page_icon="📚",
    layout="centered"
)

# Simple custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #f8f9ff;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Configure embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Qdrant client
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

def process_pdf(uploaded_file):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load and process the PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create vector store
    vector_store = Qdrant.from_documents(
        texts,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name="pdf_store"
    )

    # Clean up temporary file
    os.unlink(tmp_path)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatGroq(
        temperature=0.4,
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

# Main UI
st.markdown('<h1 class="main-title">📚 PDF Chat Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Upload a PDF and ask questions about its content</p>', unsafe_allow_html=True)

# File upload section
if st.session_state.vector_store is None:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("### 📎 Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            st.session_state.vector_store = process_pdf(uploaded_file)
        
        st.markdown('<div class="success-box">✅ PDF processed successfully! You can now ask questions.</div>', unsafe_allow_html=True)
        st.rerun()

else:
    # Show current status
    st.markdown('<div class="success-box">📄 PDF is ready for questions</div>', unsafe_allow_html=True)
    
    # Clear button
    if st.button("🗑️ Upload New PDF"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.rerun()

# Chat interface
if st.session_state.vector_store is not None:
    st.markdown("### 💬 Ask Questions")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Initialize conversation chain
                conversation_chain = get_conversation_chain(st.session_state.vector_store)
                
                # Get chat history for context
                chat_history = [(m["content"], "") for m in st.session_state.messages[:-1] if m["role"] == "user"]
                
                # Get response from conversation chain
                response = conversation_chain({"question": prompt, "chat_history": chat_history})
                
                # Display response
                st.markdown(response["answer"])
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #888; font-size: 0.8em;">Powered by Groq • LangChain • Qdrant</p>', unsafe_allow_html=True)