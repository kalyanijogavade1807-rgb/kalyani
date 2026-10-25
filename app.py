import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# Page Configuration
st.set_page_config(
    page_title="Campus Rulebook QA Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .stChatBox {
        background-color: #F9FAFB;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 10px;
        margin: 5px 0px;
        font-size: 0.9rem;
        color: #374151;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

def load_document(uploaded_file):
    """Loads uploaded PDF or TXT file into LangChain documents."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
            documents = loader.load()
        else:
            raise ValueError("Unsupported file format. Please upload a PDF or TXT file.")
        return documents
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def process_documents(documents):
    """Splits documents and creates ChromaDB vector store with HuggingFace embeddings."""
    try:
        # Split text into 500-character chunks with 80-character overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.error("No text could be extracted from the uploaded document.")
            return None

        # Use lightweight local embeddings (no API key required for embeddings)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Store in ChromaDB (in-memory persistent client)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("## ⚙️ Administration")
    st.markdown("Upload your campus handbook or academic policy document.")
    
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Required for answering student questions via GPT model."
    )
    
    model_choice = st.selectbox(
        "Select LLM Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload Policy Handbook",
        type=["pdf", "txt"],
        help="Upload a PDF or TXT campus policy document."
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing policy document and building vector database..."):
                docs = load_document(uploaded_file)
                if docs:
                    vstore = process_documents(docs)
                    if vstore:
                        st.session_state.vectorstore = vstore
                        st.session_state.doc_processed = True
                        st.session_state.messages = []
                        st.success("Document successfully processed and indexed in ChromaDB!")

    st.markdown("---")
    st.markdown("### 📌 Guidelines")
    st.markdown("""
    - **Chunk Size:** 500 characters
    - **Overlap:** 80 characters
    - **Embeddings:** `all-MiniLM-L6-v2`
    - **Vector Store:** ChromaDB
    """)

# --- Main Application Interface ---
st.markdown('<p class="main-header">🎓 Campus Rulebook QA Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about campus policies, academic guidelines, and code of conduct—grounded strictly in official documentation.</p>', unsafe_allow_html=True)

# Check if vector store is ready
if not st.session_state.doc_processed or st.session_state.vectorstore is None:
    st.info("👈 Please upload an academic policy PDF or text handbook in the sidebar and click **Process Document** to begin.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Source Quotes & Citations"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Source Chunk {i+1}</b> (Page {source.metadata.get('page', 'N/A') if hasattr(source, 'metadata') else 'N/A'}):<br>
                            {source.page_content}
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input
    if prompt := st.chat_input("Ask a question about the campus rulebook..."):
        if not api_key_input:
            st.error("Please enter your OpenAI API Key in the sidebar to ask questions.")
        else:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Retrieve relevant chunks from ChromaDB
            with st.spinner("Searching policy rulebook..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    )
                    retrieved_docs = retriever.get_relevant_documents(prompt)
                except Exception as e:
                    st.error(f"Retrieval error: {str(e)}")
                    retrieved_docs = []

            # Construct context from retrieved documents
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            system_prompt = f"""You are the Campus Rulebook QA Assistant, an official AI helper for college students and staff.
Your task is to answer user questions strictly based on the provided policy document excerpts below. 
If the answer cannot be found in the provided context, state clearly: "I cannot find the answer to this question in the provided campus rulebook."
Do not make up, infer, or extrapolate information outside the provided text. Always remain objective and accurate.

Context Excerpts:
{context_text}
"""

            # Call OpenAI API
            try:
                client = OpenAI(api_key=api_key_input)
                
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                
                assistant_response = response.choices[0].message.content

            except Exception as e:
                assistant_response = f"An error occurred while communicating with the OpenAI API: {str(e)}"
                retrieved_docs = []

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
                if retrieved_docs:
                    with st.expander("📚 View Source Quotes & Citations"):
                        for i, doc in enumerate(retrieved_docs):
                            page_num = doc.metadata.get('page', 'N/A') + 1 if 'page' in doc.metadata else 'N/A'
                            st.markdown(f"""
                            <div class="source-box">
                                <b>Source Chunk {i+1}</b> (Page {page_num}):<br>
                                {doc.page_content}
                            </div>
                            """, unsafe_allow_html=True)

            # Save assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "sources": retrieved_docs
            })