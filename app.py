import hashlib
import io
import os
import re
import uuid
from typing import Any

import chromadb
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


APP_TITLE = "Campus Rulebook QA Assistant"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f6f8fc 0%, #ffffff 46%);
    }
    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f4fa;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid #e3eaf5;
        border-radius: 18px;
        background: linear-gradient(120deg, #ffffff, #eef5ff);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        color: #17345c;
        font-size: 2rem;
    }
    .hero p {
        margin: 0;
        color: #52647d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the local embedding model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_chroma_client() -> chromadb.ClientAPI:
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.EphemeralClient()
    return st.session_state.chroma_client


def read_handbook(uploaded_file: Any) -> list[tuple[int, str]]:
    """Return non-empty page/text pairs; plain-text handbooks are page 1."""
    content = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("This PDF is password-protected. Upload an unlocked copy.")
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((page_number, text))
            return pages
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read this PDF: {exc}") from exc

    try:
        return [(1, content.decode("utf-8-sig"))]
    except UnicodeDecodeError as exc:
        raise ValueError("Could not decode this text file as UTF-8.") from exc


def split_pages(pages: list[tuple[int, str]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page_number, page_text in pages:
        text = re.sub(r"\s+", " ", page_text).strip()
        for start in range(0, len(text), step):
            chunk = text[start : start + CHUNK_SIZE].strip()
            if chunk:
                chunks.append({"text": chunk, "page": page_number})

    return chunks


def index_handbook(uploaded_file: Any) -> tuple[str, int]:
    pages = read_handbook(uploaded_file)
    chunks = split_pages(pages)
    if not chunks:
        raise ValueError(
            "No readable text was found. If this is a scanned PDF, run OCR and upload it again."
        )

    model = load_embedding_model()
    client = get_chroma_client()
    collection_name = f"handbook_{uuid.uuid4().hex}"
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    try:
        for batch_start in range(0, len(chunks), 128):
            batch = chunks[batch_start : batch_start + 128]
            embeddings = model.encode(
                [item["text"] for item in batch],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            collection.add(
                ids=[f"chunk_{batch_start + offset}" for offset in range(len(batch))],
                documents=[item["text"] for item in batch],
                embeddings=embeddings.tolist(),
                metadatas=[
                    {
                        "page": item["page"],
                        "source": uploaded_file.name,
                    }
                    for item in batch
                ],
            )
    except Exception:
        client.delete_collection(name=collection_name)
        raise

    previous_collection = st.session_state.get("collection_name")
    st.session_state.collection_name = collection_name
    st.session_state.document_name = uploaded_file.name
    st.session_state.document_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()
    st.session_state.chunk_count = len(chunks)
    st.session_state.messages = []

    if previous_collection:
        try:
            client.delete_collection(name=previous_collection)
        except Exception as exc:
            st.warning(f"The previous index could not be cleared: {exc}")

    return uploaded_file.name, len(chunks)


def get_api_key() -> str:
    entered_key = st.session_state.get("openai_api_key_input", "").strip()
    if entered_key:
        return entered_key

    try:
        configured_key = st.secrets.get("OPENAI_API_KEY", "")
        if configured_key:
            return str(configured_key).strip()
    except (FileNotFoundError, AttributeError):
        pass

    return os.getenv("OPENAI_API_KEY", "").strip()


def answer_question(question: str, collection: Any, api_key: str) -> tuple[str, list[dict[str, Any]]]:
    model = load_embedding_model()
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    sources = [
        {
            "label": f"S{index + 1}",
            "text": document,
            "page": metadata.get("page"),
            "source": metadata.get("source", st.session_state.document_name),
        }
        for index, (document, metadata) in enumerate(zip(documents, metadatas))
    ]
    if not sources:
        raise ValueError("No passages were retrieved. Try reprocessing the handbook.")

    evidence = "\n\n".join(
        f"[{source['label']}] (page {source['page']}) {source['text']}"
        for source in sources
    )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer student questions about the campus handbook. "
                    "Use only the supplied evidence. Do not use outside knowledge, "
                    "make assumptions, or infer facts not stated in the evidence. "
                    "If the evidence does not explicitly answer the question, say: "
                    "\"I couldn't find that information in the uploaded handbook.\" "
                    "For every factual statement, cite supporting evidence using its "
                    "label, such as [S1]. Keep the answer clear and concise."
                ),
            },
            {
                "role": "user",
                "content": f"Handbook evidence:\n{evidence}\n\nQuestion: {question}",
            },
        ],
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The answer service returned an empty response.")
    return answer, sources


st.markdown(
    f"""
    <div class="hero">
      <h1>🎓 {APP_TITLE}</h1>
      <p>Ask questions about your campus handbook and get answers grounded in its text.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Set up your handbook")
    uploaded_file = st.file_uploader(
        "Upload an academic policy",
        type=["pdf", "txt"],
        help="PDF and UTF-8 text handbooks are supported.",
    )
    process_clicked = st.button(
        "Process handbook",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )
    st.divider()
    st.subheader("Answer service")
    st.caption("An OpenAI API key is required to generate answers.")
    st.text_input(
        "OpenAI API key",
        type="password",
        key="openai_api_key_input",
        placeholder="sk-...",
        help="Alternatively, set OPENAI_API_KEY in your environment or Streamlit secrets.",
    )

if process_clicked and uploaded_file is not None:
    with st.spinner("Reading, chunking, embedding, and indexing the handbook..."):
        try:
            filename, chunk_count = index_handbook(uploaded_file)
            st.sidebar.success(f"Indexed {filename} into {chunk_count} chunks.")
        except Exception as exc:
            st.error(f"Could not process the handbook: {exc}")

if st.session_state.get("collection_name"):
    st.caption(
        f"📘 **{st.session_state.document_name}** · "
        f"{st.session_state.chunk_count} searchable chunks"
    )
    if not get_api_key():
        st.info(
            "Add your OpenAI API key in the sidebar to ask questions. "
            "Your handbook is already indexed locally for this session."
        )

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 View retrieved source quotes"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**[{source['label']}] {source['source']} · "
                            f"page {source['page']}**"
                        )
                        st.markdown(f"> {source['text']}")
            if message.get("error"):
                st.error(message["error"])

    question = st.chat_input("Ask a question about the handbook...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            api_key = get_api_key()
            if not api_key:
                message_text = "Please add an OpenAI API key in the sidebar before asking a question."
                st.warning(message_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": message_text}
                )
            else:
                try:
                    collection = get_chroma_client().get_collection(
                        name=st.session_state.collection_name
                    )
                    with st.spinner("Searching the handbook and preparing an answer..."):
                        answer, sources = answer_question(question, collection, api_key)
                    st.markdown(answer)
                    with st.expander("📚 View retrieved source quotes"):
                        for source in sources:
                            st.markdown(
                                f"**[{source['label']}] {source['source']} · "
                                f"page {source['page']}**"
                            )
                            st.markdown(f"> {source['text']}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as exc:
                    error_text = f"Could not generate an answer: {exc}"
                    st.error(error_text)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "I couldn't complete that request.",
                            "error": error_text,
                        }
                    )
else:
    st.info("Upload a campus policy PDF or text handbook in the sidebar to get started.")
    st.markdown(
        """
        **How it works**
        1. Upload and index a PDF or text handbook.
        2. Ask a question in the chat.
        3. Review the cited source quotes alongside each answer.
        """
    )
