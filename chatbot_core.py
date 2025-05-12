from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains. import ConversationalRetrievalChain
from langchain.schema import Document
import fitz
from langchain.chat_models import ChatOpenAI
import os
from langchain.chat_models.base import SimpleChatModel
from groq import Groq

# Layout-aware PDF loader using PyMuPDF
def load_pdf_layout_aware(pdf_path):
    documents = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))  # sort top to bottom, then left to right
        text = "\n".join([b[4] for b in blocks if b[4].strip()])
        documents.append(Document(page_content=text, metadata={"page": i + 1}))
    return documents

# Main function to build the ConversationalRetrievalChain
def build_qa_chain(
    pdf_path="/Users/braintip/AIBOOKCLUB/AGI.pdf",
    faiss_path="faiss_index"
):
    # Load layout-aware documents
    documents = load_pdf_layout_aware(pdf_path)[1:]  # Skip page 1
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load or build FAISS index
    if os.path.exists(faiss_path):
        db = FAISS.load_local(faiss_path, embeddings)
        print("✅ Loaded existing FAISS index.")
    else:
        print("⚙️ Creating new FAISS index...")
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(faiss_path)
        print("✅ FAISS index saved.")

    # Set up retriever
    retriever = db.as_retriever()

    # Initialize LLM and QA chain
    llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key="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",
    model_name="llama3-70b-8192",  # Or "mixtral-8x7b-32768"
    temperature=0.7
)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
