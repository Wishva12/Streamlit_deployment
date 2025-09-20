from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    for doc in documents:
        # Wrap into Document objects after splitting
        for chunk in text_splitter.split_text(doc.page_content):
            all_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return all_chunks

def download_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings
