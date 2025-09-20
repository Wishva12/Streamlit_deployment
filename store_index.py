from src.helper import load_pdf_file, text_split, download_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Step 1: Load and split PDF
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)

# Step 2: Get embeddings
embeddings = download_embeddings()

# Step 3: Create FAISS vectorstore
vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)

# Step 4: Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance
    search_kwargs={
        "k": 5,
        "fetch_k": 10,
        "lambda_mult": 0.5,
        "score_threshold": 0.7
    }
)
