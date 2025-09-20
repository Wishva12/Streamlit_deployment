from src.helper import load_pdf_file, text_split, download_embeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
extracted_data = load_pdf_file(data = "Data/")
text_chunks = text_split(extracted_data)
embeddings = download_embeddings()

vectorstore = Chroma.from_documents(documents = extracted_data, embedding = embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5, "score_threshold": 0.7}
)