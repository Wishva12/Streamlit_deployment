# ai_chat_engine_faiss.py - FAISS implementation replacing ChromaDB
import os
import logging
import pickle
from typing import List, Dict, Optional
from datetime import datetime
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Changed from Chroma to FAISS
from dotenv import load_dotenv
import sqlite3
import uuid
import shutil
import glob
from googletrans import Translator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Translation Service (unchanged)
class TranslationService:
    """
    A service to handle language detection and translation using Googletrans.
    """
    def __init__(self):
        self.translator = Translator()

    def translate(self, text: str, dest_lang: str) -> (str, str):
        """
        Detects the source language and translates text to the destination language.

        Args:
            text (str): The text to translate.
            dest_lang (str): The destination language code (e.g., 'en', 'si').

        Returns:
            tuple: A tuple containing the translated text and the detected source language code.
        """
        try:
            detected = self.translator.detect(text)
            source_lang = detected.lang
            
            # No need to translate if already in the destination language
            if source_lang == dest_lang:
                return text, source_lang
                
            translated = self.translator.translate(text, dest=dest_lang, src=source_lang)
            return translated.text, source_lang
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text, 'en' # Fallback


class AIProductChatEngine:
    """
    Professional AI Chat Engine for Product Consultation
    Integrates RAG with Google Gemini and FAISS for contextual responses
    """
    
    def __init__(self, pdf_path: str = None, persist_directory: str = "faiss_db", data_folder: str = "Data"):
        """
        Initialize the AI Chat Engine
        
        Args:
            pdf_path: Path to product knowledge PDF
            persist_directory: Directory to persist FAISS vector store
            data_folder: Folder containing PDF files
        """
        self.pdf_path = pdf_path or "product_knowledge.pdf"
        self.persist_directory = persist_directory
        self.data_folder = data_folder
        self.vectorstore = None
        self.rag_chain = None
        self.chat_history = []
        
        # FAISS-specific paths
        self.faiss_index_path = os.path.join(persist_directory, "index.faiss")
        self.faiss_pkl_path = os.path.join(persist_directory, "index.pkl")
        
        # Initialize components
        self._setup_embeddings()
        self._setup_vectorstore()
        self._setup_llm()
        self._setup_rag_chain()
        
        logger.info("AI Chat Engine initialized successfully with FAISS")
    
    def _setup_embeddings(self):
        """Setup Google Generative AI embeddings"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            st.error("Failed to initialize AI embeddings. Please check your Google API key.")
            raise
    
    def _check_for_pdf_updates(self) -> bool:
        """
        Check if there are new or updated PDFs in the data folder
        
        Returns:
            bool: True if vector store needs to be rebuilt
        """
        try:
            # Check if data folder exists
            if not os.path.exists(self.data_folder):
                logger.info(f"Data folder '{self.data_folder}' does not exist")
                return False
            
            # Get all PDF files in data folder
            pdf_files = glob.glob(os.path.join(self.data_folder, "*.pdf"))
            
            if not pdf_files:
                logger.info("No PDF files found in data folder")
                return False
            
            # If FAISS index doesn't exist, we need to create it
            if not os.path.exists(self.faiss_index_path) or not os.path.exists(self.faiss_pkl_path):
                logger.info("FAISS index doesn't exist, will create new one")
                return True
            
            # Check modification times of PDFs vs FAISS index
            faiss_time = min(os.path.getmtime(self.faiss_index_path), os.path.getmtime(self.faiss_pkl_path))
            
            for pdf_file in pdf_files:
                pdf_time = os.path.getmtime(pdf_file)
                if pdf_time > faiss_time:
                    logger.info(f"PDF file '{pdf_file}' is newer than FAISS index")
                    return True
            
            logger.info("No PDF updates detected")
            return False
            
        except Exception as e:
            logger.error(f"Error checking for PDF updates: {e}")
            return True  # Rebuild on error to be safe
    
    def _setup_vectorstore(self):
        """Setup or load existing FAISS vector store"""
        try:
            # Check if we need to rebuild the vector store
            needs_rebuild = self._check_for_pdf_updates()
            
            if needs_rebuild:
                logger.info("Rebuilding FAISS vector store with updated PDFs...")
                # Remove existing vector store
                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
                    logger.info("Removed existing FAISS index")
                
                # Create new vector store from PDFs
                self._create_vectorstore_from_pdfs()
            else:
                logger.info("Loading existing FAISS vector store...")
                # Load existing FAISS index
                self.vectorstore = FAISS.load_local(
                    self.persist_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS vector store loaded successfully")
                
        except Exception as e:
            logger.error(f"FAISS vector store setup failed: {e}")
            # Fallback: create empty vectorstore
            self._create_fallback_vectorstore()
    
    def _create_vectorstore_from_pdfs(self):
        """Create FAISS vector store from all PDF files in the data folder"""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Get all PDF files from data folder
            pdf_files = glob.glob(os.path.join(self.data_folder, "*.pdf"))
            
            if not pdf_files:
                logger.warning(f"No PDF files found in '{self.data_folder}' folder")
                self._create_fallback_vectorstore()
                return
            
            all_documents = []
            
            # Load all PDF files
            for pdf_file in pdf_files:
                try:
                    logger.info(f"Processing PDF: {pdf_file}")
                    loader = PyPDFLoader(pdf_file)
                    documents = loader.load()
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} pages from {os.path.basename(pdf_file)}")
                except Exception as e:
                    logger.error(f"Failed to load PDF {pdf_file}: {e}")
                    continue
            
            if not all_documents:
                logger.warning("No documents were successfully loaded")
                self._create_fallback_vectorstore()
                return
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            docs = text_splitter.split_documents(all_documents)
            
            # Create FAISS vector store
            self.vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            # Save FAISS index to disk
            self.vectorstore.save_local(self.persist_directory)
            
            logger.info(f"FAISS vector store created with {len(docs)} document chunks from {len(pdf_files)} PDF files")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS vector store from PDFs: {e}")
            self._create_fallback_vectorstore()
    
    def _create_fallback_vectorstore(self):
        """Create fallback FAISS vector store with product knowledge"""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Fallback product knowledge
            fallback_docs = [
                "Mobitel Premium Package: Rs.2999/month, 100GB Data, Unlimited Calls, Free Netflix, Free Spotify, 5G Ready",
                "Mobitel Family Package: Rs.4999/month, 300GB Shared Data, 5 SIM Cards, Disney+ & Netflix, Parental Controls",
                "Mobitel Business Package: Rs.7999/month, Unlimited Data, Priority Network, 24/7 Support, Cloud Storage",
                "All packages include island-wide coverage, no contract commitment, 30-day money-back guarantee",
                "Mobitel offers the best network coverage in Sri Lanka with 95% 4G coverage nationwide",
                "Customer support available 24/7 via phone (123), email, and live chat",
                "All packages support 5G technology where available in Colombo, Kandy, and major cities",
                "International roaming available in 200+ countries with competitive rates",
                "Family package allows sharing data across 5 SIM cards with parental controls",
                "Business package includes priority customer support and dedicated account manager"
            ]
            
            from langchain.schema import Document
            docs = [Document(page_content=content) for content in fallback_docs]
            
            # Create FAISS vector store from fallback docs
            self.vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            # Save FAISS index to disk
            self.vectorstore.save_local(self.persist_directory)
            
            logger.info("Fallback FAISS vector store created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create fallback FAISS vector store: {e}")
            raise
    
    def _setup_llm(self):
        """Setup Google Gemini LLM"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                max_tokens=200,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            st.error("Failed to initialize AI model. Please check your Google API key.")
            raise
    
    def _setup_rag_chain(self):
        """Setup RAG chain for question answering"""
        try:
            # Setup retriever with optimized parameters for FAISS
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": 5,  # Number of documents to retrieve
                    "fetch_k": 10,  # Number of documents to fetch before MMR
                    "lambda_mult": 0.5,  # Diversity parameter for MMR
                }
            )
            
            # Enhanced system prompt for better responses
            system_prompt = (
                "You are ALEX, a friendly and knowledgeable AI assistant for Mobitel product consultation. "
                "Use the provided context to answer questions about mobile packages, features, and services. "
                "Be conversational, helpful, and persuasive while remaining honest. "
                "If you don't have specific information, offer to help find it or suggest alternatives. "
                "Keep responses concise but informative, maximum 4 sentences. "
                "Always maintain a positive, professional tone.\n\n"
                "Context: {context}"
            )
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Create chains
            document_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, document_chain)
            
            logger.info("RAG chain setup completed with FAISS")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG chain: {e}")
            raise
    
    def force_reload_vectorstore(self):
        """
        Force reload of FAISS vector store from PDFs
        Call this method when you want to manually refresh the knowledge base
        """
        try:
            logger.info("Force reloading FAISS vector store...")
            
            # Remove existing vector store
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info("Removed existing FAISS vector store")
            
            # Recreate vector store
            self._create_vectorstore_from_pdfs()
            
            # Recreate RAG chain
            self._setup_rag_chain()
            
            logger.info("FAISS vector store force reload completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to force reload FAISS vector store: {e}")
            return False
    
    def add_documents_to_vectorstore(self, documents: List[str]):
        """
        Add new documents to existing FAISS vector store
        
        Args:
            documents: List of text documents to add
        """
        try:
            from langchain.schema import Document
            docs = [Document(page_content=content) for content in documents]
            
            # Add documents to existing vector store
            self.vectorstore.add_documents(docs)
            
            # Save updated index
            self.vectorstore.save_local(self.persist_directory)
            
            logger.info(f"Added {len(documents)} documents to FAISS vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS vector store: {e}")
            return False
    
    def search_similar_documents(self, query: str, k: int = 5):
        """
        Search for similar documents in the vector store
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Perform similarity search with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    'content': doc.page_content,
                    'score': float(score),
                    'metadata': doc.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def chat(self, user_input: str, customer_name: str = "Customer") -> Dict[str, str]:
        """
        Process user input and return AI response
        
        Args:
            user_input: User's message
            customer_name: Customer's name for personalization
            
        Returns:
            Dictionary with response, context, and metadata
        """
        try:
            # Enhance input with customer context
            enhanced_input = f"Customer {customer_name} asks: {user_input}"
            
            # Get AI response
            response = self.rag_chain.invoke({"input": enhanced_input})
            
            # Extract and process response
            ai_response = response.get('answer', 'I apologize, but I encountered an issue processing your request.')
            context_docs = response.get('context', [])
            
            # Personalize response
            if customer_name != "Customer" and customer_name not in ai_response:
                ai_response = f"Hi {customer_name}! {ai_response}"
            
            # Log conversation
            self._log_conversation(user_input, ai_response, customer_name)
            
            return {
                'response': ai_response,
                'context_docs': len(context_docs),
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return {
                'response': "I'm sorry, I'm experiencing technical difficulties. Please try again in a moment.",
                'context_docs': 0,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'status': 'error'
            }
    
    def _log_conversation(self, user_input: str, ai_response: str, customer_name: str):
        """Log conversation to history and database"""
        try:
            # Add to session history
            conversation_entry = {
                'timestamp': datetime.now(),
                'customer': customer_name,
                'user_input': user_input,
                'ai_response': ai_response
            }
            self.chat_history.append(conversation_entry)
            
            # Log to database
            self._save_to_database(conversation_entry)
            
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
    
    def _save_to_database(self, entry: Dict):
        """Save conversation to SQLite database"""
        try:
            conn = sqlite3.connect('product_promotion.db')
            
            # Create table if not exists
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ai_chat_logs (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    customer_name TEXT,
                    user_input TEXT,
                    ai_response TEXT,
                    session_id TEXT
                )
            ''')
            
            # Insert conversation
            conn.execute('''
                INSERT INTO ai_chat_logs VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                entry['timestamp'],
                entry['customer'],
                entry['user_input'],
                entry['ai_response'],
                st.session_state.get('chat_session_id', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database save error: {e}")
    
    def get_vectorstore_info(self):
        """Get information about the FAISS vector store"""
        try:
            if self.vectorstore:
                # Get the number of documents in the vector store
                index_size = self.vectorstore.index.ntotal
                return {
                    'total_documents': index_size,
                    'index_type': 'FAISS',
                    'embedding_dimension': self.vectorstore.index.d,
                    'index_path': self.persist_directory
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get vectorstore info: {e}")
            return None
    
    def get_conversation_summary(self) -> Dict[str, int]:
        """Get conversation statistics"""
        return {
            'total_messages': len(self.chat_history),
            'session_duration': len(self.chat_history) * 30,  # Estimated seconds
            'last_activity': self.chat_history[-1]['timestamp'] if self.chat_history else None
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []
        logger.info("Conversation history cleared")


# Streamlit Chat Interface Component (unchanged except for some FAISS-specific features)
class StreamlitChatInterface:
    """
    Streamlit-specific chat interface component
    """
    
    def __init__(self, chat_engine: AIProductChatEngine):
        self.chat_engine = chat_engine
        self.translator = TranslationService()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """
        Initialize Streamlit session state variables and add an initial message
        """
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
            
        if 'chat_session_id' not in st.session_state:
            st.session_state.chat_session_id = str(uuid.uuid4())

        # Check if it's the first time the chat is loaded in this session
        if not st.session_state.chat_messages:
            # Add a welcome message from the assistant
            initial_message = """
            Hello! I am ALEX, from Mobitel. We are introducing new packages as you recognizing as a valuable customer.

Here are some of our popular packages:
* **Mobitel Premium Package:** Our flagship plan with 100GB Data, unlimited calls, and free streaming subscriptions.
* **Mobitel Family Package:** The perfect solution for the whole family with 300GB of shared data and 5 SIM cards.
* **Mobitel Business Package:** Unlimited data and priority network access for all your business needs.

Wanna try out these?
"""
            # Translate the initial message to Sinhala for a better user experience
            sinhala_initial_message, _ = self.translator.translate(initial_message, 'si')
            
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": sinhala_initial_message,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "context_docs": 3
            })
            
    def render_chat_interface(self, customer_name: str = "Customer"):
        """
        Render the complete chat interface
        
        Args:
            customer_name: Customer's name for personalization
        """
        # Chat header
        st.markdown("### 🤖 ALEX AI - Product Consultant (FAISS Powered)")
        st.markdown(f"*Chatting with: {customer_name}*")
        
        # Show vectorstore info
        vectorstore_info = self.chat_engine.get_vectorstore_info()
        if vectorstore_info:
            st.caption(f"📚 Knowledge Base: {vectorstore_info['total_documents']} documents indexed with FAISS")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "timestamp" in message:
                        st.caption(f"⏱️ {message['timestamp']}")
        
        if prompt := st.chat_input("අපගේ මොබිටෙල් පැකේජ ගැන විමසන්න... (Ask about our packages...)"):
            self._handle_user_input(prompt, customer_name, chat_container)
    
    def _handle_user_input(self, user_input: str, customer_name: str, chat_container):
        """Handle user input and generate AI response"""
        # Add original user message to chat history
        st.session_state.chat_messages.append({
            "role": "user", 
            "content": user_input
        })
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Get AI response
        with st.spinner("ALEX හිතනවා... (ALEX is thinking...)"):
            # Translation Logic
            # 1. Translate user input to English
            english_input, source_lang = self.translator.translate(user_input, 'en')
            logger.info(f"Detected language: {source_lang}. Translated query to: '{english_input}'")

            # 2. Get response from the English-based chat engine
            response_data = self.chat_engine.chat(english_input, customer_name)
            english_response = response_data['response']
            
            # 3. Translate the English response back to the user's original language (if not English)
            if source_lang != 'en':
                final_response, _ = self.translator.translate(english_response, source_lang)
            else:
                final_response = english_response
            
            # Update the response content with the final translated version
            response_data['response'] = final_response

        # Add AI response to chat history
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response_data['response'],
            "timestamp": response_data['timestamp'],
            "context_docs": response_data['context_docs']
        })
        
        # Display AI response
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(response_data['response'])
                st.caption(f"⏱️ {response_data['timestamp']} | 📚 {response_data['context_docs']} sources")
        
        # Auto-scroll to bottom
        st.rerun()
    
    def render_chat_sidebar(self):
        """Render chat-related sidebar components"""
        with st.sidebar:
            st.markdown("### 💬 Chat Controls")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_messages = []
                self.chat_engine.clear_history()
                st.success("Chat history cleared!")
                st.rerun()
            
            # Force reload vector store button
            if st.button("🔄 Reload Knowledge Base"):
                with st.spinner("Reloading FAISS knowledge base..."):
                    success = self.chat_engine.force_reload_vectorstore()
                if success:
                    st.success("FAISS knowledge base reloaded successfully!")
                else:
                    st.error("Failed to reload FAISS knowledge base!")
                st.rerun()
            
            # FAISS-specific features
            st.markdown("### 🔍 FAISS Search")
            search_query = st.text_input("Search knowledge base:")
            if search_query and st.button("Search"):
                results = self.chat_engine.search_similar_documents(search_query)
                for i, result in enumerate(results[:3]):
                    st.markdown(f"**Result {i+1}** (Score: {result['score']:.3f})")
                    st.markdown(result['content'][:200] + "...")
                    st.markdown("---")
            
            # Vectorstore info
            vectorstore_info = self.chat_engine.get_vectorstore_info()
            if vectorstore_info:
                st.markdown("### 📊 Vector Store Info")
                st.metric("Total Documents", vectorstore_info['total_documents'])
                st.metric("Embedding Dimension", vectorstore_info['embedding_dimension'])
                st.caption(f"Index Type: {vectorstore_info['index_type']}")
            
            # Chat statistics
            if st.session_state.chat_messages:
                stats = self.chat_engine.get_conversation_summary()
                st.markdown("### 📊 Chat Stats")
                st.metric("Messages", len(st.session_state.chat_messages))
                st.metric("Session Duration", f"{len(st.session_state.chat_messages) * 30}s")
            
            # Export chat option
            if st.session_state.chat_messages and st.button("📥 Export Chat"):
                chat_export = self._export_chat_history()
                st.download_button(
                    label="💾 Download Chat Log",
                    data=chat_export,
                    file_name=f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    def _export_chat_history(self) -> str:
        """Export chat history as text"""
        export_text = f"Chat Session Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += "=" * 50 + "\n\n"
        
        for message in st.session_state.chat_messages:
            role = "Customer" if message["role"] == "user" else "ALEX AI"
            export_text += f"{role}: {message['content']}\n"
            if "timestamp" in message:
                export_text += f"Time: {message['timestamp']}\n"
            export_text += "\n"
        
        return export_text

# Main function for testing
def main():
    """Main function for standalone testing"""
    st.set_page_config(
        page_title="AI Chat Interface - FAISS",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize chat engine
    if 'chat_engine' not in st.session_state:
        with st.spinner("Initializing AI Chat Engine with FAISS..."):
            st.session_state.chat_engine = AIProductChatEngine()
    
    # Initialize chat interface
    chat_interface = StreamlitChatInterface(st.session_state.chat_engine)
    
    # Render interface
    st.title("🤖 ALEX with you (FAISS Edition)")
    
    # Customer name input
    customer_name = st.text_input("Customer Name", value="John Doe")
    
    # Render chat
    chat_interface.render_chat_interface(customer_name)
    chat_interface.render_chat_sidebar()

if __name__ == "__main__":
    main()