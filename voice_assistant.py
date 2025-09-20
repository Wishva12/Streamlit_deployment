import pyttsx3
import speech_recognition as sr
import numpy as np
import webrtcvad
import collections
import sys
import signal
from os import getenv
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
from dotenv import load_dotenv
import requests
import json
import time
import pyaudio
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os
import shutil
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
try:
    from src.helper import load_pdf_file, text_split, download_embeddings
except ImportError:
    logger.error("Cannot import src.helper module. Please ensure it exists.")
    sys.exit(1)

# Validate environment variables
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY not found in environment variables")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Setup LLM and prompt
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=75)
system_prompt = """You are Alex, a helpful Mobitel AI Assistant for customer service and product promotion. 
Answer questions based on the provided documents and customer context. Keep responses conversational and helpful.
If promoting products, ask relevant questions about customer needs."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt), 
    ("human", "Context: {context}\n\nCustomer Profile: {customer_profile}\n\nQuestion: {input}")
])

# Global variables
embeddings = download_embeddings()
vectorstore = None
retriever = None
conversation_memory = []
is_bot_speaking = threading.Event()
assistant_running = threading.Event()
current_customer_profile = {}

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
CHUNK_SIZE = 1024
VAD_AGGRESSIVENESS = 3

class VoiceAssistant:
    def __init__(self):
        self.engine = None
        self.vad = None
        self.audio = None
        self.stream = None
        self.recognizer = sr.Recognizer()
        self.setup_components()
    
    def setup_components(self):
        """Initialize TTS engine and VAD"""
        try:
            # Setup TTS
            self.engine = pyttsx3.init()
            if self.engine:
                self.engine.setProperty('rate', 160)
                self.engine.setProperty('volume', 0.9)
                voices = self.engine.getProperty('voices')
                if voices:
                    # Try to set a female voice if available
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
            
            # Setup VAD
            self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
            
            # Setup PyAudio
            self.audio = pyaudio.PyAudio()
            
            logger.info("Voice components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing voice components: {e}")
            raise
    
    def speak_message(self, text):
        """Convert text to speech with thread safety"""
        if not self.engine or not text.strip():
            return
            
        try:
            is_bot_speaking.set()
            log_message(f"Assistant: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            log_message(f"TTS Error: {e}")
        finally:
            is_bot_speaking.clear()
    
    def listen_for_speech(self, timeout=5):
        """Listen for speech input using speech_recognition"""
        if is_bot_speaking.is_set():
            return ""
            
        try:
            with sr.Microphone() as source:
                log_message("Listening...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
            log_message("Processing speech...")
            # Recognize speech using Google
            text = self.recognizer.recognize_google(audio).lower()
            log_message(f"User said: {text}")
            return text
            
        except sr.WaitTimeoutError:
            log_message("No speech detected within timeout")
            return ""
        except sr.UnknownValueError:
            log_message("Could not understand audio")
            return ""
        except sr.RequestError as e:
            log_message(f"Error with speech recognition service: {e}")
            return ""
        except Exception as e:
            log_message(f"Unexpected error in speech recognition: {e}")
            return ""
    
    def cleanup(self):
        """Cleanup audio resources"""
        try:
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def get_greeting():
    """Get time-appropriate greeting"""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"

def gemini_response(input_text, customer_profile=None):
    """Generate response using Gemini with RAG"""
    try:
        if not retriever:
            return "I'm sorry, my knowledge base is not available right now. Please try again later."
        
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        
        response = rag_chain.invoke({
            "input": input_text,
            "customer_profile": str(customer_profile) if customer_profile else "No profile available"
        })
        
        return response['answer']
        
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        return "I'm sorry, I encountered an error processing your request. Please try again."

def update_customer_profile(name, phone, segment, current_plan):
    """Update global customer profile"""
    global current_customer_profile
    current_customer_profile = {
        "name": name,
        "phone_number": phone,
        "segment": segment,
        "current_plan": current_plan
    }
    log_message(f"Customer Profile Updated: {current_customer_profile}")

def handle_conversation_flow(user_input, assistant):
    """Handle the main conversation logic"""
    user_input = user_input.lower().strip()
    
    # Exit conditions
    if any(word in user_input for word in ["exit", "quit", "goodbye", "bye", "stop", "thank you", "see you", "not interested"]):
        assistant.speak_message("Thank you for your time! Have a great day!")
        return False
    
    # Busy/not interested responses
    if any(phrase in user_input for phrase in ["busy", "no time", "not now"]):
        assistant.speak_message("No problem! I'll reach out another time. Have a good day!")
        return False
    
    # Positive responses to continue conversation
    if any(word in user_input for word in ["yes", "go ahead", "sure", "okay"]):
        assistant.speak_message(gemini_response("Promote product and ask questions also from the customer", current_customer_profile))
        return True
    
    # Negative response
    if "no" in user_input:
        assistant.speak_message("Okay, thank you for your time. Have a great day!")
        return False
    
    # Greeting responses
    if any(word in user_input for word in ["hello", "hi", "hey"]):
        assistant.speak_message("Hello! How can I help you with Mobitel services today?")
        return True
    
    # Generate AI response for other inputs
    if user_input:
        ai_response = gemini_response(user_input, current_customer_profile)
        assistant.speak_message(ai_response)
    else:
        assistant.speak_message("I didn't quite catch that. Do you want to continue our conversation?")
    
    return True

def run_conversation():
    """Main conversation loop"""
    assistant = VoiceAssistant()
    assistant_running.set()
    
    try:
        # Update status to show active
        update_status(True)
        
        # Initial greeting
        greeting = get_greeting()
        assistant.speak_message(f"{greeting}, this is Alex from Mobitel.")
        assistant.speak_message("Could you please spare a moment to talk with me about our services?")
        
        # Main conversation loop
        while assistant_running.is_set():
            if not is_bot_speaking.is_set():
                user_input = assistant.listen_for_speech(timeout=10)
                
                if user_input:
                    if not handle_conversation_flow(user_input, assistant):
                        break
                elif assistant_running.is_set():  # Only prompt if still running
                    assistant.speak_message("Are you still there? I'd be happy to continue our conversation.")
            
            time.sleep(0.1)  # Small delay to prevent busy waiting
            
    except KeyboardInterrupt:
        log_message("Assistant stopped by user")
    except Exception as e:
        logger.error(f"Error in assistant loop: {e}")
        log_message(f"Assistant error: {e}")
    finally:
        assistant.cleanup()
        assistant_running.clear()
        update_status(False)

def initialize_rag_system():
    """Initialize the RAG (Retrieval Augmented Generation) system"""
    global vectorstore, retriever
    
    try:
        log_message("Initializing RAG system...")
        
        # Check if Data directory exists and has PDF files
        if os.path.exists("Data/") and any(f.endswith('.pdf') for f in os.listdir("Data/")):
            # Process existing PDFs
            extracted_data = load_pdf_file("Data/")
            text_chunks = text_split(extracted_data)
            log_message(f"Found existing PDFs, processed into {len(text_chunks)} text chunks")
            
            # Create vector store
            vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5, "score_threshold": 0.7}
            )
            
            log_message("RAG system initialized successfully with existing documents")
        else:
            log_message("No PDF documents found in Data/ directory")
            log_message("Assistant will work with general knowledge only")
            
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        log_message(f"RAG initialization error: {e}")
        log_message("Assistant will work with general knowledge only")

# --- GUI Components ---
root = None
text_area = None
name_entry = None
phone_entry = None
segment_var = None
plan_entry = None
start_button = None
stop_button = None
status_indicator = None
blinking = None

def log_message(message):
    """Log messages to GUI and console"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    
    if text_area:
        text_area.insert(tk.END, formatted_message + "\n")
        text_area.see(tk.END)
    
    print(formatted_message)

def update_status(active):
    """Update the status indicator"""
    global blinking
    if active:
        blinking.set(True)
        log_message("Voice assistant is now active")
    else:
        blinking.set(False)
        if status_indicator:
            status_indicator.config(fg="gray")
        log_message("Voice assistant is now inactive")

def blink_status():
    """Animate the status indicator"""
    if blinking and blinking.get() and status_indicator:
        current_color = status_indicator.cget("fg")
        status_indicator.config(fg="green" if current_color == "gray" else "gray")
    root.after(500, blink_status)

def start_assistant_in_thread():
    """Start the voice assistant in a separate thread"""
    global current_customer_profile
    
    try:
        # Get customer profile from GUI
        name = name_entry.get().strip()
        phone = phone_entry.get().strip()
        segment = segment_var.get()
        current_plan = plan_entry.get().strip()
        
        # Validate inputs
        if not name:
            messagebox.showwarning("Input Required", "Please enter a customer name")
            return
        
        update_customer_profile(name, phone, segment, current_plan)
        
        # Start assistant thread
        assistant_thread = threading.Thread(target=run_conversation, daemon=True)
        assistant_thread.start()
        
        log_message("Voice assistant started...")
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        
    except Exception as e:
        logger.error(f"Error starting assistant: {e}")
        log_message(f"Failed to start assistant: {e}")
        messagebox.showerror("Error", f"Failed to start assistant: {e}")

def stop_assistant():
    """Stop the voice assistant"""
    try:
        assistant_running.clear()
        log_message("Stopping voice assistant...")
        
        # Re-enable start button
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        
        # Wait a moment for cleanup
        time.sleep(1)
        log_message("Voice assistant stopped")
        
    except Exception as e:
        logger.error(f"Error stopping assistant: {e}")
        log_message(f"Error stopping assistant: {e}")

def on_closing():
    """Handle window closing"""
    try:
        assistant_running.clear()
        time.sleep(0.5)  # Give time for cleanup
        root.destroy()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def setup_gui():
    """Setup the main GUI matching the interface design"""
    global root, text_area, name_entry, phone_entry, segment_var, plan_entry
    global start_button, stop_button, status_indicator, blinking
    
    root = tk.Tk()
    root.title("📞 Mobitel Voice AI Assistant")
    root.geometry("900x700")
    root.configure(bg="#f0f0f0")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Initialize blinking variable
    blinking = tk.BooleanVar(value=False)
    
    # Main container
    main_frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=20)
    main_frame.pack(fill="both", expand=True)
    
    # Customer Profile Section
    profile_frame = tk.LabelFrame(
        main_frame, 
        text="Customer Profile", 
        font=("Arial", 12, "bold"),
        bg="#f0f0f0",
        padx=20, 
        pady=15
    )
    profile_frame.pack(fill="x", pady=(0, 20))
    
    # Customer Name
    name_frame = tk.Frame(profile_frame, bg="#f0f0f0")
    name_frame.pack(fill="x", pady=5)
    tk.Label(name_frame, text="Customer Name:", font=("Arial", 10), bg="#f0f0f0", width=15, anchor="w").pack(side="left")
    name_entry = tk.Entry(name_frame, font=("Arial", 10), width=50)
    name_entry.pack(side="left", fill="x", expand=True, padx=(10, 0))
    name_entry.insert(0, "John Doe")
    
    # Phone Number
    phone_frame = tk.Frame(profile_frame, bg="#f0f0f0")
    phone_frame.pack(fill="x", pady=5)
    tk.Label(phone_frame, text="Phone Number:", font=("Arial", 10), bg="#f0f0f0", width=15, anchor="w").pack(side="left")
    phone_entry = tk.Entry(phone_frame, font=("Arial", 10), width=50)
    phone_entry.pack(side="left", fill="x", expand=True, padx=(10, 0))
    phone_entry.insert(0, "+94712345678")
    
    # Customer Segment
    segment_frame = tk.Frame(profile_frame, bg="#f0f0f0")
    segment_frame.pack(fill="x", pady=5)
    tk.Label(segment_frame, text="Customer Segment:", font=("Arial", 10), bg="#f0f0f0", width=15, anchor="w").pack(side="left")
    segment_var = tk.StringVar(root)
    segment_var.set("Consumer")
    segment_options = ["Consumer", "Business", "Prepaid", "Postpaid", "Enterprise"]
    segment_menu = tk.OptionMenu(segment_frame, segment_var, *segment_options)
    segment_menu.config(font=("Arial", 10), width=30)
    segment_menu.pack(side="left", padx=(10, 0))
    
    # Current Plan
    plan_frame = tk.Frame(profile_frame, bg="#f0f0f0")
    plan_frame.pack(fill="x", pady=5)
    tk.Label(plan_frame, text="Current Plan:", font=("Arial", 10), bg="#f0f0f0", width=15, anchor="w").pack(side="left")
    plan_entry = tk.Entry(plan_frame, font=("Arial", 10), width=50)
    plan_entry.pack(side="left", fill="x", expand=True, padx=(10, 0))
    plan_entry.insert(0, "Smart Package 500")
    
    # Control Buttons Section
    button_frame = tk.Frame(main_frame, bg="#f0f0f0")
    button_frame.pack(pady=20)
    
    # Start Voice Assistant Button
    start_button = tk.Button(
        button_frame,
        text="Start Voice Assistant",
        font=("Arial", 11, "bold"),
        bg="#4CAF50",
        fg="white",
        width=18,
        height=2,
        command=start_assistant_in_thread,
        relief="raised",
        bd=2
    )
    start_button.pack(side="left", padx=10)
    
    # Stop Assistant Button
    stop_button = tk.Button(
        button_frame,
        text="Stop Assistant",
        font=("Arial", 11, "bold"),
        bg="#f44336",
        fg="white",
        width=18,
        height=2,
        command=stop_assistant,
        state=tk.DISABLED,
        relief="raised",
        bd=2
    )
    stop_button.pack(side="left", padx=10)
    
    # Exit Application Button
    exit_button = tk.Button(
        button_frame,
        text="Exit Application",
        font=("Arial", 11, "bold"),
        bg="#FF9800",
        fg="white",
        width=18,
        height=2,
        command=on_closing,
        relief="raised",
        bd=2
    )
    exit_button.pack(side="left", padx=10)
    
    # Status Indicator
    status_frame = tk.Frame(main_frame, bg="#f0f0f0")
    status_frame.pack(pady=10)
    
    tk.Label(status_frame, text="Status:", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(side="left")
    status_indicator = tk.Label(status_frame, text="● INACTIVE", font=("Arial", 12, "bold"), fg="gray", bg="#f0f0f0")
    status_indicator.pack(side="left", padx=10)
    
    # Assistant Activity Log Section
    log_frame = tk.LabelFrame(
        main_frame,
        text="Assistant Activity Log",
        font=("Arial", 12, "bold"),
        bg="#f0f0f0",
        padx=15,
        pady=15
    )
    log_frame.pack(fill="both", expand=True, pady=(20, 0))
    
    text_area = scrolledtext.ScrolledText(
        log_frame,
        wrap=tk.WORD,
        width=100,
        height=20,
        font=("Consolas", 9),
        bg="#ffffff",
        fg="#333333"
    )
    text_area.pack(fill="both", expand=True)
    
    # Start status blinking animation
    blink_status()
    
    # Initial log messages
    log_message("Mobitel Voice AI Assistant initialized")
    
    # Initialize RAG system on startup
    initialize_rag_system()
    
    log_message("Fill in customer details and click 'Start Voice Assistant' to begin")
    
    root.mainloop()

if __name__ == "__main__":
    try:
        setup_gui()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
    finally:
        # Cleanup
        if 'assistant_running' in globals():
            assistant_running.clear()
        logger.info("Application shutdown complete")