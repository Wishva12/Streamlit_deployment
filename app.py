import tkinter as tk
from tkinter import filedialog, ttk
import pyttsx3
import speech_recognition as sr
import webrtcvad
import threading
import pyaudio
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os
import shutil
import tkinter.messagebox as messagebox
import requests
import json
import sqlite3
import uuid

from src.helper import load_pdf_file, text_split, download_embeddings

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# mSpace API Configuration
MSPACE_API_KEY = os.getenv("MSPACE_API_KEY")
MSPACE_SENDER_ID = os.getenv("MSPACE_SENDER_ID")
MSPACE_SMS_URL = "https://mspace.lk/api/sms/send"
MSPACE_USSD_URL = "https://mspace.lk/api/ussd"

class CustomerDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('customer_interactions.db', check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                phone_number TEXT,
                customer_name TEXT,
                interaction_type TEXT,
                conversation_summary TEXT,
                sentiment_score REAL,
                follow_up_required BOOLEAN,
                created_at TIMESTAMP,
                status TEXT
            )
        ''')
        self.conn.commit()
    
    def save_interaction(self, phone, name, summary, sentiment, follow_up):
        interaction_id = str(uuid.uuid4())
        self.conn.execute('''
            INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (interaction_id, phone, name, 'voice_call', summary, sentiment, follow_up, datetime.now(), 'completed'))
        self.conn.commit()
        return interaction_id

class MSpaceIntegration:
    def __init__(self, api_key, sender_id):
        self.api_key = api_key
        self.sender_id = sender_id
        self.customer_db = CustomerDatabase()
    
    def send_sms(self, phone_number, message):
        """Send SMS via mSpace API"""
        try:
            payload = {
                "api_key": self.api_key,
                "sender_id": self.sender_id,
                "to": phone_number,
                "message": message
            }
            response = requests.post(MSPACE_SMS_URL, json=payload)
            return response.json()
        except Exception as e:
            print(f"SMS Error: {e}")
            return None
    
    def send_conversation_summary(self, phone_number, customer_name, summary):
        """Send personalized conversation summary"""
        message = f"""
🤖 Hi {customer_name}! Thanks for chatting with Alex AI.

📋 Summary: {summary[:100]}...

💡 Interested? Reply:
• YES - Schedule callback
• INFO - Get product details
• STOP - Unsubscribe

Powered by AI & mSpace 🚀
        """.strip()
        return self.send_sms(phone_number, message)
    
    def handle_sms_response(self, phone_number, incoming_message):
        """Process incoming SMS responses"""
        message = incoming_message.upper().strip()
        
        if "YES" in message:
            self.schedule_callback(phone_number)
        elif "INFO" in message:
            self.send_detailed_product_info(phone_number)
        elif "STOP" in message:
            self.unsubscribe_customer(phone_number)
        else:
            # Use AI to generate contextual response
            ai_response = self.generate_ai_sms_response(incoming_message)
            self.send_sms(phone_number, ai_response)
    
    def schedule_callback(self, phone_number):
        callback_message = """
📞 Callback Scheduled!

Our team will call you within 24 hours.

Preferred time? Reply:
• MORNING (9-12 PM)
• AFTERNOON (1-5 PM)  
• EVENING (6-8 PM)

Or call us: 1234 (Free from Mobitel)
        """.strip()
        self.send_sms(phone_number, callback_message)
    
    def send_detailed_product_info(self, phone_number):
        info_message = """
🏆 Premium Package Details:

✅ 100GB Data + Unlimited Calls
✅ Free Netflix & Spotify
✅ 24/7 AI Assistant Support
✅ 5G Speed in Major Cities

💰 Only Rs. 2,999/month
🎁 First month FREE!

Visit: mobitel.lk/premium
Or dial *123# for instant activation
        """.strip()
        self.send_sms(phone_number, info_message)
    
    def create_ussd_menu(self):
        """Generate USSD menu for product demo"""
        return """
🤖 Alex AI Product Demo

1. Start Voice Demo
2. Get SMS Product Info
3. Schedule Callback
4. Chat with AI via SMS
5. Pricing & Offers
6. Nearest Mobitel Store
0. Exit

Reply with number choice
        """.strip()
    
    def generate_ai_sms_response(self, user_message):
        """Generate AI response for SMS conversations"""
        # This would integrate with your existing Gemini AI
        prompt = f"Generate a helpful, concise SMS response (max 160 chars) for: {user_message}"
        # Use your existing gemini_response function
        return "Thanks for your message! Our AI assistant will help you shortly. 🤖"

class EnhancedVoiceBot:
    def __init__(self):
        self.setup_ai_components()
        self.setup_voice_components()
        self.mspace = MSpaceIntegration(MSPACE_API_KEY, MSPACE_SENDER_ID)
        self.current_customer = {
            'phone': None,
            'name': None,
            'conversation_log': [],
            'sentiment_score': 0.0,
            'interest_level': 'low'
        }
    
    def setup_ai_components(self):
        """Initialize AI components"""
        global embeddings, vectorstore, retriever
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=150)
        system_prompt = """You are Alex, a friendly AI sales assistant for Mobitel. 
        Your goal is to:
        1. Understand customer needs
        2. Recommend suitable products
        3. Handle objections professionally
        4. Collect contact info for follow-up
        5. Keep responses conversational and under 30 seconds when spoken
        """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt), 
            ("human", "Context: {context}\n\nCustomer: {input}\n\nConversation History: {history}")
        ])
        
        self.embeddings = download_embeddings()
        self.vectorstore = None
        self.retriever = None
        
    def setup_voice_components(self):
        """Initialize voice recognition and TTS"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 0.9)
        
        # Try to set a more natural voice
        voices = self.engine.getProperty('voices')
        if voices and len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)  # Usually female voice
        
        self.vad = webrtcvad.Vad(3)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.FRAME_DURATION = 30
        self.FRAME_SIZE = int(self.RATE * self.FRAME_DURATION / 1000)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, 
                                 rate=self.RATE, input=True, 
                                 frames_per_buffer=self.FRAME_SIZE)
        
        self.is_bot_speaking = threading.Event()
        self.conversation_active = False
    
    def speak_message(self, text):
        """Enhanced TTS with emotion detection"""
        self.is_bot_speaking.set()
        
        # Add natural pauses and emphasis
        if "!" in text:
            self.engine.setProperty('rate', 140)  # Slower for excitement
        elif "?" in text:
            self.engine.setProperty('rate', 150)  # Medium for questions
        else:
            self.engine.setProperty('rate', 160)  # Normal speed
        
        self.engine.say(text)
        self.engine.runAndWait()
        self.is_bot_speaking.clear()
    
    def collect_customer_info(self):
        """Collect customer information during conversation"""
        self.speak_message("Before we continue, may I have your name and mobile number for our records?")
        
        # Listen for name
        name_response = self.get_voice_input_with_timeout(10)
        if name_response:
            # Extract name using simple NLP
            name = self.extract_name_from_response(name_response)
            self.current_customer['name'] = name
            self.speak_message(f"Great! Nice to meet you, {name}.")
        
        # Listen for phone number
        self.speak_message("And your mobile number?")
        phone_response = self.get_voice_input_with_timeout(15)
        if phone_response:
            phone = self.extract_phone_from_response(phone_response)
            self.current_customer['phone'] = phone
            self.speak_message("Perfect! I've noted your details.")
    
    def extract_name_from_response(self, response):
        """Extract name from voice response"""
        # Simple extraction - in production, use proper NLP
        words = response.split()
        for i, word in enumerate(words):
            if word.lower() in ['name', 'i\'m', 'im', 'called', 'call']:
                if i + 1 < len(words):
                    return words[i + 1].title()
        return "Customer"
    
    def extract_phone_from_response(self, response):
        """Extract phone number from voice response"""
        import re
        # Look for phone number patterns
        phone_pattern = r'(\d{10}|\d{3}[-.]?\d{3}[-.]?\d{4})'
        match = re.search(phone_pattern, response.replace(' ', ''))
        return match.group(1) if match else None
    
    def analyze_customer_sentiment(self, conversation_log):
        """Analyze customer sentiment throughout conversation"""
        positive_words = ['good', 'great', 'excellent', 'interested', 'yes', 'sure', 'okay']
        negative_words = ['no', 'not interested', 'busy', 'expensive', 'bad']
        
        positive_count = sum(1 for msg in conversation_log for word in positive_words if word in msg.lower())
        negative_count = sum(1 for msg in conversation_log for word in negative_words if word in msg.lower())
        
        if positive_count > negative_count:
            return min(0.8, 0.3 + (positive_count * 0.1))
        else:
            return max(0.2, 0.5 - (negative_count * 0.1))
    
    def get_voice_input_with_timeout(self, timeout=10):
        """Enhanced voice input with better timeout handling"""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        
        with sr.Microphone() as source:
            try:
                self.speak_message("I'm listening...")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
                result = recognizer.recognize_google(audio).lower()
                self.current_customer['conversation_log'].append(result)
                return result
            except sr.WaitTimeoutError:
                return "timeout"
            except sr.UnknownValueError:
                return "unclear"
            except Exception as e:
                print(f"Voice recognition error: {e}")
                return ""
    
    def enhanced_conversation_flow(self):
        """Main conversation flow with mSpace integration"""
        try:
            self.conversation_active = True
            update_status(True)
            
            # Greeting with time awareness
            greeting = self.get_time_based_greeting()
            self.speak_message(f"{greeting}! This is Alex, your AI assistant from Mobitel.")
            self.speak_message("I have some exciting offers to share. Do you have a couple of minutes?")
            
            # Wait for initial response
            initial_response = self.get_voice_input_with_timeout(8)
            
            if not initial_response or "no" in initial_response:
                self.speak_message("No worries! I'll send you the details via SMS. May I have your mobile number?")
                phone_response = self.get_voice_input_with_timeout(10)
                if phone_response:
                    phone = self.extract_phone_from_response(phone_response)
                    if phone:
                        self.mspace.send_detailed_product_info(phone)
                        self.speak_message("Perfect! Check your SMS for our latest offers. Have a great day!")
                return
            
            # Collect customer information
            self.collect_customer_info()
            
            # Main conversation loop
            conversation_rounds = 0
            max_rounds = 8
            
            while conversation_rounds < max_rounds and self.conversation_active:
                if not self.is_bot_speaking.is_set():
                    # Generate contextual AI response
                    history = " | ".join(self.current_customer['conversation_log'][-3:])  # Last 3 exchanges
                    
                    if conversation_rounds == 0:
                        ai_prompt = "Start the sales conversation by asking about their current mobile plan and needs"
                    else:
                        ai_prompt = f"Continue the sales conversation based on customer responses: {history}"
                    
                    ai_response = self.gemini_response(ai_prompt, history)
                    self.speak_message(ai_response)
                    
                    # Listen for customer response
                    customer_response = self.get_voice_input_with_timeout(12)
                    
                    if customer_response == "timeout":
                        self.speak_message("Are you still there? I'm here to help with any questions.")
                        customer_response = self.get_voice_input_with_timeout(8)
                        if customer_response == "timeout":
                            break
                    
                    if customer_response and customer_response != "unclear":
                        # Check for conversation enders
                        if any(phrase in customer_response for phrase in ["thank you", "goodbye", "not interested", "hang up"]):
                            break
                        
                        # Analyze interest level
                        if any(phrase in customer_response for phrase in ["interested", "tell me more", "sounds good", "yes"]):
                            self.current_customer['interest_level'] = 'high'
                        elif any(phrase in customer_response for phrase in ["maybe", "think about it", "later"]):
                            self.current_customer['interest_level'] = 'medium'
                    
                    conversation_rounds += 1
            
            # Conversation conclusion with follow-up
            self.conclude_conversation()
            
        except Exception as e:
            print(f"Conversation error: {e}")
            self.speak_message("I apologize, there seems to be a technical issue. Thank you for your time!")
        finally:
            self.conversation_active = False
            update_status(False)
    
    def conclude_conversation(self):
        """Conclude conversation with appropriate follow-up"""
        customer_name = self.current_customer.get('name', 'valued customer')
        customer_phone = self.current_customer.get('phone')
        
        # Generate conversation summary
        conversation_summary = self.generate_conversation_summary()
        
        # Analyze sentiment
        sentiment_score = self.analyze_customer_sentiment(self.current_customer['conversation_log'])
        self.current_customer['sentiment_score'] = sentiment_score
        
        if sentiment_score > 0.6:  # Positive interaction
            self.speak_message(f"Thank you {customer_name}! I'll send you a summary and our best offer via SMS.")
            follow_up_required = True
        elif sentiment_score > 0.3:  # Neutral interaction
            self.speak_message(f"Thanks for your time, {customer_name}. I'll send you some information to review at your convenience.")
            follow_up_required = True
        else:  # Negative interaction
            self.speak_message(f"Thank you for your time, {customer_name}. Have a wonderful day!")
            follow_up_required = False
        
        # Send SMS follow-up if phone number available
        if customer_phone and follow_up_required:
            self.mspace.send_conversation_summary(customer_phone, customer_name, conversation_summary)
        
        # Save interaction to database
        if customer_phone:
            self.mspace.customer_db.save_interaction(
                customer_phone, customer_name, conversation_summary, 
                sentiment_score, follow_up_required
            )
    
    def generate_conversation_summary(self):
        """Generate AI-powered conversation summary"""
        conversation_text = " ".join(self.current_customer['conversation_log'])
        summary_prompt = f"Summarize this sales conversation in 2-3 sentences, focusing on customer needs and interest level: {conversation_text}"
        return self.gemini_response(summary_prompt, "")
    
    def gemini_response(self, input_text, history=""):
        """Enhanced Gemini response with conversation context"""
        try:
            qa_chain = create_stuff_documents_chain(self.llm, self.prompt)
            if self.retriever:
                rag_chain = create_retrieval_chain(self.retriever, qa_chain)
                response = rag_chain.invoke({
                    "input": input_text,
                    "history": history
                })
                return response['answer']
            else:
                # Fallback without RAG
                response = self.llm.invoke(f"Context: Mobitel sales conversation\nHistory: {history}\nCustomer: {input_text}")
                return response.content
        except Exception as e:
            print(f"AI Response Error: {e}")
            return "I understand. Let me help you with that."
    
    def get_time_based_greeting(self):
        """Generate time-appropriate greeting"""
        hour = datetime.now().hour
        if hour < 12:
            return "Good morning"
        elif hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"

# Initialize the enhanced bot
enhanced_bot = EnhancedVoiceBot()

def upload_and_train():
    """Enhanced PDF training with better error handling"""
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        try:
            # Show progress
            progress_window = tk.Toplevel(root)
            progress_window.title("Training AI...")
            progress_window.geometry("300x100")
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=20)
            progress_bar.start()
            
            # Clear old data
            if os.path.exists("Data/"):
                shutil.rmtree("Data/")
            os.makedirs("Data/", exist_ok=True)
            
            # Copy and process PDF
            new_pdf_path = os.path.join("Data/", os.path.basename(file_path))
            shutil.copy(file_path, new_pdf_path)
            
            # Load and split the new PDF
            extracted_data = load_pdf_file("Data/")
            text_chunks = text_split(extracted_data)
            
            # Recreate vectorstore and retriever
            enhanced_bot.vectorstore = Chroma.from_texts(texts=text_chunks, embedding=enhanced_bot.embeddings)
            enhanced_bot.retriever = enhanced_bot.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
            )
            
            progress_bar.stop()
            progress_window.destroy()
            
            messagebox.showinfo("Success", f"Successfully trained on {os.path.basename(file_path)}!\n\nAlex is now ready with updated product knowledge.")
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Failed to process PDF: {str(e)}")

def update_status(active):
    """Update UI status with enhanced visual feedback"""
    if active:
        blinking.set(True)
        status_label.config(text="🔊 Alex is actively talking...", fg="green")
        start_button.config(state="disabled", text="Conversation in Progress")
    else:
        blinking.set(False)
        phone_icon.config(fg="black")
        status_label.config(text="🤖 Alex is ready to help", fg="blue")
        start_button.config(state="normal", text="Start Talking with Alex")

def blink_icon():
    """Enhanced blinking animation"""
    if blinking.get():
        current_color = phone_icon.cget("fg")
        new_color = "green" if current_color == "black" else "black"
        phone_icon.config(fg=new_color)
    root.after(500, blink_icon)

def show_analytics():
    """Show conversation analytics dashboard"""
    analytics_window = tk.Toplevel(root)
    analytics_window.title("Alex AI Analytics")
    analytics_window.geometry("500x400")
    
    # Add analytics content here
    tk.Label(analytics_window, text="📊 Conversation Analytics", font=("Helvetica", 16, "bold")).pack(pady=10)
    
    # Sample analytics - replace with real data from database
    analytics_text = """
📞 Total Conversations: 47
✅ Successful Engagements: 32 (68%)
📱 SMS Follow-ups Sent: 28
🎯 High Interest Customers: 15
📈 Average Sentiment Score: 0.72
⏱️ Average Call Duration: 4.2 minutes
    """
    
    tk.Label(analytics_window, text=analytics_text, font=("Helvetica", 10), justify="left").pack(pady=20)

# Enhanced GUI setup
root = tk.Tk()
root.title("Alex AI - Enhanced Voice Bot with mSpace Integration")
root.geometry("500x450")
root.configure(bg="#f0f0f0")

# Header
header_frame = tk.Frame(root, bg="#2c3e50", height=80)
header_frame.pack(fill="x")
header_frame.pack_propagate(False)

tk.Label(header_frame, text="🤖 Alex AI Assistant", font=("Helvetica", 18, "bold"), 
         fg="white", bg="#2c3e50").pack(pady=20)

blinking = tk.BooleanVar(value=False)

# Phone icon with enhanced styling
phone_icon = tk.Label(root, text="📞", font=("Helvetica", 50), fg="black", bg="#f0f0f0")
phone_icon.pack(pady=20)

# Status label
status_label = tk.Label(root, text="🤖 Alex is ready to help", font=("Helvetica", 12), 
                       fg="blue", bg="#f0f0f0")
status_label.pack(pady=5)

# Button frame
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=20)

# Enhanced buttons
train_button = tk.Button(button_frame, text="📤 Upload & Train New PDF", 
                        font=("Helvetica", 11), bg="#3498db", fg="white",
                        command=upload_and_train, padx=20, pady=8)
train_button.pack(pady=5)

start_button = tk.Button(button_frame, text="🎙️ Start Talking with Alex", 
                        font=("Helvetica", 12, "bold"), bg="#27ae60", fg="white",
                        command=lambda: threading.Thread(target=enhanced_bot.enhanced_conversation_flow).start(),
                        padx=20, pady=10)
start_button.pack(pady=5)

analytics_button = tk.Button(button_frame, text="📊 View Analytics", 
                           font=("Helvetica", 11), bg="#e74c3c", fg="white",
                           command=show_analytics, padx=20, pady=8)
analytics_button.pack(pady=5)

# Footer
footer_frame = tk.Frame(root, bg="#34495e", height=60)
footer_frame.pack(fill="x", side="bottom")
footer_frame.pack_propagate(False)

tk.Label(footer_frame, text="Powered by Mobitel mSpace API & Google AI", 
         font=("Helvetica", 10), fg="white", bg="#34495e").pack(pady=15)

blink_icon()
root.mainloop()