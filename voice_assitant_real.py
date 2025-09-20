import tkinter as tk
from tkinter import filedialog
import pyttsx3
import speech_recognition as sr
import webrtcvad
import threading
import pyaudio
import numpy as np
from datetime import datetime
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

from src.helper import load_pdf_file, text_split, download_embeddings

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Setup LLM and prompt
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=75)
system_prompt = "You are a helpful assistant. Answer the question based on the documents."
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Context: {context}\n\nQuestion: {input}")])

# Global variables
embeddings = download_embeddings()
vectorstore = None
retriever = None
conversation_memory = []
is_bot_speaking = threading.Event()


# Voice setup
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 0.9)

vad = webrtcvad.Vad(3)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAME_SIZE)

def speak_message(text):
    is_bot_speaking.set()
    engine.say(text)
    engine.runAndWait()
    is_bot_speaking.clear()

def is_speech():
    audio_frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
    return vad.is_speech(audio_frame, RATE)

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            while not is_speech():
                pass
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio).lower()
        except:
            return ""

def get_greeting():
    hour = datetime.now().hour
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

def gemini_response(input_text):
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain) 
    response = rag_chain.invoke({"input": input_text})
    return response['answer']


def run_conversation():
    update_status(True)
    speak_message(f"{get_greeting()}, this is Alex.")
    speak_message("Could you please spare a moment to talk with me?")

    while True:
        if not is_bot_speaking.is_set():
            user_response = get_voice_input()
            if user_response:
                if any(x in user_response for x in ["yes", "go ahead", "sure", "okay"]):
                    speak_message(gemini_response("Promote product and ask questions also from the customer"))
                    while True:
                        if not is_bot_speaking.is_set():
                            user_response = get_voice_input()
                            if user_response:
                                if any(x in user_response for x in ["thank you", "see you", "not interested"]):
                                    speak_message("Thank you for your time! Have a great day!")
                                    update_status(False)
                                    return
                                elif any(x in user_response for x in ["busy", "no time"]):
                                    speak_message("No problem! I’ll reach out another time. Have a good day!")
                                    update_status(False)
                                    return
                                else:
                                    speak_message(gemini_response(user_response))
                elif "no" in user_response:
                    speak_message("Okay, thank you for your time. Have a great day!")
                    update_status(False)
                    return
                else:
                    speak_message("I didn't quite catch that. Do you want to continue our conversation?")

            else:
                speak_message("Are you still there? I'd be happy to continue our conversation.")

def upload_and_train():
    global vectorstore, retriever  

    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        try:
            shutil.rmtree("Data/")  
            os.makedirs("Data/", exist_ok=True)
            new_pdf_path = os.path.join("Data/", os.path.basename(file_path))
            shutil.copy(file_path, new_pdf_path)

            
            extracted_data = load_pdf_file("Data/")
            text_chunks = text_split(extracted_data)

            
            vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5, "score_threshold": 0.7}
            )

            messagebox.showinfo("Success", "Successfully trained on new PDF. Previous context cleared.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process PDF: {str(e)}")



def update_status(active):
    if active:
        blinking.set(True)
    else:
        blinking.set(False)
        phone_icon.config(fg="black")

def blink_icon():
    if blinking.get():
        current_color = phone_icon.cget("fg")
        phone_icon.config(fg="green" if current_color == "black" else "black")
    root.after(500, blink_icon)


root = tk.Tk()
root.title("Voice AI Bot - Alex")
root.geometry("400x350")

blinking = tk.BooleanVar(value=False)

phone_icon = tk.Label(root, text="📞", font=("Helvetica", 40), fg="black")
phone_icon.pack(pady=20)

train_button = tk.Button(root, text="📤 Upload & Train New PDF", font=("Helvetica", 12), command=upload_and_train)
train_button.pack(pady=10)

start_button = tk.Button(root, text="Start Talking with Alex", font=("Helvetica", 14), command=lambda: threading.Thread(target=run_conversation).start())
start_button.pack(pady=10)

blink_icon()
root.mainloop()

