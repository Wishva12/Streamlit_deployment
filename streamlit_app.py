# streamlit_app.py - Minimal version with Chat & SMS only
import streamlit as st
import requests
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
import os
import time
import pandas as pd
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

# Conditional import for AI chat engine with FAISS
try:
    from ai_chat_engine_faiss import AIProductChatEngine, StreamlitChatInterface
    AI_CHAT_AVAILABLE = True
    AI_CHAT_TYPE = "FAISS"
except ImportError:
    AI_CHAT_AVAILABLE = False
    AI_CHAT_TYPE = "None"
    
    # Simple fallback chat
    class SimpleChatInterface:
        def __init__(self):
            if 'simple_chat_messages' not in st.session_state:
                st.session_state.simple_chat_messages = []
        
        def render_chat_interface(self, customer_name):
            st.markdown("### 💬 ALEX AI Chat Assistant")
            
            # Display messages
            for message in st.session_state.simple_chat_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about our mobile packages..."):
                # Add user message
                st.session_state.simple_chat_messages.append({
                    "role": "user", 
                    "content": prompt
                })
                
                # Generate response
                response = self.generate_response(prompt, customer_name)
                st.session_state.simple_chat_messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.rerun()
        
        def generate_response(self, user_input, customer_name):
            user_lower = user_input.lower()
            
            if any(word in user_lower for word in ['hello', 'hi', 'hey']):
                return f"Hello {customer_name}! I'm ALEX from Mobitel. How can I help you with our mobile packages today?"
            
            elif any(word in user_lower for word in ['price', 'cost', 'package']):
                return """Here are our current packages:

🌟 **Premium Package** - Rs.2,999/month
• 100GB Data, Unlimited Calls, Netflix, Spotify

👨‍👩‍👧‍👦 **Family Package** - Rs.4,999/month  
• 300GB Shared Data, 5 SIM Cards, Disney+ & Netflix

💼 **Business Package** - Rs.7,999/month
• Unlimited Data, Priority Network, 24/7 Support

Which package interests you most?"""
            
            elif any(word in user_lower for word in ['premium']):
                return """🌟 **Premium Package Details**

**Price:** Rs.2,999/month
**Perfect for:** Heavy data users and entertainment lovers

**What you get:**
• 100GB high-speed data
• Unlimited local calls
• Free Netflix subscription
• Free Spotify Premium
• 5G ready network
• 50GB hotspot sharing

**Special Offer:** First month FREE + Samsung Galaxy A34 (Worth Rs.79,900)

Would you like me to help you activate this package?"""
            
            elif any(word in user_lower for word in ['family']):
                return """👨‍👩‍👧‍👦 **Family Package Details**

**Price:** Rs.4,999/month
**Perfect for:** Families with 3+ members

**What you get:**
• 300GB shared data across 5 SIM cards
• Unlimited family calls
• Disney+ and Netflix subscriptions
• Parental controls
• Family locator service

**Special Offer:** 50% off for 3 months + Free 4G Router

Great for keeping the whole family connected!"""
            
            elif any(word in user_lower for word in ['business']):
                return """💼 **Business Package Details**

**Price:** Rs.7,999/month
**Perfect for:** Growing businesses and remote teams

**What you get:**
• Unlimited high-speed data
• Priority network access
• 24/7 dedicated support
• 1TB cloud storage
• Conference calling features
• VPN access

**Special Offer:** Free setup + 6 months Microsoft 365

Ideal for maintaining business productivity!"""
            
            else:
                return f"Thank you for your question, {customer_name}. I can help you with information about our Premium, Family, or Business packages. What would you like to know?"
        
        def render_chat_sidebar(self):
            st.sidebar.markdown("### 💬 Chat Controls")
            if st.sidebar.button("Clear Chat"):
                st.session_state.simple_chat_messages = []
                st.rerun()

load_dotenv()

MSPACE_API_KEY = os.getenv("MSPACE_API_KEY", "demo_api_key_12345")
MSPACE_SENDER_ID = os.getenv("MSPACE_SENDER_ID", "MOBITEL")
MSPACE_SMS_URL = "https://api.mspace.lk/sms/send"

# Product Configuration
PRODUCTS = {
    "premium": {
        "name": "Premium Package",
        "price": 2999,
        "features": ["100GB Data", "Unlimited Calls", "Free Netflix", "Free Spotify", "5G Ready", "Hotspot 50GB"],
        "description": "Our flagship package with premium entertainment and unlimited connectivity",
        "discount": "🎁 First month FREE + Free Samsung Galaxy A34 (Worth Rs.79,900)",
        "target_audience": "Heavy data users, entertainment lovers"
    },
    "family": {
        "name": "Family Package",
        "price": 4999,
        "features": ["300GB Shared Data", "5 SIM Cards", "Unlimited Family Calls", "Disney+ & Netflix", "Parental Controls", "Family Locator"],
        "description": "Complete family connectivity solution with premium entertainment",
        "discount": "🎁 3 months at 50% + Free 4G Router + Family Safety Suite",
        "target_audience": "Families with 3+ members, parents with children"
    },
    "business": {
        "name": "Business Package",
        "price": 7999,
        "features": ["Unlimited Data", "Priority Network", "24/7 Support", "Cloud Storage 1TB", "Conference Calling", "VPN Access"],
        "description": "Enterprise-grade connectivity for growing businesses",
        "discount": "💼 Free setup + 6 months Microsoft 365 + Dedicated account manager",
        "target_audience": "SMEs, remote teams, digital businesses"
    }
}

# Database initialization
def init_database():
    conn = sqlite3.connect('product_promotion.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS customer_interactions (
            id TEXT PRIMARY KEY,
            phone_number TEXT,
            customer_name TEXT,
            selected_product TEXT,
            interaction_type TEXT,
            status TEXT,
            created_at TIMESTAMP,
            notes TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sms_campaigns (
            id TEXT PRIMARY KEY,
            phone_number TEXT,
            product_name TEXT,
            message_content TEXT,
            delivery_status TEXT,
            sent_at TIMESTAMP,
            campaign_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

# mSpace SMS Integration
class MSpaceProductPromotion:
    def __init__(self):
        self.api_key = MSPACE_API_KEY
        self.sender_id = MSPACE_SENDER_ID
        
    def send_personalized_sms(self, phone_number, product_key, customer_name=None, context="general"):
        product = PRODUCTS[product_key]
        customer_name = customer_name or "Valued Customer"
        
        if context == "followup":
            message = f"""🎯 Hi {customer_name}! Following up on your interest in our {product['name']}.

🎁 EXCLUSIVE OFFER: {product['discount']}

Perfect for: {product['target_audience']}

Reply YES for instant activation or CALL for specialist support.

ALEX AI Assistant 🤖 | Mobitel"""

        elif context == "competitive":
            message = f"""🏆 {customer_name}, ready for better mobile service?

Our {product['name']} offers:
✅ Better value than competitors
✅ Island-wide coverage
✅ 24/7 local support

🎁 SWITCHING BONUS: {product['discount']}

Reply SWITCH for porting assistance!"""

        else:  # general context
            message = f"""🌟 Hi {customer_name}! Discover your perfect mobile solution.

🏆 {product['name']} - Rs.{product['price']}/month

✅ Features:
{chr(10).join([f'• {feature}' for feature in product['features'][:4]])}

🎁 SPECIAL OFFER: {product['discount']}

Reply YES for activation or CHAT for more info.

Mobitel - Your Connected Future 🚀"""

        try:
            # Simulate mSpace API call (replace with actual API call)
            payload = {
                "api_key": self.api_key,
                "sender_id": self.sender_id,
                "to": phone_number,
                "message": message
            }
            
            # For demo purposes - simulate API response
            response_data = {
                "status": "success",
                "message_id": f"msg_{uuid.uuid4().hex[:8]}",
                "delivery_status": "sent"
            }
            
            # Log to database
            self.log_sms_campaign(phone_number, product['name'], message, "sent", context)
            
            return response_data
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def log_sms_campaign(self, phone_number, product_name, message_content, status, campaign_type="general"):
        conn = sqlite3.connect('product_promotion.db')
        conn.execute('''
            INSERT INTO sms_campaigns VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), phone_number, product_name, message_content, 
              status, datetime.now(), campaign_type))
        conn.commit()
        conn.close()

# Streamlit Configuration
st.set_page_config(
    page_title="🚀 Mobitel Promotion Hub",
    page_icon="📱",
    layout="wide"
)

# Initialize
init_database()
mspace_api = MSpaceProductPromotion()

# Session state initialization
if 'show_ai_chat' not in st.session_state:
    st.session_state.show_ai_chat = False
if 'customer_name' not in st.session_state:
    st.session_state.customer_name = ""
if 'customer_phone' not in st.session_state:
    st.session_state.customer_phone = ""

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #e31837 0%, #ff6b6b 100%);
        color: white;
        text-align: center;
        font-size: 2.2em;
        font-weight: bold;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(227, 24, 55, 0.3);
    }
    
    .product-card {
        border: 2px solid #e31837;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('''
<div class="main-header">
    🚀 MOBITEL PROMOTION HUB
    <br><small>AI Chat & SMS Campaign Platform</small>
</div>
''', unsafe_allow_html=True)

# Sidebar - Customer Profile
with st.sidebar:
    st.markdown("### 👤 Customer Profile")
    
    customer_name = st.text_input("Customer Name", 
                                  value=st.session_state.customer_name,
                                  placeholder="John Doe")
    customer_phone = st.text_input("Phone Number", 
                                   value=st.session_state.customer_phone,
                                   placeholder="+94771234567")
    
    # Update session state
    st.session_state.customer_name = customer_name
    st.session_state.customer_phone = customer_phone
    
    if customer_phone and customer_name:
        st.success("✅ Profile Active")
        st.markdown(f"**Name:** {customer_name}")
        st.markdown(f"**Phone:** {customer_phone}")
    
    st.markdown("---")
    
    # Quick Stats
    conn = sqlite3.connect('product_promotion.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM customer_interactions")
    total_interactions = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM sms_campaigns")
    total_sms = cursor.fetchone()[0]
    conn.close()
    
    st.metric("Total Interactions", total_interactions)
    st.metric("SMS Campaigns", total_sms)

# Main Content - Two Column Layout
col1, col2 = st.columns(2)

# SMS Campaign Section
with col1:
    st.markdown("## 📱 SMS Campaigns")
    
    if st.session_state.customer_phone:
        campaign_type = st.selectbox("Campaign Type", 
            ["🌟 General Introduction", "🏆 Competitive Offer", "🔄 Follow-up"])
        
        st.markdown("#### Choose Package:")
        for key, product in PRODUCTS.items():
            with st.expander(f"{product['name']} - Rs.{product['price']}/month"):
                st.markdown(f"**Target:** {product['target_audience']}")
                st.markdown(f"**Features:** {', '.join(product['features'][:3])}")
                
                context_map = {
                    "🌟 General Introduction": "general",
                    "🏆 Competitive Offer": "competitive", 
                    "🔄 Follow-up": "followup"
                }
                
                if st.button(f"📤 Send SMS", key=f"sms_{key}"):
                    with st.spinner("Sending SMS..."):
                        result = mspace_api.send_personalized_sms(
                            st.session_state.customer_phone, 
                            key, 
                            st.session_state.customer_name, 
                            context_map[campaign_type]
                        )
                        if result["status"] == "success":
                            st.success(f"✅ SMS sent! ID: {result['message_id']}")
                        else:
                            st.error(f"❌ Failed: {result['message']}")
    else:
        st.info("Enter customer details to send SMS campaigns")

# AI Chat Section
with col2:
    st.markdown("## 🤖 AI Chat Assistant")
    
    if st.session_state.customer_phone and st.session_state.customer_name:
        chat_button = st.button(
            "🛑 End Chat" if st.session_state.show_ai_chat else "🚀 Start Chat",
            key="toggle_chat"
        )
        
        if chat_button:
            st.session_state.show_ai_chat = not st.session_state.show_ai_chat
            if not st.session_state.show_ai_chat:
                # Clear chat history
                if hasattr(st.session_state, 'simple_chat_messages'):
                    st.session_state.simple_chat_messages = []
                if hasattr(st.session_state, 'chat_messages'):
                    st.session_state.chat_messages = []
                st.rerun()
        
        # Chat Interface
        if st.session_state.show_ai_chat:
            if 'chat_system' not in st.session_state:
                if AI_CHAT_AVAILABLE:
                    try:
                        chat_engine = AIProductChatEngine()
                        st.session_state.chat_system = StreamlitChatInterface(chat_engine)
                        st.success("✅ Advanced AI Chat Ready!")
                    except:
                        st.session_state.chat_system = SimpleChatInterface()
                        st.info("✅ Basic AI Chat Ready!")
                else:
                    st.session_state.chat_system = SimpleChatInterface()
            
            # Render chat
            st.session_state.chat_system.render_chat_interface(st.session_state.customer_name)
    else:
        st.info("Enter customer details to start AI chat")

# Products Showcase
st.markdown("---")
st.markdown("## 📦 Our Packages")

product_cols = st.columns(len(PRODUCTS))
for idx, (key, product) in enumerate(PRODUCTS.items()):
    with product_cols[idx]:
        st.markdown(f"""
        <div class="product-card">
            <h3>{product['name']}</h3>
            <h2>Rs.{product['price']:,}/month</h2>
            <p>{product['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        for feature in product['features'][:4]:
            st.markdown(f"• {feature}")
        
        if st.session_state.customer_phone and st.button(f"Send {product['name']} Info", key=f"quick_{key}"):
            result = mspace_api.send_personalized_sms(
                st.session_state.customer_phone, key, st.session_state.customer_name
            )
            if result["status"] == "success":
                st.success("SMS sent!")

# Analytics
if st.button("📊 View Analytics"):
    st.markdown("### 📈 Campaign Analytics")
    
    conn = sqlite3.connect('product_promotion.db')
    
    # SMS performance
    df_sms = pd.read_sql_query("""
        SELECT campaign_type, COUNT(*) as count, product_name
        FROM sms_campaigns 
        GROUP BY campaign_type, product_name
    """, conn)
    
    if not df_sms.empty:
        fig = px.bar(df_sms, x='campaign_type', y='count', color='product_name')
        st.plotly_chart(fig, use_container_width=True)
    
    conn.close()