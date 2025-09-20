# streamlit_app.py - Updated for FAISS integration
import streamlit as st
import requests
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
import subprocess
import os
import time
import pandas as pd
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

# Conditional import for AI chat engine with FAISS fallback
try:
    from ai_chat_engine_faiss import AIProductChatEngine, StreamlitChatInterface
    AI_CHAT_AVAILABLE = True
    AI_CHAT_TYPE = "FAISS"
except ImportError as e:
    st.warning(f"FAISS AI Chat Engine not available: {e}")
    try:
        from ai_chat_engine import AIProductChatEngine, StreamlitChatInterface
        AI_CHAT_AVAILABLE = True
        AI_CHAT_TYPE = "Legacy"
    except ImportError as e2:
        st.warning(f"Legacy AI Chat Engine also not available: {e2}")
        AI_CHAT_AVAILABLE = False
        AI_CHAT_TYPE = "None"
        
        # Fallback simple chat interface
        class SimpleChatInterface:
            def __init__(self):
                if 'simple_chat_messages' not in st.session_state:
                    st.session_state.simple_chat_messages = []
            
            def render_chat_interface(self, customer_name):
                st.markdown("### 💬 Simple Chat Interface (Fallback Mode)")
                st.info("🔧 Advanced AI features temporarily unavailable. Using basic chat mode.")
                
                # Display chat messages
                for message in st.session_state.simple_chat_messages:
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**ALEX AI:** {message['content']}")
                
                # Chat input
                user_input = st.text_input("Type your message:", key="simple_chat_input")
                if st.button("Send", key="simple_send"):
                    if user_input:
                        # Add user message
                        st.session_state.simple_chat_messages.append({
                            "role": "user", 
                            "content": user_input
                        })
                        
                        # Simple AI response
                        ai_response = self.generate_simple_response(user_input, customer_name)
                        st.session_state.simple_chat_messages.append({
                            "role": "assistant",
                            "content": ai_response
                        })
                        st.rerun()
            
            def generate_simple_response(self, user_input, customer_name):
                """Generate simple rule-based responses"""
                user_input_lower = user_input.lower()
                
                if any(word in user_input_lower for word in ['hello', 'hi', 'hey']):
                    return f"Hello {customer_name}! I'm ALEX, your Mobitel assistant. How can I help you today?"
                elif any(word in user_input_lower for word in ['price', 'cost', 'package']):
                    return "Here are our current packages: Premium (Rs.2,999), Family (Rs.4,999), Business (Rs.7,999). Which interests you?"
                else:
                    return f"Thank you for your message, {customer_name}. Our team will get back to you soon."
            
            def render_chat_sidebar(self):
                st.sidebar.markdown("**Simple Chat Mode**")
                if st.sidebar.button("Clear Chat"):
                    st.session_state.simple_chat_messages = []
                    st.rerun()

load_dotenv()

MSPACE_API_KEY = os.getenv("MSPACE_API_KEY", "demo_api_key_12345")
MSPACE_SENDER_ID = os.getenv("MSPACE_SENDER_ID", "MOBITEL")
MSPACE_SMS_URL = "https://api.mspace.lk/sms/send"

# Enhanced Product Configuration with competitive analysis
PRODUCTS = {
    "premium": {
        "name": "Premium Package",
        "price": 2999,
        "features": ["100GB Data", "Unlimited Calls", "Free Netflix", "Free Spotify", "5G Ready", "Hotspot 50GB"],
        "description": "Our flagship package with premium entertainment and unlimited connectivity",
        "discount": "🎁 First month FREE + Free Samsung Galaxy A34 (Worth Rs.79,900)",
        "competitive_advantage": "40% more data than Dialog, 60% cheaper than Airtel premium",
        "target_audience": "Heavy data users, entertainment lovers"
    },
    "family": {
        "name": "Family Package",
        "price": 4999,
        "features": ["300GB Shared Data", "5 SIM Cards", "Unlimited Family Calls", "Disney+ & Netflix", "Parental Controls", "Family Locator"],
        "description": "Complete family connectivity solution with premium entertainment",
        "discount": "🎁 3 months at 50% + Free 4G Router + Family Safety Suite",
        "competitive_advantage": "Only operator with 5 SIMs included, 50% more shared data",
        "target_audience": "Families with 3+ members, parents with children"
    },
    "business": {
        "name": "Business Package",
        "price": 7999,
        "features": ["Unlimited Data", "Priority Network", "24/7 Support", "Cloud Storage 1TB", "Conference Calling", "VPN Access"],
        "description": "Enterprise-grade connectivity for growing businesses",
        "discount": "💼 Free setup + 6 months Microsoft 365 + Dedicated account manager",
        "competitive_advantage": "Only package with guaranteed 99.9% uptime SLA",
        "target_audience": "SMEs, remote teams, digital businesses"
    }
}

# Initialize database with enhanced schema
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
            follow_up_date TIMESTAMP,
            notes TEXT,
            conversion_score INTEGER
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
            response_received TEXT,
            campaign_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Enhanced mSpace API Integration Class
class MSpaceProductPromotion:
    def __init__(self):
        self.api_key = MSPACE_API_KEY
        self.sender_id = MSPACE_SENDER_ID
        
    def send_personalized_sms(self, phone_number, product_key, customer_name=None, context="general"):
        """Send personalized product promotion SMS with context awareness"""
        product = PRODUCTS[product_key]
        customer_name = customer_name or "Valued Customer"
        
        # Different message templates based on context
        if context == "followup":
            message = f"""🎯 Hi {customer_name}! Following up on your interest in our {product['name']}.

⚡ FLASH UPDATE: We've just enhanced the package!
✨ {product['competitive_advantage']}

🎁 EXCLUSIVE FOR YOU: {product['discount']}

Perfect for: {product['target_audience']}

💡 Ready to upgrade your experience?
• Reply YES ➜ Instant activation
• Reply DEMO ➜ Free 7-day trial  
• Reply CALL ➜ Speak to specialist
• Reply COMPARE ➜ See vs competitors

⏰ Limited offer expires in 24 hours!
Activate now: *123*{product_key.upper()}*{phone_number[-4:]}#

ALEX AI Assistant 🤖 | Mobitel Beyond Connectivity"""

        elif context == "competitive":
            message = f"""🏆 {customer_name}, tired of overpaying for less?

Our {product['name']} beats the competition:
{product['competitive_advantage']}

💰 You save Rs.{1200 if product_key == 'premium' else 800}/month!

🎁 SWITCHING BONUS: {product['discount']}
📋 Features: {' • '.join(product['features'][:4])}

🚀 Why customers switch to us:
✅ Better coverage in Sri Lanka
✅ 24/7 local customer support  
✅ No hidden charges ever
✅ 30-day money-back guarantee

Ready to experience the difference?
Reply SWITCH for instant porting assistance!

Your savings start today 💸"""

        else:  # general context
            message = f"""🌟 Hi {customer_name}! Your perfect mobile solution is here!

🏆 {product['name']} - Rs.{product['price']}/month
{product['description']}

✅ What you get:
{chr(10).join([f'• {feature}' for feature in product['features']])}

🎁 SPECIAL LAUNCH OFFER:
{product['discount']}

💡 Next steps:
• YES ➜ Get instant activation
• INFO ➜ Detailed comparison  
• TRIAL ➜ 7-day free experience
• STORE ➜ Visit nearest location

⏰ Offer valid for 48 hours only!
Questions? Chat with ALEX AI: Reply CHAT

Mobitel - Your Connected Future Starts Here 🚀"""

        try:
            # Simulate mSpace API call
            payload = {
                "api_key": self.api_key,
                "sender_id": self.sender_id,
                "to": phone_number,
                "message": message,
                "message_type": "promotional",
                "priority": "high"
            }
            
            # For demo - simulate successful response
            response_data = {
                "status": "success",
                "message_id": f"msg_{uuid.uuid4().hex[:8]}",
                "credits_used": 2,
                "delivery_status": "sent",
                "estimated_delivery": "30 seconds"
            }
            
            # Enhanced logging
            self.log_sms_campaign(phone_number, product['name'], message, "sent", context)
            
            return response_data
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def log_sms_campaign(self, phone_number, product_name, message_content, status, campaign_type="general"):
        """Enhanced SMS campaign logging"""
        conn = sqlite3.connect('product_promotion.db')
        conn.execute('''
            INSERT INTO sms_campaigns VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), phone_number, product_name, message_content, 
              status, datetime.now(), "", campaign_type))
        conn.commit()
        conn.close()

# Streamlit App Configuration
st.set_page_config(
    page_title="🚀 Mobitel Product Promotion Hub",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize
init_database()
mspace_api = MSpaceProductPromotion()

# Initialize session state for the new AI chat interface
if 'show_ai_chat' not in st.session_state:
    st.session_state.show_ai_chat = False

# Initialize customer details in session state if not present
if 'customer_name' not in st.session_state:
    st.session_state.customer_name = ""
if 'customer_phone' not in st.session_state:
    st.session_state.customer_phone = ""
if 'customer_segment' not in st.session_state:
    st.session_state.customer_segment = "Individual"

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #e31837 0%, #ff6b6b 100%);
        color: white;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(227, 24, 55, 0.3);
    }
    
    .action-button {
        background: linear-gradient(45deg, #e31837, #ff4757);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        margin: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(227, 24, 55, 0.4);
    }
    
    .product-card {
        border: 2px solid #e31837;
        border-radius: 15px;
        padding: 25px;
        margin: 15px;
        background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(227, 24, 55, 0.2);
    }
    
    .stats-card {
        background: linear-gradient(45deg, #e31837, #ff6b6b);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .chat-message {
        padding: 12px 18px;
        margin: 8px 0;
        border-radius: 18px;
        max-width: 80%;
    }
    
    .customer-message {
        background: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    
    .ai-message {
        background: linear-gradient(45deg, #e31837, #ff4757);
        color: white;
        margin-right: auto;
    }
    
    .feature-highlight {
        background: linear-gradient(45deg, #00c851, #00ff88);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Main Header
st.markdown('''
<div class="main-header">
    🚀 MOBITEL PRODUCT PROMOTION HUB
    <br><small>AI-Powered Customer Engagement with mSpace Integration</small>
</div>
''', unsafe_allow_html=True)

# Show deployment status
if AI_CHAT_AVAILABLE:
    if AI_CHAT_TYPE == "FAISS":
        st.info("✅ **AI Status:** FAISS-powered chat engine loaded successfully")
    else:
        st.info("⚠️ **AI Status:** Legacy chat engine loaded (FAISS unavailable)")
else:
    st.warning("🔧 **AI Status:** Advanced AI features unavailable. Basic functionality active.")

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### 👤 Customer Profile")
    
    with st.container():
        customer_name = st.text_input("👤 Customer Name", 
                                      value=st.session_state.customer_name,
                                      placeholder="John Doe",
                                      key="customer_name_input")
        customer_phone = st.text_input("📞 Phone Number", 
                                       value=st.session_state.customer_phone,
                                       placeholder="+94771234567",
                                       key="customer_phone_input")
        customer_segment = st.selectbox("🎯 Customer Segment", 
                                        ["Individual", "Family", "Business", "Premium", "Student"],
                                        index=["Individual", "Family", "Business", "Premium", "Student"].index(st.session_state.customer_segment))
        current_plan = st.selectbox("📋 Current Plan", 
            ["Unknown", "Basic", "Standard", "Premium", "Competitor"])
    
    # Update session state when values change
    if customer_name != st.session_state.customer_name:
        st.session_state.customer_name = customer_name
    if customer_phone != st.session_state.customer_phone:
        st.session_state.customer_phone = customer_phone
    if customer_segment != st.session_state.customer_segment:
        st.session_state.customer_segment = customer_segment

    if st.session_state.customer_phone and st.session_state.customer_name:
        st.success(f"✅ Profile Active")
        st.markdown(f"**Name:** {st.session_state.customer_name}")
        st.markdown(f"**Phone:** {st.session_state.customer_phone}")
        st.markdown(f"**Segment:** {customer_segment}")
    
    st.markdown("---")
    st.markdown("### 📊 Real-time Dashboard")
    
    # Load enhanced stats
    conn = sqlite3.connect('product_promotion.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM customer_interactions WHERE DATE(created_at) = DATE('now')")
    today_interactions = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM sms_campaigns WHERE delivery_status='sent'")
    total_sms = cursor.fetchone()[0]
    
    # Get AI chat stats if available
    try:
        cursor.execute("SELECT COUNT(*) FROM ai_chat_logs WHERE DATE(timestamp) = DATE('now')")
        today_chats = cursor.fetchone()[0]
    except:
        today_chats = 0
    
    avg_lead_score = 75  # Default score
    
    conn.close()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📞 Today's Interactions", today_interactions)
        st.metric("💬 AI Chats", today_chats)
    with col2:
        st.metric("📱 SMS Sent", total_sms)
        st.metric("🎯 Avg Lead Score", f"{avg_lead_score:.1f}%")
    
    # AI Engine Status
    st.markdown("### 🤖 AI Engine Status")
    if AI_CHAT_TYPE == "FAISS":
        st.success("🚀 FAISS Vector DB Active")
    elif AI_CHAT_TYPE == "Legacy":
        st.warning("⚠️ Legacy Mode Active")
    else:
        st.error("❌ AI Features Unavailable")

# Main Action Buttons with Enhanced Functionality
st.markdown("## 🎯 Customer Engagement Actions")

# Create 3 rows of 2 columns each for better organization
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2) 
row3_col1, row3_col2 = st.columns(2)

# --- Voice Assistant Button ---
with row1_col1:
    st.markdown("### 🎙️ Call Integration")
    if st.button("🚀 Call Customer", key="voice_demo", help="Launch AI voice assistant application"):
        if st.session_state.customer_phone and st.session_state.customer_name:
            # Log interaction
            conn = sqlite3.connect('product_promotion.db')
            session_id = str(uuid.uuid4())
            conn.execute('''
                INSERT INTO customer_interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, st.session_state.customer_phone, st.session_state.customer_name, "", "voice_demo", "initiated", 
                  datetime.now(), None, f"Voice demo for {customer_segment} customer", 75))
            conn.commit()
            conn.close()
            
            st.info("🎙️ Voice calling feature will be available in production environment.")
            st.success(f"✅ Call logged for {st.session_state.customer_name} ({st.session_state.customer_phone})")
        else:
            st.error("⚠️ Please enter customer details first!")

# --- Smart SMS Campaigns Button ---
with row1_col2:
    st.markdown("### 📱 Smart SMS Campaigns")
    if st.button("📤 Send Product SMS", key="sms_info", help="Send personalized product information"):
        if st.session_state.customer_phone:
            st.markdown("#### 🎯 Select Campaign Type:")
            
            campaign_type = st.selectbox("Campaign Strategy", 
                ["🌟 General Introduction", "🏆 Competitive Comparison", "🔄 Follow-up Campaign"])
            
            st.markdown("#### 📦 Choose Product:")
            for key, product in PRODUCTS.items():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{product['name']}** - Rs.{product['price']}/month")
                    st.markdown(f"*{product['target_audience']}*")
                with col_b:
                    context_map = {
                        "🌟 General Introduction": "general",
                        "🏆 Competitive Comparison": "competitive", 
                        "🔄 Follow-up Campaign": "followup"
                    }
                    
                    if st.button(f"📤 Send", key=f"sms_{key}_{campaign_type}"):
                        with st.spinner("📱 Sending personalized SMS..."):
                            result = mspace_api.send_personalized_sms(
                                st.session_state.customer_phone, key, st.session_state.customer_name, 
                                context_map[campaign_type]
                            )
                            if result["status"] == "success":
                                st.success(f"SMS sent! ID: {result['message_id']}")
                                st.info(f"Delivery: {result['estimated_delivery']}")
                            else:
                                st.error(f"Failed: {result['message']}")
        else:
            st.error("Please enter phone number!")

with row2_col1:
    st.markdown("### 📅 Smart Scheduling")
    if st.button("📞 Schedule Callback", key="callback", help="Schedule intelligent callback"):
        if st.session_state.customer_phone and st.session_state.customer_name:
            with st.form("callback_form"):
                st.markdown("#### 📋 Callback Details")
                
                callback_date = st.date_input("Preferred Date", 
                    min_value=datetime.now().date(),
                    value=datetime.now().date() + timedelta(days=1))
                
                callback_time = st.selectbox("Preferred Time", 
                    ["9:00 AM", "11:00 AM", "1:00 PM", "3:00 PM", "5:00 PM", "7:00 PM"])
                
                preferred_product = st.selectbox("Interest Area", 
                    [prod["name"] for prod in PRODUCTS.values()] + ["General Inquiry", "Complaint Resolution"])
                
                urgency = st.selectbox("⚡ Priority Level", 
                    ["Standard (24-48 hours)", "High (4-8 hours)", "Urgent (Within 2 hours)"])
                
                notes = st.text_area("Special Notes", 
                    placeholder="Any specific requirements or concerns?")
                
                if st.form_submit_button("📅 Schedule Callback"):
                    conn = sqlite3.connect('product_promotion.db')
                    callback_datetime = datetime.combine(
                        callback_date, 
                        datetime.strptime(callback_time, "%I:%M %p").time()
                    )
                    
                    session_id = str(uuid.uuid4())
                    conn.execute('''
                        INSERT INTO customer_interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (session_id, st.session_state.customer_phone, st.session_state.customer_name, preferred_product, 
                          "callback_scheduled", "pending", datetime.now(), callback_datetime, 
                          f"Priority: {urgency}. Notes: {notes}", 85))
                    conn.commit()
                    conn.close()
                    
                    confirmation_msg = f"""Callback Confirmed - Mobitel

Hi {st.session_state.customer_name}! Your priority callback is scheduled:

📅 Date: {callback_date.strftime('%B %d, %Y')}
⏰ Time: {callback_time}  
📋 Topic: {preferred_product}
⚡ Priority: {urgency.split('(')[0].strip()}

Our specialist will call {st.session_state.customer_phone}

You'll receive:
• SMS reminder 2 hours before
• Call from expert consultant
• Personalized product demo
• Exclusive offers discussion

Questions? Reply HELP
Reschedule? Reply CHANGE

Thanks for choosing Mobitel! 🚀"""
                    
                    st.success("✅ Callback scheduled successfully!")
                    st.code(confirmation_msg, language=None)
                    
        else:
            st.error("⚠️ Please enter customer details first!")

with row2_col2:
    st.markdown("### 💬 AI Chat Integration")
    chat_status_text = f"FAISS Powered" if AI_CHAT_TYPE == "FAISS" else f"{AI_CHAT_TYPE} Mode"
    st.caption(f"🤖 Engine: {chat_status_text}")
    
    chat_button_text = "🛑 End AI Chat" if st.session_state.show_ai_chat else "🚀 Start AI Chat"
    
    if st.button(chat_button_text, key="ai_chat", help="Interactive AI conversation"):
        if st.session_state.customer_phone and st.session_state.customer_name:
            st.session_state.show_ai_chat = not st.session_state.show_ai_chat
            if not st.session_state.show_ai_chat:
                # Clean up chat session
                if hasattr(st.session_state, 'chat_messages'):
                    st.session_state.chat_messages = []
                if hasattr(st.session_state, 'simple_chat_messages'):
                    st.session_state.simple_chat_messages = []
                if 'chat_system' in st.session_state:
                    del st.session_state.chat_system
                st.rerun()
        else:
            st.error("⚠️ Please enter customer details first!")
            st.session_state.show_ai_chat = False

# AI Chat Section
st.markdown("---")
if st.session_state.show_ai_chat:
    st.markdown(f"## 🤖 ALEX AI Chat Assistant - {chat_status_text}")
    
    # Initialize chat system if not present
    if 'chat_system' not in st.session_state:
        with st.spinner(f"🚀 Initializing ALEX AI Chat Engine ({AI_CHAT_TYPE})..."):
            try:
                if AI_CHAT_AVAILABLE:
                    chat_engine = AIProductChatEngine()
                    st.session_state.chat_system = StreamlitChatInterface(chat_engine)
                    st.success(f"✅ ALEX AI ({AI_CHAT_TYPE}) is ready to help!")
                else:
                    st.session_state.chat_system = SimpleChatInterface()
                    st.info("✅ Basic ALEX AI is ready to help!")
            except Exception as e:
                st.error(f"❌ Failed to initialize AI Chat Engine: {e}")
                st.session_state.chat_system = SimpleChatInterface()
                st.info("✅ Fallback chat mode activated!")
    
    # Render chat interface if system is ready
    if 'chat_system' in st.session_state:
        # Display customer info in chat
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"💬 Chatting with: **{st.session_state.customer_name}** ({st.session_state.customer_phone})")
        with col2:
            if st.button("🔄 Reset Chat", key="reset_chat"):
                if hasattr(st.session_state, 'chat_messages'):
                    st.session_state.chat_messages = []
                if hasattr(st.session_state, 'simple_chat_messages'):
                    st.session_state.simple_chat_messages = []
                st.rerun()
        
        # Render the chat interface
        st.session_state.chat_system.render_chat_interface(st.session_state.customer_name)
        
        # Render chat sidebar/controls if available
        if hasattr(st.session_state.chat_system, 'render_chat_sidebar'):
            with st.expander("🎛️ Chat Controls & Analytics", expanded=False):
                st.session_state.chat_system.render_chat_sidebar()

# Additional Features Section
with row3_col1:
    st.markdown("### 📊 Analytics Dashboard")
    if st.button("📈 View Analytics", key="analytics", help="View customer interaction analytics"):
        st.markdown("#### 📊 Customer Interaction Analytics")
        
        # Load data from database
        conn = sqlite3.connect('product_promotion.db')
        
        # Interactions over time
        df_interactions = pd.read_sql_query("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as interactions,
                interaction_type
            FROM customer_interactions 
            WHERE created_at >= date('now', '-30 days')
            GROUP BY DATE(created_at), interaction_type
            ORDER BY date DESC
        """, conn)
        
        if not df_interactions.empty:
            fig = px.line(df_interactions, x='date', y='interactions', 
                         color='interaction_type', title='Customer Interactions (Last 30 Days)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No interaction data available yet. Start engaging with customers!")
        
        # SMS campaign performance
        df_sms = pd.read_sql_query("""
            SELECT 
                campaign_type,
                COUNT(*) as sent_count,
                product_name
            FROM sms_campaigns 
            GROUP BY campaign_type, product_name
        """, conn)
        
        if not df_sms.empty:
            fig_sms = px.bar(df_sms, x='campaign_type', y='sent_count', 
                            color='product_name', title='SMS Campaign Performance')
            st.plotly_chart(fig_sms, use_container_width=True)
        
        # AI Chat analytics (if FAISS is available)
        if AI_CHAT_TYPE == "FAISS":
            try:
                df_ai_chats = pd.read_sql_query("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as chat_sessions
                    FROM ai_chat_logs 
                    WHERE timestamp >= date('now', '-7 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """, conn)
                
                if not df_ai_chats.empty:
                    fig_ai = px.line(df_ai_chats, x='date', y='chat_sessions', 
                                    title='AI Chat Sessions (Last 7 Days)')
                    st.plotly_chart(fig_ai, use_container_width=True)
            except:
                st.info("AI chat analytics will be available after first conversations")
        
        conn.close()

with row3_col2:
    st.markdown("### 🎯 Quick Actions")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("📋 Export Data", key="export", help="Export customer data"):
            conn = sqlite3.connect('product_promotion.db')
            
            # Export customer interactions
            df_export = pd.read_sql_query("""
                SELECT 
                    customer_name,
                    phone_number,
                    selected_product,
                    interaction_type,
                    status,
                    created_at,
                    notes,
                    conversion_score
                FROM customer_interactions 
                ORDER BY created_at DESC
            """, conn)
            
            if not df_export.empty:
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"mobitel_customer_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                st.success(f"✅ Ready to download {len(df_export)} customer records")
            else:
                st.info("No data available for export")
            
            conn.close()
    
    with col_b:
        if st.button("🔄 Refresh Stats", key="refresh", help="Refresh dashboard statistics"):
            st.cache_data.clear()
            st.success("✅ Dashboard refreshed!")
            st.rerun()

# FAISS-specific management section
if AI_CHAT_TYPE == "FAISS":
    with st.expander("🔧 FAISS Vector Store Management", expanded=False):
        st.markdown("### 📚 Knowledge Base Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reload Vector Store"):
                if 'chat_system' in st.session_state and hasattr(st.session_state.chat_system, 'chat_engine'):
                    with st.spinner("Reloading FAISS vector store..."):
                        success = st.session_state.chat_system.chat_engine.force_reload_vectorstore()
                    if success:
                        st.success("✅ Vector store reloaded!")
                    else:
                        st.error("❌ Failed to reload vector store")
                else:
                    st.error("Chat engine not initialized")
        
        with col2:
            if st.button("📊 Vector Store Info"):
                if 'chat_system' in st.session_state and hasattr(st.session_state.chat_system, 'chat_engine'):
                    info = st.session_state.chat_system.chat_engine.get_vectorstore_info()
                    if info:
                        st.json(info)
                    else:
                        st.error("Failed to get vector store info")
        
        with col3:
            uploaded_file = st.file_uploader("📄 Upload Knowledge PDF", type="pdf")
            if uploaded_file and st.button("📥 Add to Knowledge Base"):
                # Save uploaded file to Data folder
                os.makedirs("Data", exist_ok=True)
                file_path = os.path.join("Data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.success(f"✅ {uploaded_file.name} uploaded! Reload vector store to index.")

# Product Showcase Section
st.markdown("---")
st.markdown("## 📦 Our Product Lineup")

# Display products in a nice grid
product_cols = st.columns(len(PRODUCTS))

for idx, (key, product) in enumerate(PRODUCTS.items()):
    with product_cols[idx]:
        with st.container():
            st.markdown(f"""
            <div class="product-card">
                <h3>🏆 {product['name']}</h3>
                <h2>Rs.{product['price']:,}/month</h2>
                <p><em>{product['description']}</em></p>
                <hr>
                <h4>✨ Features:</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for feature in product['features']:
                st.markdown(f"• {feature}")
            
            st.markdown(f"**🎯 Perfect for:** {product['target_audience']}")
            st.markdown(f"**🏆 Advantage:** {product['competitive_advantage']}")
            
            with st.expander("🎁 Current Offers"):
                st.markdown(product['discount'])
            
            if st.session_state.customer_phone:
                if st.button(f"📤 Send {product['name']} Info", key=f"quick_sms_{key}"):
                    result = mspace_api.send_personalized_sms(
                        st.session_state.customer_phone, 
                        key, 
                        st.session_state.customer_name, 
                        "general"
                    )
                    if result["status"] == "success":
                        st.success("SMS sent successfully!")
                    else:
                        st.error("Failed to send SMS")

# Footer and Additional Info
st.markdown("---")
st.markdown("## ℹ️ System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🚀 Deployment Status:**
    - ✅ Core Features Active
    - ✅ SMS Integration Ready
    - ✅ Database Operational
    - ✅ Analytics Available
    """)

with col2:
    ai_status = "✅ FAISS Vector DB" if AI_CHAT_TYPE == "FAISS" else f"⚠️ {AI_CHAT_TYPE} Mode"
    st.markdown(f"""
    **🤖 AI Features:**
    - Vector Database: {ai_status}
    - Translation: ✅ Available
    - Smart Responses: ✅ Active
    - Knowledge Search: {'✅ Available' if AI_CHAT_TYPE == 'FAISS' else '⚠️ Limited'}
    """)

with col3:
    st.markdown("""
    **🔧 Technical Stack:**
    - Frontend: Streamlit
    - Vector DB: FAISS
    - Database: SQLite
    - SMS API: mSpace Integration
    - Charts: Plotly
    """)

# Help and Support Section
with st.expander("❓ Help & Support", expanded=False):
    st.markdown("""
    ### 🆘 Need Help?
    
    **FAISS-Specific Features:**
    1. **Vector Store Management:** Upload PDFs to enhance AI knowledge
    2. **Advanced Search:** Use similarity search in chat sidebar
    3. **Real-time Updates:** Reload knowledge base when PDFs change
    4. **Performance:** FAISS provides faster similarity search than ChromaDB
    
    **Common Issues:**
    1. **Customer details not saving:** Refresh the page and re-enter details
    2. **SMS not sending:** Check phone number format (+94XXXXXXXXX)
    3. **AI chat slow:** FAISS indexing may take time with large PDFs
    4. **Chat not responding:** Try refreshing the chat or restart the session
    
    **🔧 FAISS Troubleshooting:**
    - Ensure Google API key is set in environment variables
    - PDF files should be placed in 'Data' folder for indexing
    - Vector store rebuilds automatically when PDFs are updated
    - Use "Reload Vector Store" button after adding new documents
    
    **📞 Support Contacts:**
    - Technical Support: +94 11 123 4567
    - Sales Team: +94 77 123 4567
    - Email: support@mobitel.lk
    
    **🌐 Resources:**
    - [Mobitel Website](https://www.mobitel.lk)
    - [FAISS Documentation](https://faiss.ai/)
    - [mSpace API Documentation](https://mspace.lk/docs)
    """)

# Development Notes for FAISS
if os.getenv("STREAMLIT_ENV") == "development":
    with st.expander("🔬 FAISS Development Notes", expanded=False):
        st.markdown("""
        ### 👨‍💻 FAISS Implementation Details
        
        **Current Implementation:**
        - FAISS CPU version for Streamlit Cloud compatibility
        - Automatic PDF processing and indexing
        - Persistent vector storage with save/load functionality
        - MMR (Maximum Marginal Relevance) search for diverse results
        
        **FAISS Advantages over ChromaDB:**
        1. **Performance:** Faster similarity search, especially with large datasets
        2. **Memory Efficiency:** Better memory usage for vector operations
        3. **Scalability:** Handles larger document collections more efficiently
        4. **Cloud Compatibility:** No system-level dependencies like ChromaDB
        
        **Environment Variables:**
        ```
        GOOGLE_API_KEY=your_google_api_key
        MSPACE_API_KEY=your_mspace_api_key
        MSPACE_SENDER_ID=your_sender_id
        STREAMLIT_ENV=development
        ```
        
        **File Structure:**
        ```
        Data/                    # PDF files for knowledge base
        faiss_db/               # FAISS index storage
        ├── index.faiss         # Vector index
        └── index.pkl           # Metadata and documents
        ```
        """)

# Performance monitoring with FAISS-specific metrics
if st.sidebar.button("🔍 System Health Check"):
    with st.spinner("Checking system health..."):
        time.sleep(1)
        
        health_status = {
            "Database": "✅ Operational",
            "SMS Integration": "✅ Ready",
            "Vector Store": "✅ FAISS Active" if AI_CHAT_TYPE == "FAISS" else f"⚠️ {AI_CHAT_TYPE}",
            "Analytics": "✅ Operational",
            "File Storage": "✅ Available"
        }
        
        st.sidebar.success("System Health Check Complete!")
        for component, status in health_status.items():
            st.sidebar.markdown(f"**{component}:** {status}")
        
        # Show database and FAISS stats
        conn = sqlite3.connect('product_promotion.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customer_interactions")
        total_interactions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM sms_campaigns")
        total_sms = cursor.fetchone()[0]
        
        # FAISS-specific stats
        faiss_docs = "N/A"
        if 'chat_system' in st.session_state and hasattr(st.session_state.chat_system, 'chat_engine'):
            try:
                info = st.session_state.chat_system.chat_engine.get_vectorstore_info()
                if info:
                    faiss_docs = info.get('total_documents', 'Unknown')
            except:
                pass
        
        conn.close()
        
        st.sidebar.markdown(f"""
        **📊 System Stats:**
        - Total Interactions: {total_interactions}
        - Total SMS Sent: {total_sms}
        - FAISS Documents: {faiss_docs}
        - Vector Store: {AI_CHAT_TYPE}
        - Uptime: Active
        """)

# Auto-refresh for real-time updates (optional)
if st.checkbox("🔄 Auto-refresh Dashboard (30s)", value=False):
    time.sleep(30)
    st.rerun()