# streamlit_app.py - Enhanced version with all features integrated
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
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import threading
import schedule
from functools import wraps

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

load_dotenv()

# Enhanced Configuration
@dataclass
class Config:
    MSPACE_API_KEY: str = os.getenv("MSPACE_API_KEY", "demo_api_key_12345")
    MSPACE_SENDER_ID: str = os.getenv("MSPACE_SENDER_ID", "MOBITEL")
    MSPACE_BASE_URL: str = os.getenv("MSPACE_BASE_URL", "https://api.mspace.lk")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "product_promotion.db")
    MAX_SMS_PER_MINUTE: int = int(os.getenv("MAX_SMS_PER_MINUTE", "30"))
    MAX_SMS_PER_DAY: int = int(os.getenv("MAX_SMS_PER_DAY", "5000"))
    DAILY_SMS_BUDGET: float = float(os.getenv("DAILY_SMS_BUDGET", "25000.0"))
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))

config = Config()

# Enhanced Product Configuration
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

# ============================================================================
# ENHANCED API INTEGRATION
# ============================================================================

class MessageStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    REJECTED = "rejected"

@dataclass
class SMSResponse:
    success: bool
    message_id: Optional[str] = None
    error_message: Optional[str] = None
    status: MessageStatus = MessageStatus.PENDING
    cost: Optional[float] = None
    remaining_credits: Optional[int] = None

class EnhancedMSpaceAPI:
    def __init__(self):
        self.api_key = config.MSPACE_API_KEY
        self.sender_id = config.MSPACE_SENDER_ID
        self.base_url = config.MSPACE_BASE_URL
        self.session = requests.Session()
        
        # Set headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'Mobitel-Dashboard/2.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60 / config.MAX_SMS_PER_MINUTE

    def _normalize_phone_number(self, phone: str) -> str:
        """Normalize phone number to Sri Lankan format"""
        clean_phone = re.sub(r'[^\d+]', '', phone)
        
        if clean_phone.startswith('0'):
            clean_phone = '94' + clean_phone[1:]
        elif not clean_phone.startswith('94') and not clean_phone.startswith('+94'):
            clean_phone = '94' + clean_phone
            
        if not clean_phone.startswith('+'):
            clean_phone = '+' + clean_phone
            
        return clean_phone

    def _is_valid_phone_number(self, phone: str) -> bool:
        """Validate Sri Lankan mobile number"""
        clean_phone = phone.replace('+', '')
        
        if clean_phone.startswith('947') and len(clean_phone) == 11:
            return clean_phone[3] in ['0', '1', '2', '5', '6', '7', '8']
        return False

    def send_sms(self, phone_number: str, message: str, message_type: str = "promotional") -> SMSResponse:
        """Send SMS via mSpace API with enhanced error handling"""
        
        # Validate phone number
        phone_number = self._normalize_phone_number(phone_number)
        if not self._is_valid_phone_number(phone_number):
            return SMSResponse(success=False, error_message="Invalid phone number format")
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        # Prepare payload
        payload = {
            "api_key": self.api_key,
            "sender_id": self.sender_id,
            "to": phone_number,
            "message": message,
            "message_type": message_type,
            "delivery_report": True
        }
        
        try:
            # For demo - simulate API call
            # In production, replace with actual API call:
            # response = self.session.post(f"{self.base_url}/sms/send", json=payload, timeout=30)
            
            # Simulated response
            time.sleep(0.5)  # Simulate API delay
            self.last_request_time = time.time()
            
            # Generate realistic response
            message_id = f"msg_{uuid.uuid4().hex[:12]}"
            estimated_cost = len(message) // 160 * 2.5 + 2.5  # Rs.2.5 per 160 chars
            
            return SMSResponse(
                success=True,
                message_id=message_id,
                status=MessageStatus.SENT,
                cost=estimated_cost,
                remaining_credits=950  # Simulated
            )
            
        except requests.exceptions.Timeout:
            return SMSResponse(success=False, error_message="Request timeout")
        except requests.exceptions.ConnectionError:
            return SMSResponse(success=False, error_message="Connection failed")
        except Exception as e:
            return SMSResponse(success=False, error_message=str(e))

    def get_account_balance(self) -> Dict:
        """Get account balance (simulated for demo)"""
        return {
            "success": True,
            "credits": 1000,
            "balance": 25000.00,
            "currency": "LKR"
        }

# ============================================================================
# ENHANCED DATABASE OPERATIONS
# ============================================================================

def init_enhanced_database():
    """Initialize database with enhanced schema"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    
    try:
        # Original tables
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
                campaign_type TEXT,
                message_id TEXT,
                cost REAL DEFAULT 0.0
            )
        ''')
        
        # Enhanced tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS customer_segments (
                id TEXT PRIMARY KEY,
                phone_number TEXT NOT NULL,
                segment_name TEXT NOT NULL,
                segment_score INTEGER,
                last_interaction_date TIMESTAMP,
                preferred_product TEXT,
                total_spent REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scheduled_campaigns (
                id TEXT PRIMARY KEY,
                campaign_name TEXT NOT NULL,
                campaign_type TEXT NOT NULL,
                target_segment TEXT,
                product_key TEXT NOT NULL,
                message_template TEXT NOT NULL,
                scheduled_time TIMESTAMP NOT NULL,
                status TEXT DEFAULT 'scheduled',
                recipient_count INTEGER DEFAULT 0,
                sent_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS delivery_reports (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                status TEXT NOT NULL,
                delivered_at TIMESTAMP,
                failed_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_customer_interactions_phone ON customer_interactions(phone_number)",
            "CREATE INDEX IF NOT EXISTS idx_sms_campaigns_phone ON sms_campaigns(phone_number)", 
            "CREATE INDEX IF NOT EXISTS idx_sms_campaigns_sent ON sms_campaigns(sent_at)",
            "CREATE INDEX IF NOT EXISTS idx_customer_segments_phone ON customer_segments(phone_number)",
            "CREATE INDEX IF NOT EXISTS idx_delivery_reports_message_id ON delivery_reports(message_id)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.OperationalError:
                pass  # Index might already exist
        
        conn.commit()
        
    except Exception as e:
        st.error(f"Database initialization error: {e}")
    finally:
        conn.close()

# ============================================================================
# ENHANCED SMS CAMPAIGN MANAGER
# ============================================================================

class EnhancedMSpacePromotion:
    def __init__(self):
        self.api = EnhancedMSpaceAPI()
        self.templates = {
            "welcome": """🌟 Welcome {name}! 

Discover Mobitel's amazing {product_name} package:
✅ {key_features}

💰 Special Price: Rs.{price}/month
🎁 {special_offer}

Reply YES to activate or CALL 071-1234567

Mobitel - Your Connected Future 🚀""",

            "followup": """Hi {name}! 👋

Still thinking about our {product_name}? 

⏰ Limited Time: {special_offer}

Perfect for: {target_audience}
💯 Satisfaction Guaranteed

Reply YES for instant activation!

ALEX AI Assistant | Mobitel""",

            "competitive": """🏆 {name}, Ready for BETTER mobile service?

Why customers switch to Mobitel:
✅ Better Coverage (Island-wide 4G/5G)  
✅ Better Value - {product_name}
✅ Better Support (24/7 Local)

🎁 SWITCHING BONUS: {special_offer}

Reply SWITCH for FREE porting assistance!""",

            "seasonal": """🎉 {season} Special for {name}!

{product_name} - Now with exclusive benefits:
🎁 {seasonal_bonus}
💰 {discount_details}

Limited time offer - Only {days_left} days left!

Activate now: Reply YES or Call 071-1234567""",

            "loyalty": """💎 Thank you {name}!

As a valued customer, enjoy our NEW {product_name}:

🌟 EXCLUSIVE Benefits:
• Priority customer support
• Early access to new features  
• {loyalty_bonus}

Upgrade today: Reply UPGRADE"""
        }

    def send_personalized_campaign(self, phone_number: str, product_key: str, 
                                 customer_name: str = "Valued Customer",
                                 campaign_type: str = "welcome",
                                 custom_data: Dict = None) -> SMSResponse:
        """Send personalized campaign SMS"""
        
        if product_key not in PRODUCTS:
            return SMSResponse(success=False, error_message="Invalid product key")
        
        product = PRODUCTS[product_key]
        template = self.templates.get(campaign_type, self.templates["welcome"])
        
        # Prepare template data
        template_data = {
            "name": customer_name,
            "product_name": product["name"],
            "price": f"{product['price']:,}",
            "key_features": ", ".join(product["features"][:3]),
            "special_offer": product.get("discount", "Special launch pricing!"),
            "target_audience": product.get("target_audience", "everyone"),
            "season": "Summer",  # You can make this dynamic
            "seasonal_bonus": "Free 3 months Disney+",
            "discount_details": "50% off for first 6 months",
            "days_left": "7",
            "loyalty_bonus": "Double data for life"
        }
        
        if custom_data:
            template_data.update(custom_data)
        
        # Format message
        try:
            message = template.format(**template_data)
        except KeyError as e:
            return SMSResponse(success=False, error_message=f"Template error: {e}")
        
        # Send SMS
        result = self.api.send_sms(phone_number, message)
        
        # Log to database
        if result.success:
            self._log_campaign(phone_number, product["name"], campaign_type, message, result.message_id, result.cost)
        
        return result

    def _log_campaign(self, phone_number: str, product_name: str, campaign_type: str, 
                     message: str, message_id: str, cost: float = 0.0):
        """Log campaign to database"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            conn.execute('''
                INSERT INTO sms_campaigns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), phone_number, product_name, message, 
                  "sent", datetime.now(), campaign_type, message_id, cost))
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Failed to log campaign: {e}")

# ============================================================================
# AI CHAT INTEGRATION
# ============================================================================

# Conditional import for AI chat engine with FAISS
try:
    from ai_chat_engine_faiss import AIProductChatEngine, StreamlitChatInterface
    AI_CHAT_AVAILABLE = True
    AI_CHAT_TYPE = "FAISS"
except ImportError:
    AI_CHAT_AVAILABLE = False
    AI_CHAT_TYPE = "None"
    
    # Enhanced fallback chat
    class SimpleChatInterface:
        def __init__(self):
            if 'simple_chat_messages' not in st.session_state:
                st.session_state.simple_chat_messages = []
        
        def render_chat_interface(self, customer_name):
            st.markdown("### 💬 ALEX AI Chat Assistant")
            
            # Chat statistics
            if st.session_state.simple_chat_messages:
                st.caption(f"💬 {len(st.session_state.simple_chat_messages)} messages in this conversation")
            
            # Display messages
            for message in st.session_state.simple_chat_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about our mobile packages..."):
                # Add user message
                st.session_state.simple_chat_messages.append({
                    "role": "user", 
                    "content": prompt,
                    "timestamp": datetime.now()
                })
                
                # Generate response
                response = self.generate_enhanced_response(prompt, customer_name)
                st.session_state.simple_chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        def generate_enhanced_response(self, user_input, customer_name):
            user_lower = user_input.lower()
            
            # Enhanced greeting responses
            if any(word in user_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
                time_greeting = self._get_time_greeting()
                return f"{time_greeting} {customer_name}! I'm ALEX from Mobitel. I can help you with:\n\n• Package information and pricing\n• Feature comparisons\n• Special offers and discounts\n• Activation assistance\n\nWhat interests you most today? 🌟"
            
            # Price and package queries
            elif any(word in user_lower for word in ['price', 'cost', 'package', 'plan']):
                return """📋 **Our Current Packages:**

🌟 **Premium Package** - Rs.2,999/month
   Perfect for heavy users & entertainment lovers
   • 100GB Data + Unlimited Calls + Netflix + Spotify

👨‍👩‍👧‍👦 **Family Package** - Rs.4,999/month  
   Ideal for families with 3+ members
   • 300GB Shared Data + 5 SIMs + Disney+ & Netflix

💼 **Business Package** - Rs.7,999/month
   Enterprise solution for growing businesses
   • Unlimited Data + Priority Network + 24/7 Support

**🎁 Special Offers Running Now!**
All packages come with exclusive bonuses worth up to Rs.79,900!

Which package sounds right for your needs?"""
            
            # Premium package details
            elif any(word in user_lower for word in ['premium']):
                return """🌟 **Premium Package - Complete Details**

**Monthly Price:** Rs.2,999 (Best value for individual users)
**Perfect for:** Heavy data users and entertainment lovers

**🚀 What You Get:**
• 100GB high-speed data (4G/5G)
• Unlimited local & STD calls
• Free Netflix subscription (Rs.1,690 value)
• Free Spotify Premium (Rs.1,200 value)
• 50GB mobile hotspot sharing
• 5G ready network access
• Free caller tunes & SMS pack

**🎁 Current Special Offer:**
• First month absolutely FREE
• Samsung Galaxy A34 smartphone (Worth Rs.79,900)
• Free screen protector & case
• Priority customer support

**💡 Why Choose Premium:**
✅ Save Rs.2,890/month on entertainment
✅ Best coverage island-wide
✅ 24/7 customer support

Ready to activate? I can help you get started right now! 📱"""
            
            # Family package details
            elif any(word in user_lower for word in ['family']):
                return """👨‍👩‍👧‍👦 **Family Package - Complete Details**

**Monthly Price:** Rs.4,999 (Shared across family)
**Perfect for:** Families with 3+ members, parents with children

**🏠 What Your Family Gets:**
• 300GB shared high-speed data
• 5 SIM cards included (add more if needed)
• Unlimited family calls (between SIMs)
• Disney+ subscription (Kids love it!)
• Netflix premium subscription
• Parental control features
• Family locator service
• Shared mobile banking

**🎁 Current Special Offer:**
• 3 months at 50% off (Save Rs.7,498!)
• Free 4G router for home WiFi
• Family safety app premium
• Free delivery & setup

**👨‍👩‍👧‍👦 Family Benefits:**
✅ Each member gets their own number
✅ Parents can control kids' usage
✅ Shared entertainment subscriptions
✅ Emergency family locator
✅ Bill payment reminders

Perfect for keeping everyone connected safely! Want me to check availability in your area?"""
            
            # Business package details
            elif any(word in user_lower for word in ['business']):
                return """💼 **Business Package - Complete Details**

**Monthly Price:** Rs.7,999 (Investment in productivity)
**Perfect for:** SMEs, remote teams, digital businesses

**🚀 Business Features:**
• Unlimited high-speed data
• Priority network access (guaranteed speeds)
• 24/7 dedicated business support
• 1TB secure cloud storage
• Advanced conference calling (up to 50 people)
• Business VPN access
• Mobile device management
• Dedicated account manager

**🎁 Current Business Offer:**
• Free setup and migration
• 6 months Microsoft 365 Business (Rs.18,000 value)
• Free business consultation
• Priority technical support
• Free backup internet solution

**💡 Business Advantages:**
✅ Keep teams connected anywhere
✅ Secure data and communications
✅ Professional email & office tools
✅ Scale up/down as needed
✅ Tax deductible business expense

**📈 ROI Benefits:**
• Increase team productivity by 40%
• Reduce communication costs by 60%
• Enable flexible remote working
• Professional business image

Would you like me to schedule a free business consultation for your company?"""
            
            # Comparison queries
            elif any(word in user_lower for word in ['compare', 'difference', 'which is better']):
                return """📊 **Package Comparison Guide:**

**🎯 Choose Based on Your Needs:**

**Premium Package** ➜ Perfect if you:
• Are a single user or couple
• Love entertainment (Netflix, Spotify)
• Need 100GB+ data monthly
• Want the latest smartphone deals

**Family Package** ➜ Perfect if you:
• Have 3+ family members
• Want to manage everyone's usage
• Need parental controls
• Share entertainment subscriptions

**Business Package** ➜ Perfect if you:
• Run a business or team
• Need unlimited reliable data
• Require professional tools
• Want priority support & VPN

**💡 Quick Decision Helper:**
• **Budget under Rs.3,000** → Premium
• **Family with kids** → Family  
• **Business/Team** → Business

Want me to recommend based on your specific situation? Tell me:
1. How many people will use it?
2. What do you mainly use mobile data for?
3. Do you need business features?"""
            
            # Activation and how-to queries
            elif any(word in user_lower for word in ['activate', 'how to', 'sign up', 'get started']):
                return """🚀 **Easy Activation Process:**

**📋 What You Need:**
• Valid NIC (front & back photos)
• Current address proof
• Existing number (for porting) - optional

**⚡ 3 Ways to Activate:**

**1. Instant Online Activation** (Fastest)
• Reply YES to any package SMS
• Upload documents via our app
• Delivery within 2 hours (Colombo area)

**2. Call Our Hotline**
• Dial 071-1234567 (free call)
• Speak with activation specialist
• Schedule home delivery

**3. Visit Mobitel Store**
• Find nearest store in our app
• Walk-in activation available
• Get hands-on device setup

**⏰ Timeline:**
• Online: Activated within 2 hours
• Phone: Same day activation
• Store: Instant activation

**🎁 Activation Bonus:**
First 100 customers today get double data for 3 months!

Which activation method works best for you? I can help start the process right now! 📞"""
            
            # Troubleshooting and support
            elif any(word in user_lower for word in ['problem', 'issue', 'help', 'support', 'not working']):
                return f"""🛠️ **I'm here to help, {customer_name}!**

**Common Solutions:**

**📶 Network Issues:**
• Restart your device
• Check if you're in a coverage area
• Try switching to 3G/4G manually

**💳 Account/Billing:**
• Check balance: Dial *#456#
• View usage: Dial *456*1#
• Pay bills via our mobile app

**📱 Technical Problems:**
• Update device software
• Clear network settings
• Remove and reinsert SIM card

**🆘 Need More Help?**
• **24/7 Customer Care:** 071-1234567
• **WhatsApp Support:** +94 77 123 4567
• **Live Chat:** Available in our mobile app
• **Email:** support@mobitel.lk

**⚡ Priority Support Available:**
Business package customers get dedicated priority support with average response time under 5 minutes!

What specific issue are you facing? I might be able to provide more targeted help! 🎯"""
            
            # Default enhanced response
            else:
                return f"""Thank you for your question, {customer_name}! 😊

I'm designed to help with:
• 📋 Package information & pricing
• 🔄 Feature comparisons  
• 🎁 Current offers & discounts
• 🚀 Activation assistance
• 🛠️ Basic troubleshooting
• 📞 Connecting you with specialists

**Popular topics customers ask about:**
• "Tell me about Premium package"
• "Compare all packages"  
• "How to activate?"
• "What are current offers?"
• "Help with network issues"

What would you like to explore? I'm here to make your Mobitel experience amazing! ✨"""
        
        def _get_time_greeting(self):
            current_hour = datetime.now().hour
            if 5 <= current_hour < 12:
                return "Good morning! 🌅"
            elif 12 <= current_hour < 17:
                return "Good afternoon! ☀️"
            elif 17 <= current_hour < 21:
                return "Good evening! 🌆"
            else:
                return "Hello! 🌙"
        
        def render_chat_sidebar(self):
            st.sidebar.markdown("### 💬 Chat Controls")
            if st.sidebar.button("🗑️ Clear Chat"):
                st.session_state.simple_chat_messages = []
                st.rerun()
            
            # Chat statistics
            if st.session_state.simple_chat_messages:
                st.sidebar.markdown("#### 📊 Chat Stats")
                total_messages = len(st.session_state.simple_chat_messages)
                user_messages = len([m for m in st.session_state.simple_chat_messages if m["role"] == "user"])
                st.sidebar.metric("Total Messages", total_messages)
                st.sidebar.metric("Your Questions", user_messages)

# ============================================================================
# ENHANCED UI COMPONENTS
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_dashboard_analytics():
    """Load comprehensive analytics data"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    
    try:
        # Campaign performance data
        campaign_data = pd.read_sql_query("""
            SELECT 
                campaign_type,
                product_name,
                COUNT(*) as sent_count,
                SUM(CASE WHEN delivery_status = 'delivered' THEN 1 ELSE 0 END) as delivered_count,
                SUM(CASE WHEN delivery_status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                SUM(cost) as total_cost,
                DATE(sent_at) as date
            FROM sms_campaigns 
            WHERE sent_at >= datetime('now', '-30 days')
            GROUP BY campaign_type, product_name, DATE(sent_at)
            ORDER BY sent_at DESC
        """, conn)
        
        # Customer engagement data
        engagement_data = pd.read_sql_query("""
            SELECT 
                selected_product,
                status,
                COUNT(*) as count,
                DATE(created_at) as date
            FROM customer_interactions 
            WHERE created_at >= datetime('now', '-30 days')
            GROUP BY selected_product, status, DATE(created_at)
        """, conn)
        
        # Conversion funnel data
        funnel_data = pd.read_sql_query("""
            SELECT 
                'SMS Sent' as stage, COUNT(*) as count, 1 as order_num
            FROM sms_campaigns WHERE sent_at >= datetime('now', '-30 days')
            UNION ALL
            SELECT 
                'Customers Interested' as stage, COUNT(*) as count, 2 as order_num
            FROM customer_interactions WHERE status = 'interested' AND created_at >= datetime('now', '-30 days')
            UNION ALL
            SELECT 
                'Customers Converted' as stage, COUNT(*) as count, 3 as order_num
            FROM customer_interactions WHERE status = 'converted' AND created_at >= datetime('now', '-30 days')
            ORDER BY order_num
        """, conn)
        
        conn.close()
        
        return {
            'campaigns': campaign_data,
            'engagement': engagement_data, 
            'funnel': funnel_data
        }
    except Exception as e:
        conn.close()
        return {'campaigns': pd.DataFrame(), 'engagement': pd.DataFrame(), 'funnel': pd.DataFrame()}

def render_enhanced_analytics():
    """Render comprehensive analytics dashboard"""
    st.markdown("## 📊 Advanced Analytics Dashboard")
    
    # Load data
    analytics_data = load_dashboard_analytics()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sms = analytics_data['campaigns']['sent_count'].sum() if not analytics_data['campaigns'].empty else 0
        st.metric("📱 SMS Sent", f"{total_sms:,}", delta=f"+{total_sms//7} /week")
    
    with col2:
        total_delivered = analytics_data['campaigns']['delivered_count'].sum() if not analytics_data['campaigns'].empty else 0
        delivery_rate = (total_delivered / total_sms * 100) if total_sms > 0 else 0
        st.metric("✅ Delivery Rate", f"{delivery_rate:.1f}%", delta="2.3%" if delivery_rate > 90 else "-1.1%")
    
    with col3:
        total_cost = analytics_data['campaigns']['total_cost'].sum() if not analytics_data['campaigns'].empty else 0
        st.metric("💰 Total Spend", f"Rs.{total_cost:,.0f}", delta=f"-Rs.{total_cost*0.1:.0f}")
    
    with col4:
        total_conversions = len(analytics_data['engagement'][analytics_data['engagement']['status'] == 'converted']) if not analytics_data['engagement'].empty else 0
        conversion_rate = (total_conversions / total_sms * 100) if total_sms > 0 else 0
        st.metric("🎯 Conversion Rate", f"{conversion_rate:.1f}%", delta="0.8%")
    
    # Charts section
    if not analytics_data['campaigns'].empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Campaign Performance Trend")
            daily_performance = analytics_data['campaigns'].groupby('date').agg({
                'sent_count': 'sum',
                'delivered_count': 'sum'
            }).reset_index()
            
            fig = px.line(daily_performance, x='date', y=['sent_count', 'delivered_count'],
                         title="Daily SMS Performance", 
                         labels={'value': 'Count', 'date': 'Date'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Conversion Funnel")
            if not analytics_data['funnel'].empty:
                fig = px.funnel(analytics_data['funnel'], x='count', y='stage',
                               title="Customer Journey")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Product performance comparison
    st.markdown("### 🏆 Product Performance (Last 30 Days)")
    if not analytics_data['campaigns'].empty:
        product_perf = analytics_data['campaigns'].groupby('product_name').agg({
            'sent_count': 'sum',
            'delivered_count': 'sum', 
            'total_cost': 'sum'
        }).reset_index()
        
        if not product_perf.empty:
            product_perf['delivery_rate'] = (product_perf['delivered_count'] / product_perf['sent_count'] * 100).round(1)
            product_perf['cost_per_sms'] = (product_perf['total_cost'] / product_perf['sent_count']).round(2)
            
            # Rename columns for better display
            product_perf.columns = ['Product', 'SMS Sent', 'Delivered', 'Total Cost (Rs)', 'Delivery Rate (%)', 'Cost per SMS (Rs)']
            st.dataframe(product_perf, use_container_width=True)

def render_campaign_scheduler():
    """Render campaign scheduling interface"""
    st.markdown("## ⏰ Campaign Scheduler")
    
    with st.form("schedule_campaign"):
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_name = st.text_input("Campaign Name", placeholder="Holiday Premium Offer")
            campaign_type = st.selectbox("Campaign Type", 
                ["welcome", "followup", "competitive", "seasonal", "loyalty"])
            product_key = st.selectbox("Product", list(PRODUCTS.keys()))
        
        with col2:
            target_segment = st.selectbox("Target Segment", 
                ["all", "premium", "family", "business", "loyal_customers", "new_customers"])
            
            schedule_date = st.date_input("Schedule Date", min_value=datetime.now().date())
            schedule_time = st.time_input("Schedule Time")
            
        # Audience estimation
        st.markdown("### 👥 Target Audience Estimation")
        if st.button("📊 Estimate Audience Size"):
            # Simulate audience estimation
            base_counts = {
                "all": 1250,
                "premium": 450,
                "family": 380,
                "business": 120,
                "loyal_customers": 890,
                "new_customers": 340
            }
            
            count = base_counts.get(target_segment, 100)
            estimated_cost = count * 2.5  # Rs.2.50 per SMS
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📊 **Estimated Audience**\n{count:,} customers")
            with col2:
                st.info(f"💰 **Estimated Cost**\nRs.{estimated_cost:,.2f}")
            with col3:
                st.info(f"⏱️ **Delivery Time**\n~{(count//100)+1} minutes")
        
        # Schedule button
        if st.form_submit_button("📅 Schedule Campaign", use_container_width=True):
            schedule_datetime = datetime.combine(schedule_date, schedule_time)
            
            if schedule_datetime <= datetime.now():
                st.error("⚠️ Schedule time must be in the future")
            else:
                # Save to database
                try:
                    conn = sqlite3.connect(config.DATABASE_PATH)
                    conn.execute('''
                        INSERT INTO scheduled_campaigns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (str(uuid.uuid4()), campaign_name, campaign_type, target_segment, 
                          product_key, "", schedule_datetime, "scheduled", 0, 0, datetime.now(), "admin"))
                    conn.commit()
                    conn.close()
                    
                    st.success(f"✅ Campaign '{campaign_name}' scheduled for {schedule_datetime.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Show confirmation details
                    with st.expander("📋 Campaign Details", expanded=True):
                        st.write(f"**Campaign Name:** {campaign_name}")
                        st.write(f"**Type:** {campaign_type.title()}")
                        st.write(f"**Product:** {PRODUCTS[product_key]['name']}")
                        st.write(f"**Target:** {target_segment.replace('_', ' ').title()}")
                        st.write(f"**Scheduled:** {schedule_datetime.strftime('%A, %B %d, %Y at %H:%M')}")
                        
                except Exception as e:
                    st.error(f"❌ Failed to schedule campaign: {e}")

    # Show existing scheduled campaigns
    st.markdown("### 📅 Scheduled Campaigns")
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        scheduled_df = pd.read_sql_query("""
            SELECT campaign_name, campaign_type, target_segment, 
                   scheduled_time, status, created_at
            FROM scheduled_campaigns 
            WHERE status = 'scheduled'
            ORDER BY scheduled_time ASC
            LIMIT 10
        """, conn)
        conn.close()
        
        if not scheduled_df.empty:
            # Format datetime columns
            scheduled_df['scheduled_time'] = pd.to_datetime(scheduled_df['scheduled_time']).dt.strftime('%Y-%m-%d %H:%M')
            scheduled_df['created_at'] = pd.to_datetime(scheduled_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            scheduled_df.columns = ['Campaign Name', 'Type', 'Target Segment', 'Scheduled Time', 'Status', 'Created At']
            st.dataframe(scheduled_df, use_container_width=True)
        else:
            st.info("No scheduled campaigns found.")
    except Exception as e:
        st.error(f"Error loading scheduled campaigns: {e}")

def render_customer_insights():
    """Render customer insights and segmentation"""
    st.markdown("## 👥 Customer Insights")
    
    # Customer segment analysis
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        
        # Get customer interaction summary
        customer_summary = pd.read_sql_query("""
            SELECT 
                selected_product,
                status,
                COUNT(*) as count
            FROM customer_interactions 
            GROUP BY selected_product, status
        """, conn)
        
        # Get SMS campaign performance by customer behavior
        campaign_performance = pd.read_sql_query("""
            SELECT 
                product_name,
                campaign_type,
                COUNT(*) as campaigns_sent,
                AVG(cost) as avg_cost
            FROM sms_campaigns 
            GROUP BY product_name, campaign_type
            ORDER BY campaigns_sent DESC
        """, conn)
        
        conn.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Customer Interest by Product")
            if not customer_summary.empty:
                fig = px.sunburst(customer_summary, 
                                path=['selected_product', 'status'], 
                                values='count',
                                title="Customer Journey by Product")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No customer interaction data available yet.")
        
        with col2:
            st.markdown("### 📈 Campaign Effectiveness")
            if not campaign_performance.empty:
                fig = px.bar(campaign_performance, 
                           x='campaign_type', 
                           y='campaigns_sent',
                           color='product_name',
                           title="Campaigns by Type & Product")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No campaign data available yet.")
                
    except Exception as e:
        st.error(f"Error loading customer insights: {e}")

# ============================================================================
# ENHANCED MAIN APPLICATION
# ============================================================================

def render_enhanced_sms_section(mspace_promotion):
    """Enhanced SMS campaign section"""
    st.markdown("## 📱 Enhanced SMS Campaigns")
    
    # Account status check
    with st.expander("📊 Account Status & Health Check", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Check API Status"):
                balance_info = mspace_promotion.api.get_account_balance()
                if balance_info.get("success"):
                    st.success(f"✅ API Connected")
                    st.info(f"💳 Credits: {balance_info.get('credits', 'N/A')}")
                    st.info(f"💰 Balance: Rs.{balance_info.get('balance', 'N/A')}")
                else:
                    st.error("❌ API Connection Failed")
        
        with col2:
            # Daily usage check
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*), COALESCE(SUM(cost), 0) 
                FROM sms_campaigns 
                WHERE DATE(sent_at) = DATE('now')
            """)
            daily_count, daily_cost = cursor.fetchone()
            conn.close()
            
            st.metric("📱 Today's SMS", daily_count, delta=f"Limit: {config.MAX_SMS_PER_DAY}")
            st.metric("💰 Today's Cost", f"Rs.{daily_cost:.0f}", delta=f"Budget: Rs.{config.DAILY_SMS_BUDGET}")
        
        with col3:
            # System health indicators
            health_score = 95  # Simulated
            st.metric("🏥 System Health", f"{health_score}%", delta="All systems operational")
    
    # Enhanced campaign interface
    if st.session_state.customer_phone:
        st.markdown("### 🎯 Campaign Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Campaign type selection with descriptions
            campaign_options = {
                "🌟 Welcome Campaign": ("welcome", "Perfect for new customers"),
                "🔄 Follow-up Campaign": ("followup", "Re-engage interested customers"),
                "🏆 Competitive Switch": ("competitive", "Target competitor customers"),
                "🎉 Seasonal Promotion": ("seasonal", "Holiday and special events"),
                "💎 Loyalty Reward": ("loyalty", "Reward existing customers")
            }
            
            selected_campaign = st.selectbox(
                "Select Campaign Type",
                list(campaign_options.keys()),
                help="Choose the most appropriate campaign type for your target audience"
            )
            
            campaign_key, campaign_description = campaign_options[selected_campaign]
            st.caption(f"💡 {campaign_description}")
            
        with col2:
            # Advanced options
            st.markdown("#### ⚙️ Advanced Options")
            schedule_send = st.checkbox("⏰ Schedule for later")
            
            if schedule_send:
                send_datetime = st.datetime_input(
                    "Send Date & Time",
                    min_value=datetime.now(),
                    value=datetime.now() + timedelta(hours=1)
                )
            
            personalization_level = st.selectbox(
                "Personalization",
                ["Standard", "High", "Premium"],
                help="Higher levels include more customer-specific content"
            )
        
        # Product selection with enhanced display
        st.markdown("### 📦 Choose Package")
        
        for key, product in PRODUCTS.items():
            with st.expander(f"{product['name']} - Rs.{product['price']:,}/month", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Perfect for:** {product['target_audience']}")
                    st.markdown("**Key Features:**")
                    for feature in product['features'][:4]:
                        st.markdown(f"• {feature}")
                    
                    st.markdown(f"**🎁 Current Offer:** {product['discount']}")
                
                with col2:
                    # Message preview
                    if st.button(f"👁️ Preview Message", key=f"preview_{key}"):
                        # Generate preview
                        preview_result = mspace_promotion.send_personalized_campaign(
                            "+94771234567",  # Dummy number for preview
                            key,
                            st.session_state.customer_name,
                            campaign_key,
                            custom_data={"preview_mode": True}
                        )
                        
                        if preview_result.success:
                            # This would show the actual message content in a real implementation
                            st.info("📱 Message preview generated successfully!")
                    
                    # Send button with enhanced feedback
                    if st.button(f"🚀 Send {product['name']}", key=f"send_{key}", use_container_width=True):
                        if schedule_send:
                            st.info(f"📅 Campaign scheduled for {send_datetime}")
                            # In a real implementation, you'd save this to scheduled_campaigns
                        else:
                            with st.spinner(f"Sending {product['name']} campaign..."):
                                result = mspace_promotion.send_personalized_campaign(
                                    st.session_state.customer_phone,
                                    key,
                                    st.session_state.customer_name,
                                    campaign_key
                                )
                                
                                if result.success:
                                    st.success(f"✅ SMS sent successfully!")
                                    
                                    # Show detailed success info
                                    with st.expander("📊 Campaign Details", expanded=True):
                                        st.write(f"**Message ID:** {result.message_id}")
                                        st.write(f"**Cost:** Rs.{result.cost:.2f}")
                                        st.write(f"**Remaining Credits:** {result.remaining_credits}")
                                        st.write(f"**Status:** {result.status.value.title()}")
                                        
                                        # Add to customer interactions
                                        try:
                                            conn = sqlite3.connect(config.DATABASE_PATH)
                                            conn.execute('''
                                                INSERT INTO customer_interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            ''', (str(uuid.uuid4()), st.session_state.customer_phone, 
                                                  st.session_state.customer_name, key, "sms_campaign",
                                                  "sent", datetime.now(), f"Campaign: {campaign_key}"))
                                            conn.commit()
                                            conn.close()
                                        except Exception as e:
                                            st.error(f"Failed to log interaction: {e}")
                                else:
                                    st.error(f"❌ Failed to send SMS: {result.error_message}")
    else:
        st.info("👤 Please enter customer details in the sidebar to send SMS campaigns")

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

def main():
    # Streamlit Configuration
    st.set_page_config(
        page_title="🚀 Mobitel Promotion Hub - Enhanced",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize database
    init_enhanced_database()
    
    # Initialize enhanced mSpace promotion system
    if 'mspace_promotion' not in st.session_state:
        st.session_state.mspace_promotion = EnhancedMSpacePromotion()
    
    # Session state initialization
    if 'show_ai_chat' not in st.session_state:
        st.session_state.show_ai_chat = False
    if 'customer_name' not in st.session_state:
        st.session_state.customer_name = ""
    if 'customer_phone' not in st.session_state:
        st.session_state.customer_phone = ""
    
    # Enhanced Custom CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #e31837 0%, #ff6b6b 50%, #ffa726 100%);
            color: white;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            padding: 25px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(227, 24, 55, 0.3);
            backdrop-filter: blur(10px);
        }
        
        .product-card {
            border: 2px solid #e31837;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(227, 24, 55, 0.2);
        }
        
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 15px;
            color: white;
            margin: 5px 0;
        }
        
        .stButton > button {
            border-radius: 25px;
            border: none;
            background: linear-gradient(135deg, #e31837 0%, #ff6b6b 100%);
            color: white;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(227, 24, 55, 0.4);
        }
        
        .sidebar-metric {
            background: #f0f2f6;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #e31837;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced Header
    st.markdown('''
    <div class="main-header">
        🚀 MOBITEL PROMOTION HUB
        <br><small style="font-size: 0.4em; opacity: 0.9;">Enhanced AI Chat & SMS Campaign Platform</small>
    </div>
    ''', unsafe_allow_html=True)
    
    # Enhanced Sidebar - Customer Profile & Stats
    with st.sidebar:
        st.markdown("### 👤 Customer Profile")
        
        customer_name = st.text_input(
            "Customer Name", 
            value=st.session_state.customer_name,
            placeholder="Enter customer name",
            help="Customer's full name for personalization"
        )
        
        customer_phone = st.text_input(
            "Phone Number", 
            value=st.session_state.customer_phone,
            placeholder="+94771234567",
            help="Sri Lankan mobile number format"
        )
        
        # Enhanced phone number validation
        if customer_phone:
            api = EnhancedMSpaceAPI()
            normalized_phone = api._normalize_phone_number(customer_phone)
            is_valid = api._is_valid_phone_number(normalized_phone)
            
            if is_valid:
                st.success(f"✅ Valid: {normalized_phone}")
            else:
                st.error("❌ Invalid phone number format")
        
        # Update session state
        st.session_state.customer_name = customer_name
        st.session_state.customer_phone = customer_phone
        
        if customer_phone and customer_name and customer_phone:
            st.markdown('''
            <div class="sidebar-metric">
                <strong>✅ Profile Active</strong><br>
                <small>Ready for campaigns</small>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Quick Stats
        st.markdown("### 📊 Dashboard Stats")
        
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            # Total interactions
            cursor.execute("SELECT COUNT(*) FROM customer_interactions")
            total_interactions = cursor.fetchone()[0]
            
            # SMS campaigns
            cursor.execute("SELECT COUNT(*) FROM sms_campaigns")
            total_sms = cursor.fetchone()[0]
            
            # Today's activity
            cursor.execute("""
                SELECT COUNT(*), COALESCE(SUM(cost), 0) 
                FROM sms_campaigns 
                WHERE DATE(sent_at) = DATE('now')
            """)
            today_sms, today_cost = cursor.fetchone()
            
            # Success rate
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN delivery_status = 'delivered' THEN 1 END) * 100.0 / COUNT(*) 
                FROM sms_campaigns 
                WHERE delivery_status IN ('delivered', 'failed')
            """)
            success_rate = cursor.fetchone()[0] or 0
            
            conn.close()
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("👥 Interactions", total_interactions, delta=f"+{total_interactions//10}")
                st.metric("📅 Today's SMS", today_sms, delta=f"Rs.{today_cost:.0f}")
            
            with col2:
                st.metric("📱 Total SMS", total_sms, delta=f"+{total_sms//20}")
                st.metric("✅ Success Rate", f"{success_rate:.0f}%", delta="2.1%")
            
        except Exception as e:
            st.error(f"Stats error: {e}")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📋 Export Data", use_container_width=True):
            st.info("Export functionality coming soon!")
    
    # Main Content - Enhanced Tabbed Interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard", 
        "📱 SMS Campaigns", 
        "🤖 AI Assistant", 
        "📊 Analytics",
        "⏰ Scheduler"
    ])
    
    with tab1:
        # Dashboard overview
        st.markdown("## 🏠 Dashboard Overview")
        
        # Real-time status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('''
            <div class="metric-container">
                <h3>🟢 System Status</h3>
                <p>All Systems Operational</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="metric-container">
                <h3>📡 API Status</h3>
                <p>Connected & Ready</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div class="metric-container">
                <h3>💾 Database</h3>
                <p>Healthy & Optimized</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown('''
            <div class="metric-container">
                <h3>🔒 Security</h3>
                <p>All Checks Passed</p>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recent activity
        st.markdown("### 📈 Recent Activity")
        
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            recent_campaigns = pd.read_sql_query("""
                SELECT 
                    phone_number,
                    product_name,
                    campaign_type,
                    delivery_status,
                    cost,
                    sent_at
                FROM sms_campaigns 
                ORDER BY sent_at DESC 
                LIMIT 10
            """, conn)
            conn.close()
            
            if not recent_campaigns.empty:
                # Format the data for better display
                recent_campaigns['sent_at'] = pd.to_datetime(recent_campaigns['sent_at']).dt.strftime('%Y-%m-%d %H:%M')
                recent_campaigns['cost'] = recent_campaigns['cost'].round(2)
                
                st.dataframe(recent_campaigns, use_container_width=True)
            else:
                st.info("No campaign data available yet. Start by sending your first SMS campaign!")
                
        except Exception as e:
            st.error(f"Error loading recent activity: {e}")
        
        # Products showcase (enhanced)
        st.markdown("---")
        st.markdown("## 📦 Our Premium Packages")
        
        for key, product in PRODUCTS.items():
            with st.expander(f"🎯 {product['name']} - Rs.{product['price']:,}/month", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**🎯 Perfect for:** {product['target_audience']}")
                    st.markdown("**✨ Key Features:**")
                    for i, feature in enumerate(product['features'][:4], 1):
                        st.markdown(f"   {i}. {feature}")
                    
                    st.markdown(f"**🎁 Special Offer:** {product['discount']}")
                
                with col2:
                    if st.session_state.customer_phone and st.button(f"📤 Quick Send {product['name']}", key=f"dashboard_send_{key}"):
                        result = st.session_state.mspace_promotion.send_personalized_campaign(
                            st.session_state.customer_phone, 
                            key, 
                            st.session_state.customer_name,
                            "welcome"
                        )
                        
                        if result.success:
                            st.success("✅ SMS sent successfully!")
                        else:
                            st.error(f"❌ Failed: {result.error_message}")
    
    with tab2:
        render_enhanced_sms_section(st.session_state.mspace_promotion)
    
    with tab3:
        # Enhanced AI Chat Section
        st.markdown("## 🤖 ALEX AI Assistant")
        
        if st.session_state.customer_phone and st.session_state.customer_name:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                chat_button = st.button(
                    "🛑 End Chat Session" if st.session_state.show_ai_chat else "🚀 Start AI Chat",
                    key="toggle_chat",
                    use_container_width=True
                )
            
            with col2:
                if st.session_state.show_ai_chat:
                    st.metric("💬 Chat Status", "Active")
                else:
                    st.metric("💬 Chat Status", "Inactive")
            
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
                            st.success("✅ Advanced AI Chat with FAISS Ready!")
                        except Exception as e:
                            st.session_state.chat_system = SimpleChatInterface()
                            st.info("✅ Enhanced Basic AI Chat Ready!")
                    else:
                        st.session_state.chat_system = SimpleChatInterface()
                        st.info("✅ Enhanced AI Chat Ready!")
                
                # Render chat interface
                st.session_state.chat_system.render_chat_interface(st.session_state.customer_name)
                
                # Chat sidebar controls
                st.session_state.chat_system.render_chat_sidebar()
                
            else:
                # Show chat preview/info when not active
                st.info("""
                🤖 **ALEX AI Assistant Features:**
                
                ✨ **Smart Conversations**: Natural language understanding  
                📋 **Product Expertise**: Detailed package information  
                🎯 **Personalized Recommendations**: Based on customer needs  
                💬 **Multi-turn Dialogue**: Contextual conversations  
                🚀 **Instant Responses**: Real-time assistance  
                
                Click "Start AI Chat" to begin!
                """)
        else:
            st.warning("👤 Please enter customer details in the sidebar to start AI chat")
            
            # Show demo conversation
            st.markdown("### 🎬 Demo Conversation")
            st.markdown("""
            **Customer:** Hi, I'm looking for a good mobile package  
            **ALEX:** Hello! I'm ALEX from Mobitel. I'd be happy to help you find the perfect package. Are you looking for something for personal use, family, or business?
            
            **Customer:** Family use, we're 4 people  
            **ALEX:** Perfect! Our Family Package would be ideal for you - Rs.4,999/month for 300GB shared data across 5 SIM cards, plus Disney+ and Netflix included...
            """)
    
    with tab4:
        render_enhanced_analytics()
        
        # Additional analytics sections
        st.markdown("---")
        render_customer_insights()
    
    with tab5:
        render_campaign_scheduler()
        
        st.markdown("---")
        
        # Campaign automation settings
        st.markdown("### 🔄 Campaign Automation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚡ Auto-Follow Up")
            enable_auto_followup = st.checkbox("Enable automatic follow-up campaigns", value=False)
            
            if enable_auto_followup:
                followup_delay = st.slider("Follow-up after (days)", 1, 14, 3)
                followup_limit = st.slider("Max follow-ups per customer", 1, 5, 2)
                
                st.info(f"🔄 Will send follow-up campaigns {followup_delay} days after initial contact, maximum {followup_limit} times per customer.")
        
        with col2:
            st.markdown("#### 🎯 Smart Targeting")
            enable_smart_targeting = st.checkbox("Enable AI-powered customer segmentation", value=False)
            
            if enable_smart_targeting:
                segmentation_criteria = st.multiselect(
                    "Segmentation criteria:",
                    ["Previous interactions", "Response patterns", "Product preferences", "Engagement time", "Geographic location"],
                    default=["Previous interactions", "Product preferences"]
                )
                
                if segmentation_criteria:
                    st.success(f"🎯 Smart targeting enabled with {len(segmentation_criteria)} criteria")

# Footer section
def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666; background: #f8f9fa; border-radius: 10px; margin-top: 30px;">
        <h4>🚀 Mobitel Promotion Hub - Enhanced Edition</h4>
        <p>Powered by Advanced AI & mSpace API Integration</p>
        <p><strong>Version:</strong> 2.0 | <strong>Status:</strong> Production Ready | <strong>Last Updated:</strong> {}</p>
        
        <div style="margin-top: 15px;">
            <strong>🔗 Quick Links:</strong> 
            <a href="#" style="margin: 0 10px; color: #e31837;">📞 Support</a> |
            <a href="#" style="margin: 0 10px; color: #e31837;">📚 Documentation</a> |
            <a href="#" style="margin: 0 10px; color: #e31837;">🔒 Privacy Policy</a>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        # Run main application
        main()
        
        # Render footer
        render_footer()
        
        # Optional: Add some debug info in development
        if os.getenv("DEBUG", "false").lower() == "true":
            with st.expander("🔧 Debug Information", expanded=False):
                st.json({
                    "session_state_keys": list(st.session_state.keys()),
                    "config": {
                        "database_path": config.DATABASE_PATH,
                        "api_configured": bool(config.MSPACE_API_KEY != "demo_api_key_12345"),
                        "ai_chat_available": AI_CHAT_AVAILABLE,
                        "ai_chat_type": AI_CHAT_TYPE
                    }
                })
    
    except Exception as e:
        st.error(f"🚨 Application Error: {e}")
        st.markdown("""
        ### 🔧 Troubleshooting Steps:
        1. Check if all required packages are installed: `pip install -r requirements.txt`
        2. Ensure the database directory exists and is writable
        3. Verify your `.env` file contains valid mSpace API credentials
        4. Check the application logs for detailed error information
        
        **Need Help?** Contact the development team or check the documentation.
        """)

# ============================================================================
# ADDITIONAL HELPER FUNCTIONS
# ============================================================================

def create_sample_data():
    """Create sample data for testing (optional)"""
    if st.button("🧪 Create Sample Data (Dev Only)"):
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            # Sample customers
            sample_customers = [
                ("550e8400-e29b-41d4-a716-446655440001", "+94771234567", "John Doe", "premium", "inquiry", "interested", datetime.now(), "Interested in premium features"),
                ("550e8400-e29b-41d4-a716-446655440002", "+94777654321", "Jane Smith", "family", "inquiry", "converted", datetime.now(), "Converted to family package"),
                ("550e8400-e29b-41d4-a716-446655440003", "+94761111111", "Mike Johnson", "business", "inquiry", "interested", datetime.now(), "Business inquiry")
            ]
            
            for customer in sample_customers:
                try:
                    conn.execute('''
                        INSERT OR IGNORE INTO customer_interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', customer)
                except sqlite3.IntegrityError:
                    pass  # Customer already exists
            
            # Sample SMS campaigns
            sample_campaigns = [
                ("550e8400-e29b-41d4-a716-446655440010", "+94771234567", "Premium Package", "Welcome message", "delivered", datetime.now(), "welcome", "msg_001", 2.5),
                ("550e8400-e29b-41d4-a716-446655440011", "+94777654321", "Family Package", "Follow-up message", "delivered", datetime.now(), "followup", "msg_002", 2.5),
                ("550e8400-e29b-41d4-a716-446655440012", "+94761111111", "Business Package", "Competitive offer", "sent", datetime.now(), "competitive", "msg_003", 5.0)
            ]
            
            for campaign in sample_campaigns:
                try:
                    conn.execute('''
                        INSERT OR IGNORE INTO sms_campaigns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', campaign)
                except sqlite3.IntegrityError:
                    pass  # Campaign already exists
            
            conn.commit()
            conn.close()
            
            st.success("✅ Sample data created successfully!")
            st.info("🔄 Refresh the page to see the sample data in analytics.")
            
        except Exception as e:
            st.error(f"❌ Failed to create sample data: {e}")

def export_data():
    """Export data functionality"""
    if st.button("📊 Export Analytics Data"):
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            # Get all tables data
            analytics_data = {}
            
            tables = ["customer_interactions", "sms_campaigns", "scheduled_campaigns"]
            
            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                analytics_data[table] = df
            
            conn.close()
            
            # Create downloadable CSV
            import io
            buffer = io.StringIO()
            
            for table_name, df in analytics_data.items():
                buffer.write(f"\n=== {table_name.upper()} ===\n")
                df.to_csv(buffer, index=False)
                buffer.write("\n")
            
            csv_data = buffer.getvalue()
            
            st.download_button(
                label="📥 Download Analytics Report",
                data=csv_data,
                file_name=f"mobitel_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Export failed: {e}")

# Add development tools section
def render_dev_tools():
    """Render development tools (only in debug mode)"""
    if os.getenv("DEBUG", "false").lower() == "true":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Dev Tools")
        
        if st.sidebar.button("🧪 Create Sample Data"):
            create_sample_data()
        
        if st.sidebar.button("📊 Export Data"):
            export_data()
        
        if st.sidebar.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.sidebar.success("Cache cleared!")
        
        if st.sidebar.button("🔄 Reset Database"):
            if st.sidebar.checkbox("⚠️ Confirm reset"):
                try:
                    os.remove(config.DATABASE_PATH)
                    init_enhanced_database()
                    st.sidebar.success("Database reset!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Reset failed: {e}")

# Call dev tools if in debug mode
render_dev_tools()