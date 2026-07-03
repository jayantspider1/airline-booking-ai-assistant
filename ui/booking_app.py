# =========================================================
# IMPORT LIBRARIES
# =========================================================

# Streamlit for chatbot UI
import streamlit as st

# OpenAI SDK
from openai import OpenAI

# Load environment variables
from dotenv import load_dotenv

# OS operations
import os


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

# Load .env file
load_dotenv()


# =========================================================
# OPENROUTER CLIENT
# =========================================================

# Create OpenRouter client
client = OpenAI(

    api_key=os.getenv("OPENROUTER_API_KEY"),

    base_url="https://openrouter.ai/api/v1"

)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

# AI Model
MODEL = "openai/gpt-oss-120b"


# =========================================================
# STREAMLIT PAGE SETTINGS
# =========================================================

# Configure page
st.set_page_config(

    page_title="AI Airline Agent",

    page_icon="✈️",

    layout="wide"

)


# =========================================================
# PAGE TITLE
# =========================================================

# Main app title
st.title("✈️ AI Airline Booking Agent")


# =========================================================
# CHAT MEMORY
# =========================================================

# Store conversation history
if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# FIRST AI MESSAGE
# =========================================================

# Welcome message when app opens first time
if len(st.session_state.messages) == 0:

    welcome_message = """
Hello 👋

I am your AI Airline Booking Assistant.

I can help you with:

✈️ Flight booking  
💺 Seat selection  
💰 Ticket prices  
🧳 Baggage support  
❌ Flight cancellations  
⏰ Flight timings  
🌍 Travel suggestions  

How can I help you today?
"""

    # Save assistant welcome message
    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": welcome_message
        }

    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

# Show all previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# USER CHAT INPUT
# =========================================================

# Input box at bottom
user_input = st.chat_input(

    "Type your message..."

)


# =========================================================
# HANDLE USER MESSAGE
# =========================================================

# When user sends message
if user_input:

    # =====================================================
    # SHOW USER MESSAGE
    # =====================================================

    with st.chat_message("user"):

        st.markdown(user_input)


    # =====================================================
    # SAVE USER MESSAGE
    # =====================================================

    st.session_state.messages.append(

        {
            "role": "user",
            "content": user_input
        }

    )


    # =====================================================
    # CALL AI MODEL
    # =====================================================

    response = client.chat.completions.create(

        model=MODEL,

        messages=[

            {
                "role": "system",

                "content": """
You are an intelligent AI Airline Booking Assistant.

Your job is to help users naturally like a real airline support agent.

You help users with:

- Flight booking
- Flight timings
- Ticket prices
- Seat selection
- Baggage support
- Airline recommendations
- Flight cancellations
- Travel planning

Your personality:

- Friendly
- Professional
- Conversational
- Helpful

Rules:

- Ask follow-up questions naturally
- Guide the user step-by-step
- Keep responses clean
- Talk naturally like ChatGPT
- Make the conversation realistic

Example:

User:
I want to travel from Delhi to Mumbai

Assistant:
Sure! What date would you like to travel?

User:
Tomorrow morning

Assistant:
Great! Which class would you prefer?
- Economy
- Business

User:
Economy

Assistant:
Perfect! Let me check the best available flights for you.
"""
            }

        ]

        +

        st.session_state.messages

    )


    # =====================================================
    # EXTRACT AI RESPONSE
    # =====================================================

    ai_response = response.choices[0].message.content


    # =====================================================
    # DISPLAY AI RESPONSE
    # =====================================================

    with st.chat_message("assistant"):

        st.markdown(ai_response)


    # =====================================================
    # SAVE AI RESPONSE
    # =====================================================

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": ai_response
        }

    )