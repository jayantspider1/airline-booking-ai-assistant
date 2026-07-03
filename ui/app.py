import json
import os

import chromadb
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Airline AI Support Agent",
    page_icon="✈️",
    layout="wide"
)


# ==========================================
# LOAD ENV VARIABLES
# ==========================================

load_dotenv("../.env")

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)


# ==========================================
# INITIALIZE LLM
# ==========================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MODEL = "openai/gpt-oss-120b:free"


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# LOAD CHROMADB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="../vector_db"
)

collection = chroma_client.get_collection(
    name="airline_support_knowledge"
)


# ==========================================
# FUNCTIONS
# ==========================================

def check_baggage_policy(airline):

    policies = {
        "Delta": "Delta allows 1 checked bag up to 23kg.",
        "United": "United allows one cabin bag.",
        "American": "American Airlines baggage fees may apply."
    }

    return policies.get(
        airline,
        "Policy not found."
    )


def calculate_compensation(delay_hours):

    if delay_hours >= 5:

        return (
            "Passenger eligible for hotel, meals, "
            "and compensation."
        )

    elif delay_hours >= 3:

        return (
            "Passenger eligible for meal vouchers."
        )

    else:

        return "No compensation available."


def check_flight_status(flight_number):

    database = {

        "AI202": "Delayed by 2 hours",

        "DL404": "Boarding",

        "UA101": "Cancelled"

    }

    return database.get(
        flight_number,
        "Flight not found."
    )


# ==========================================
# RAG SEARCH
# ==========================================

def search_airline_knowledge(
    query,
    top_k=3
):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]

    return "\n\n".join(documents)


# ==========================================
# TOOLS
# ==========================================

tools = [

    {
        "type": "function",

        "function": {

            "name": "check_baggage_policy",

            "description": "Get baggage policy",

            "parameters": {

                "type": "object",

                "properties": {

                    "airline": {
                        "type": "string"
                    }

                },

                "required": ["airline"]

            }
        }
    },

    {
        "type": "function",

        "function": {

            "name": "calculate_compensation",

            "description": "Calculate compensation",

            "parameters": {

                "type": "object",

                "properties": {

                    "delay_hours": {
                        "type": "integer"
                    }

                },

                "required": ["delay_hours"]

            }
        }
    },

    {
        "type": "function",

        "function": {

            "name": "check_flight_status",

            "description": "Check flight status",

            "parameters": {

                "type": "object",

                "properties": {

                    "flight_number": {
                        "type": "string"
                    }

                },

                "required": ["flight_number"]

            }
        }
    }
]


# ==========================================
# AI AGENT
# ==========================================

def airline_agent(user_query):

    response = client.chat.completions.create(

        model=MODEL,

        messages=[

            {
                "role": "system",

                "content": """
You are an intelligent airline customer support AI agent.

You help users with:
- baggage support
- flight delays
- compensation
- refunds
- airline complaints
- flight status

Be conversational and friendly.
Talk naturally like ChatGPT.
Use tools whenever needed.
Use RAG knowledge whenever useful.
"""
            },

            {
                "role": "user",
                "content": user_query
            }

        ],

        tools=tools

    )

    message = response.choices[0].message


    # ======================================
    # TOOL CALLING
    # ======================================

    if message.tool_calls:

        tool_call = message.tool_calls[0]

        tool_name = tool_call.function.name

        tool_args = json.loads(
            tool_call.function.arguments
        )

        if tool_name == "calculate_compensation":

            result = calculate_compensation(
                tool_args["delay_hours"]
            )

        elif tool_name == "check_baggage_policy":

            result = check_baggage_policy(
                tool_args["airline"]
            )

        elif tool_name == "check_flight_status":

            result = check_flight_status(
                tool_args["flight_number"]
            )

        else:

            result = "Tool not found."


        final_response = client.chat.completions.create(

            model=MODEL,

            messages=[

                {
                    "role": "user",
                    "content": user_query
                },

                message.model_dump(),

                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }

            ]

        )

        return final_response.choices[0].message.content


    # ======================================
    # RAG RESPONSE
    # ======================================

    else:

        rag_result = search_airline_knowledge(
            user_query
        )

        final_prompt = f"""
Customer Question:
{user_query}

Airline Knowledge:
{rag_result}

Give a conversational and helpful airline support response.
"""

        final_response = client.chat.completions.create(

            model=MODEL,

            messages=[

                {
                    "role": "user",
                    "content": final_prompt
                }

            ]

        )

        return final_response.choices[0].message.content


# ==========================================
# CHAT MEMORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================
# FIRST ASSISTANT MESSAGE
# ==========================================

if len(st.session_state.messages) == 0:

    welcome_message = """
Hello 👋

I am your AI Airline Customer Support Assistant.

I can help you with:

✈️ Flight delays  
🧳 Baggage policies  
💰 Compensation  
❌ Airline complaints  
📄 Refund support  
📞 Flight status  

How can I help you today?
"""

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": welcome_message
        }

    )


# ==========================================
# PAGE TITLE
# ==========================================

st.title("✈️ Airline AI Customer Support Agent")


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# USER INPUT
# ==========================================

user_input = st.chat_input(
    "Type your airline support question..."
)


# ==========================================
# HANDLE USER INPUT
# ==========================================

if user_input:

    # ======================================
    # SHOW USER MESSAGE
    # ======================================

    with st.chat_message("user"):

        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append(

        {
            "role": "user",
            "content": user_input
        }

    )


    # ======================================
    # AI RESPONSE
    # ======================================

    with st.spinner("Thinking..."):

        response = airline_agent(
            user_input
        )


    # ======================================
    # SHOW AI RESPONSE
    # ======================================

    with st.chat_message("assistant"):

        st.markdown(response)

    # Save AI response
    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": response
        }

    )