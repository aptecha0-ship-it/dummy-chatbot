import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Load environment variables (Make sure GOOGLE_API_KEY is in your .env file)
load_dotenv()

st.title("My First Chatbot")

# 1. Initialize the Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",  # Using the stable Gemini 2.5 Flash model
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 2. Let the user choose a persona 
persona = st.selectbox("Choose a Persona:", ["Doctor", "Engineer", "Plumber"])

# 3. Define the Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that speaks like a professional {persona}."),
    ("human", "{user_input}"),
])

# 4. Handle Chat Input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Display what the user typed
    with st.chat_message("user"):
        st.write(user_input)

    # Format the prompt with runtime variables
    formatted_prompt = prompt_template.invoke({
        "persona": persona,
        "user_input": user_input
    })

    # Get response from Gemini
    with st.spinner("Thinking..."):
        ai_msg = model.invoke(formatted_prompt)
    
    # Display the Ai response
    with st.chat_message("assistant"):
        st.write(ai_msg.content[0]["text"])
