import os
import re
import numpy as np
import time
import joblib
import gc
import base64
import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from typing import Type
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from RAG.tools.pdf_search_tool import DocumentSearchTool
from RAG.tools.llm_chat_tool import start_llm_chat, send_llm_message
from RAG.tools.api_tools import YahooFinanceTool, NewsAPITool, CryptoAPITool
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)


llm = LLM(
    model="gemini/gemini-2.0-flash-exp",
    temperature=0.5
)

if not os.path.exists('data'):
    os.makedirs('data')

try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}




def create_agents_and_tasks(primary_tool, query, mode):
    """Creates a Crew with the given primary tool (if any), web search tool, and API tools based on the mode."""
    web_search_tool = SerperDevTool()
    yahoo_finance_tool = YahooFinanceTool()
    news_api_tool = NewsAPITool()
    crypto_api_tool = CryptoAPITool()

    if mode == 'pdf':
        tools = [primary_tool, web_search_tool, yahoo_finance_tool, news_api_tool, crypto_api_tool]
        goal = "Retrieve the most relevant information from the PDF first, then from web search or APIs if needed."
        retrieval_description = "Retrieve the most relevant information from the PDF for the user query: {query}."
        retrieval_output = "The most relevant information from the PDF."
    elif mode == 'live_data':
        tools = [web_search_tool, yahoo_finance_tool, news_api_tool, crypto_api_tool]
        goal = (
            "Retrieve live data from the appropriate source based on the user query. "
            "Use Yahoo Finance for stock and financial data, CryptoAPI for cryptocurrency information, "
            "NewsAPI for latest news, and web search for general queries."
        )
        retrieval_description = (
            "Retrieve the most relevant information from the available sources for the user query: {query}. "
            "Specify the source of the information (e.g., Yahoo Finance, NewsAPI, CryptoAPI, or web search)."
        )
        retrieval_output = (
            "The most relevant information along with the source, formatted as 'Source: [source name]\nData: [retrieved data]'."
        )

    retriever_agent = Agent(
        role=f"Retrieve relevant information to answer the user query: {query}",
        goal=goal,
        backstory=(
            "You're a meticulous analyst with a keen eye for detail. "
            f"You're known for your ability to understand user queries: {query} "
            "and retrieve knowledge from the most suitable knowledge base."
        ),
        verbose=True,
        tools=tools,
        llm=llm
    )

    response_synthesizer_agent = Agent(
        role=f"Response synthesizer agent for the user query: {query}",
        goal=(
            "Synthesize the retrieved information into a concise and coherent response "
            f"based on the user query: {query}. If no information is retrieved, "
            'respond with "I\'m sorry, I couldn\'t find the information you\'re looking for." '
            "Include the source of the information if available."
        ),
        backstory=(
            "You're a skilled communicator with a knack for turning "
            "complex information into clear and concise responses about finance."
        ),
        verbose=True,
        llm=llm
    )

    retrieval_task = Task(
        description=retrieval_description.format(query=query),
        expected_output=retrieval_output,
        agent=retriever_agent
    )

    response_task = Task(
        description=f"Synthesize the final response for the user query: {query}. Include the source if provided.",
        expected_output=(
            "A concise and coherent response based on the retrieved information "
            f"for the user query: {query}. If a source is specified, include it as 'Source: [source name]'. "
            'If no information is found, respond with "I\'m sorry, I couldn\'t find the information you\'re looking for."'
        ),
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True
    )
    return crew


def reset_chat():
    """Clear the chat history and related session state."""
    try:
        st.session_state.messages = []
        if 'gemini_history' in st.session_state:
            st.session_state.gemini_history = []
        
        if 'chat_id' in st.session_state:
            chat_id = st.session_state.chat_id
            files_to_delete = [
                f'data/{chat_id}-st_messages',
                f'data/{chat_id}-gemini_messages'
            ]
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        if 'pdf_tool' in st.session_state:
            del st.session_state.pdf_tool
        if 'pdf_filename' in st.session_state:
            del st.session_state.pdf_filename
                
        if 'chat_id' in st.session_state:
            del st.session_state.chat_id
        
        gc.collect()
    except Exception as e:
        st.error(f"Error clearing chat: {str(e)}")

def display_pdf(file_path=None):
    """Displays the uploaded PDF in an iframe."""
    if file_path and os.path.exists(file_path):
        display_name = st.session_state.get('pdf_filename', os.path.basename(file_path))
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="600px"
            type="application/pdf"
        >
        </iframe>
        """
        st.markdown(f"### Preview of {display_name}")
        st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit App
st.set_page_config(page_title="FinBOT", page_icon="🤖")

# Sidebar
with st.sidebar:
    st.write('# Chat Sessions')

    options = ['New Chat'] + list(past_chats.keys())
    default_index = 0
    if 'chat_id' in st.session_state and st.session_state.chat_id in past_chats:
        default_index = options.index(st.session_state.chat_id)

    selected = st.selectbox(
        'Select a chat session',
        options,
        index=default_index,
        format_func=lambda x: 'New Chat' if x == 'New Chat' else past_chats[x]['title']
    )

    if selected == 'New Chat':
        with st.form(key='new_chat_form'):
            chat_title = st.text_input('Enter chat title')
            mode = st.selectbox('Select mode', options=['Chat with LLM','Chat with PDF','Chat with Live Data'])
            submit_button = st.form_submit_button('Create Chat')
            if submit_button and chat_title:
                new_chat_id = str(time.time())
                past_chats[new_chat_id] = {
                    'title': chat_title, 
                    'mode': mode,
                    'pdf_filename': None,
                }
                joblib.dump(past_chats, 'data/past_chats_list')
                st.session_state.chat_id = new_chat_id
                st.session_state.chat_title = chat_title
                st.session_state.mode = mode
                st.session_state.messages = []
                if mode in ['Chat with LLM', 'Chat with Live Data']:
                    st.session_state.gemini_history = []
                else:
                    if mode == 'Chat with PDF':
                        st.session_state.pdf_tool = None
                st.rerun()

    elif selected in past_chats:
        st.session_state.chat_id = selected
        st.session_state.chat_title = past_chats[selected]['title']
        st.session_state.mode = past_chats[selected]['mode']
        st.session_state.pdf_filename = past_chats[selected].get('pdf_filename')
        
        st.session_state.messages = []
        messages_file = f'data/{selected}-st_messages'
        if os.path.exists(messages_file):
            st.session_state.messages = joblib.load(messages_file)

        if st.session_state.mode in ['Chat with LLM', 'Chat with Live Data']:
            try:
                st.session_state.gemini_history = joblib.load(f'data/{selected}-gemini_messages')
            except:
                st.session_state.gemini_history = []
        elif st.session_state.mode == 'Chat with PDF':
            pdf_path = f'data/{selected}_pdf.pdf'
            if os.path.exists(pdf_path):
                if 'pdf_tool' not in st.session_state or st.session_state.pdf_tool is None:
                    try:
                        st.session_state.pdf_tool = DocumentSearchTool(
                            file_path=pdf_path,
                            chat_title=st.session_state.chat_title,
                            chat_id=st.session_state.chat_id
                        )
                    except Exception as e:
                        st.error(f"Error loading PDF: {str(e)}")
                        st.session_state.pdf_tool = None
            else:
                st.session_state.pdf_tool = None

    if 'chat_id' in st.session_state:
        if st.session_state.mode == 'Chat with PDF':
            st.write('### PDF Management')
            if st.session_state.get('pdf_tool') is None:
                uploaded_file = st.file_uploader('Upload PDF', type=['pdf'])
                if uploaded_file is not None:
                    pdf_path = f'data/{st.session_state.chat_id}_pdf.pdf'
                    original_filename = uploaded_file.name
                    with open(pdf_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    try:
                        with st.spinner("Indexing PDF... This may take a minute..."):
                            st.session_state.pdf_filename = original_filename
                            st.session_state.pdf_tool = DocumentSearchTool(
                                file_path=pdf_path,
                                chat_title=st.session_state.chat_title,
                                chat_id=st.session_state.chat_id
                            )
                        st.success("PDF indexed successfully!")
                        past_chats[st.session_state.chat_id]['pdf_filename'] = original_filename
                        joblib.dump(past_chats, 'data/past_chats_list')
                    except Exception as e:
                        st.error(f"PDF processing failed: {str(e)}")
                        st.session_state.pdf_tool = None
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
            else:
                pdf_path = f'data/{st.session_state.chat_id}_pdf.pdf'
                if os.path.exists(pdf_path):
                    if st.checkbox("Show PDF Preview", value=True):
                        display_pdf(file_path=pdf_path)

    if 'chat_id' in st.session_state and st.session_state.chat_id in past_chats:
        if st.button('🗑️ Delete Chat'):
            chat_id = st.session_state.chat_id
            chat_title = past_chats[chat_id]['title']
            sanitized_title = re.sub(r'[^a-zA-Z0-9_]', '_', chat_title)[:50]
            if st.session_state.mode == 'Chat with PDF':
                collection_name = f"{sanitized_title}_{chat_id}"
            else:
                collection_name = None
            if collection_name:
                try:
                    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
                    if collection_name in [col.name for col in client.get_collections().collections]:
                        client.delete_collection(collection_name)
                except Exception as e:
                    print(f"Error deleting collection: {str(e)}")
            
            del past_chats[chat_id]
            joblib.dump(past_chats, 'data/past_chats_list')
            for file in [f'data/{chat_id}_pdf.pdf', 
                        f'data/{chat_id}-st_messages', f'data/{chat_id}-gemini_messages']:
                if os.path.exists(file):
                    os.remove(file)
            if 'chat_id' in st.session_state:
                del st.session_state.chat_id
            st.rerun()

    st.button("Clear Chat History", on_click=reset_chat)

# Main Chat Interface
if 'chat_id' not in st.session_state:
    st.write("# FinBOT 🤖")
    st.write("Start a new chat from the sidebar to begin.")
else:
    st.write(f"# {st.session_state.chat_title}")    

    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])
    
    if st.session_state.mode == 'Chat with PDF':
        if st.session_state.get('pdf_tool') is not None:
            prompt = st.chat_input("Ask a question about your PDF...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    with st.spinner("Analyzing document..."):
                        try:
                            crew = create_agents_and_tasks(st.session_state.pdf_tool, prompt, mode='pdf')
                            result = crew.kickoff(inputs={"query": prompt}).raw
                        except Exception as e:
                            result = f"Error processing request: {str(e)}"
                    
                    lines = result.split('\n')
                    for i, line in enumerate(lines):
                        full_response += line
                        if i < len(lines) - 1:
                            full_response += '\n'
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.1)
                    
                    message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": result})
                joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
        else:
            st.info("Please upload a PDF in the sidebar to begin chatting.")
    
    elif st.session_state.mode == 'Chat with LLM':
        try:
            if 'gemini_history' not in st.session_state:
                st.session_state.gemini_history = []
            st.session_state.chat = start_llm_chat(st.session_state.gemini_history)
        except Exception as e:
            st.error(f"Error initializing chat: {str(e)}")
            st.stop()

        prompt = st.chat_input("Your message here...")
        if prompt:
            try:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                response = send_llm_message(st.session_state.chat, prompt)
                with st.chat_message("assistant", avatar="✨"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response:
                        for ch in chunk.text.split(' '):
                            full_response += ch + ' '
                            time.sleep(0.05)
                            message_placeholder.write(full_response + "▌")
                    message_placeholder.write(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response, "avatar": "✨"})
                st.session_state.gemini_history = st.session_state.chat.history
                joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
                joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')
            except Exception as e:
                st.error(f"Error sending message: {str(e)}")
    
    
    elif st.session_state.mode == 'Chat with Live Data':
        prompt = st.chat_input("Ask a question about live data (stocks, news, crypto)...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner("Fetching data from API..."):
                    try:
                        crew = create_agents_and_tasks(primary_tool=None, query=prompt, mode='live_data')
                        result = crew.kickoff(inputs={"query": prompt}).raw
                    except Exception as e:
                        result = f"Error processing request: {str(e)}"
                lines = result.split('\n')
                for i, line in enumerate(lines):
                    full_response += line
                    if i < len(lines) - 1:
                        full_response += '\n'
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.1)
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": result})
            joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')

if __name__ == "__main__":
    pass  