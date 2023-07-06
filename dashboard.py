import tiktoken
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from concurrent.futures import Future
st.set_page_config(layout="wide")
import time

from typing import Optional
import logging
logging.basicConfig(level=logging.INFO)

from threading import Thread, Event
from chatgpt_util import ChatGPTUtil
from runner import summarize

@st.cache_resource
def get_chatgpt_util():
    return ChatGPTUtil()

@st.cache_resource
def get_event():
    return Event()

@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

@st.cache_resource
def get_shared_dict():
    return {}

chatgpt_util = get_chatgpt_util()
tokenizer = get_tokenizer()
shared_dict = get_shared_dict()
thread_event = get_event()

SYSTEM_PROMPT = """\
    You will be provided with the text scraped from a companies website in the following format:
    ### COMPANY TEXT ###
    <company text>
    ####################
    
    Given this context, summarize the buisness's main services in 2-3 sentences. 
    
    Then, based on the buisness's services, suggest potential AI use cases for the buisness. 
    For each use case, explain in laymans's terms how the AI system would work and how it would benefit the buisness.
    """
    

    
if "exec_thread" not in st.session_state:
    st.session_state['exec_thread'] = None
if "progress_text" not in st.session_state:
    st.session_state['progress_text'] = ""
if "progress_num" not in st.session_state:
    st.session_state['progress_num'] = 0
if "system_prompt" not in st.session_state:
    st.session_state['system_prompt'] = SYSTEM_PROMPT
if "ai_output" not in st.session_state:
    st.session_state['ai_output'] = ""
    
    


    
st.title("Kate's AI Assistant")
st.header("AI Use Case Generator")

# Force responsive layout for columns also on mobile
st.write(
    """<style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>""",
    unsafe_allow_html=True,
)

def start_summarize_runner(website_url: Optional[str]):
    global shared_dict, thread_event
    
    shared_dict['output'] = ""
    thread_event.set()
    thread = Thread(target=summarize, args=(website_url, chatgpt_util, tokenizer, thread_event, shared_dict,), daemon=True)
    cntx_thread = add_script_run_ctx(thread)
    cntx_thread.start()
    st.session_state['exec_thread'] = cntx_thread


logging.error(f"exec_thread: {st.session_state['exec_thread']}")

    
col1, col2 = st.columns(2)

with col1:
    ai_output_area = st.empty()
    ai_output_area.text_area(label="AI Output", value=st.session_state['ai_output'], height=400, key='ai_output_1')

with  col2:
    
    st.text_area(label="System Prompt", value=st.session_state['system_prompt'], height=200, key='system_prompt_input')
    chat_response = st.text_input(label="Enter a website URL to summarize and get AI suggestions", key='text_input')
    
    chat_response_button = st.button(
            label="GO",
            type="primary",
            on_click=summarize,
            args=(chat_response,chatgpt_util, tokenizer, shared_dict, st.session_state['system_prompt']),
            key='text_input_button')
    
    
    progress_bar_slot = st.empty()


# if 'future' in shared_dict:
    
#     future: Future = shared_dict['future']
#     while future.running():    
#         progress_bar_slot.progress(1, 'Loading AI Output...')
#         time.sleep(1)
        
#     progress_bar_slot.progress(100, "Done!")
ai_output_area.text_area(label="AI Output", value=st.session_state['ai_output'], height=400, key='final_ai_output')
    # st.session_state['exec_thread'] = None
    # progress_bar_slot.empty()
    # shared_dict.pop('future')
