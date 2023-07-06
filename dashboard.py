import tiktoken
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

st.set_page_config(layout="wide")
import time

from typing import Optional
import logging
logging.basicConfig(level=logging.INFO)

from threading import Thread, Event
from chatgpt_util import ChatGPTUtil
from web_crawler import crawl


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
    
_USER_PROMPT = """\
    ### COMPANY TEXT ###
    {COMPANY_TEXT}
    ####################
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
    


def summarize(website_url: Optional[str]) -> str:
    global shared_dict, thread_event
    
    if website_url is None:
        return ""
    
    thread_event.set()
    st.session_state['progress_num'] = 10
    st.session_state['progress_text'] = "Crawling website..."
    logging.info(f"cache hit: {website_url in shared_dict}")
    if website_url not in shared_dict:
        logging.info(f'Website url not in cache: {website_url} \n Cache: {shared_dict.keys()}')
        shared_dict[website_url] = crawl(website_url, chatgpt_util, tokenizer)    
    
    st.session_state['progress_text'] = "Generating system prompt..."
    st.session_state['progress_num'] = 25
    logging.info(f"website_summary:\n {shared_dict[website_url]}")
    logging.info(f"system_prompt:\n {st.session_state['system_prompt']}")
    
    formatted_user_prompt = _USER_PROMPT.format(COMPANY_TEXT=shared_dict[website_url])
    messages = [
        {'role': 'system', 'content': st.session_state['system_prompt']},
        {'role': 'user', 'content': formatted_user_prompt}]
    
    st.session_state['progress_text'] = "Generating AI suggestions..."
    st.session_state['progress_num'] = 50
    ai_suggestions = chatgpt_util.get_chat_completion(
        messages=messages,
        model='gpt-4')[0]
    
    st.session_state['ai_output'] = ai_suggestions
    thread_event.clear()
    
def start_summarize_runner(website_url: Optional[str]):
    
    thread = Thread(target=summarize, args=(website_url,), daemon=True)
    cntx_thread = add_script_run_ctx(thread)
    cntx_thread.start()
    st.session_state['exec_thread'] = cntx_thread

    
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


col1, col2 = st.columns(2)

with col1:
    ai_output_area = st.empty()
    ai_output_area.text_area(label="AI Output", value=st.session_state['ai_output'], height=400)

with  col2:
    
    st.text_area(label="System Prompt", value=st.session_state['system_prompt'], height=200)
    chat_response = st.text_input(label="Enter a website URL to summarize and get AI suggestions")
    chat_response_button = st.button(
            label="Send Response",
            type="primary",
            on_click=start_summarize_runner,
            args=(chat_response,))
    
    progress_bar_slot = st.empty()

    
if st.session_state['exec_thread'] is not None:
    
   # ai_output_area.text_area(label="AI Output", value="Processing...", height=400)
    exec_thread: Thread = st.session_state['exec_thread']
    while thread_event.is_set():
        progress_bar_slot.progress(st.session_state['progress_num'], st.session_state['progress_text'])
        time.sleep(1)
        
    progress_bar_slot.progress(100, "Done!")
   # ai_output_area.text_area(label="AI Output", value=st.session_state['ai_output'], height=400)
    st.session_state['exec_thread'] = None
    progress_bar_slot.empty()