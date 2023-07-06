from typing import Optional
import logging
logging.basicConfig(level=logging.INFO)
from streamlit.runtime.scriptrunner import add_script_run_ctx
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from threading import Thread

from chatgpt_util import ChatGPTUtil
from web_crawler import crawl

_USER_PROMPT = """\
    ### COMPANY TEXT ###
    {COMPANY_TEXT}
    ####################
    """

def summarize(website_url: Optional[str], chatgpt_util: ChatGPTUtil, tokenizer, thread_event, shared_dict) -> str:
    st.session_state['exec_thread'] = "running"
    
    if website_url is None:
        return ""
    
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
    

def start_summarize_runner_two(website_url: Optional[str], chatgpt_util: ChatGPTUtil, tokenizer, thread_event, shared_dict):
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(summarize, website_url, chatgpt_util, tokenizer, thread_event, shared_dict)
        return future.result()
    
    # shared_dict['output'] = ""
    # thread_event.set()
    # thread = Thread(target=summarize, args=(website_url, chatgpt_util, tokenizer, thread_event, shared_dict,), daemon=True)
    # cntx_thread = add_script_run_ctx(thread)
    # cntx_thread.start()
    # st.session_state['exec_thread'] = cntx_thread