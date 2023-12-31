from typing import Optional
import logging
logging.basicConfig(level=logging.INFO)
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import streamlit  as st
from chatgpt_util import ChatGPTUtil
from web_crawler import crawl

_USER_PROMPT = """\
    ### COMPANY TEXT ###
    {COMPANY_TEXT}
    ####################
    """

def summarize(website_url: Optional[str], chatgpt_util: ChatGPTUtil, tokenizer, shared_dict, stystem_prompt):    
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
    # logging.info(f"system_prompt:\n {st.session_state['system_prompt']}")
    
    formatted_user_prompt = _USER_PROMPT.format(COMPANY_TEXT=shared_dict[website_url])
    messages = [
        {'role': 'system', 'content': stystem_prompt},
        {'role': 'user', 'content': formatted_user_prompt}]
    
    st.session_state['progress_text'] = "Generating AI suggestions..."
    st.session_state['progress_num'] = 50
    ai_suggestions = chatgpt_util.get_chat_completion(
        messages=messages,
        model='gpt-4')[0]
    st.session_state['ai_output'] = ai_suggestions
    st.session_state['progress_text'] = "Generating user summary..."    
