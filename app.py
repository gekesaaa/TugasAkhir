import warnings
from transformers import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

import streamlit as st

st.set_page_config(
    page_title="Mental Health Assistant",
    layout="wide",
    # initial_sidebar_state="collapsed"
)

st.switch_page(
    "pages/chat-full.py"
)