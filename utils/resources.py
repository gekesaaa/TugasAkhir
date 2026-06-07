import os
import psutil
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import nltk
from supabase import create_client
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# DEBUG UTILITIES
# =====================================================
def log_step(msg):
    print(f"[INFO] {msg}")

def show_ram(stage):
    process = psutil.Process(os.getpid())

    ram_mb = (
        process.memory_info().rss
        / 1024
        / 1024
    )
    print("\n" + "=" * 60)
    print(f"[RAM] {stage}")
    print(f"PID : {os.getpid()}")
    print(f"RAM : {ram_mb:.2f} MB")
    print("=" * 60)

# =====================================================
# SUPABASE
# =====================================================
@st.cache_resource
def get_supabase():
    log_step("Loading Supabase")
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        supabase = create_client(
            url,
            key
        )
        log_step("Supabase loaded")
        return supabase

    except Exception as e:
        log_step(
            f"Supabase Error: {e}"
        )
        return None

# =====================================================
# MAIN RESOURCES
# =====================================================
@st.cache_resource
def load_resources():
    print("LOAD RESOURCES EXECUTED")
    log_step("START LOAD RESOURCES")
    # -------------------------------------------------
    # NLTK
    # -------------------------------------------------
    log_step("Checking NLTK punkt")
    try:
        nltk.data.find(
            "tokenizers/punkt"
        )
    except LookupError:
        log_step(
            "Downloading punkt"
        )
        nltk.download(
            "punkt"
        )

    try:
        nltk.data.find(
            "tokenizers/punkt_tab"
        )
    except LookupError:
        log_step(
            "Downloading punkt_tab"
        )
        nltk.download(
            "punkt_tab"
        )
    show_ram("START")

    # -------------------------------------------------
    # KNOWLEDGE BASE
    # -------------------------------------------------
    log_step(
        "Loading knowledge-base.json"
    )
    df = pd.read_json(
        "knowledge-base.json"
    )
    show_ram("AFTER JSON")
    log_step(
        f"Knowledge base loaded: {len(df)} rows"
    )

    # -------------------------------------------------
    # FAISS
    # -------------------------------------------------
    log_step(
        "Loading FAISS index"
    )
    faiss_index = faiss.read_index(
        "faissDown.index"
    )
    show_ram(
        "AFTER FAISS"
    )
    log_step(
        f"FAISS loaded total vectors={faiss_index.ntotal}"
    )

    # -------------------------------------------------
    # RETRIEVAL MODEL
    # -------------------------------------------------
    log_step(
        "Loading SentenceTransformer"
    )
    retrieval_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    show_ram(
        "AFTER RETRIEVAL MODEL"
    )
    log_step(
        "SentenceTransformer loaded"
    )

    # -------------------------------------------------
    # TOKENIZER
    # -------------------------------------------------
    log_step(
        "Loading tokenizer"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "gekesa/rte"
    )
    show_ram(
        "AFTER TOKENIZER"
    )
    log_step(
        "Tokenizer loaded"
    )

    # -------------------------------------------------
    # RTE MODEL
    # -------------------------------------------------
    log_step(
        "Loading RTE model"
    )
    rte_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            "gekesa/rte"
        )
    )
    rte_model.eval()
    show_ram(
        "AFTER RTE MODEL"
    )
    log_step(
        "RTE model loaded"
    )
    show_ram(
        "FINISHED"
    )
    log_step(
        "LOAD RESOURCES FINISHED"
    )
    return {
        "df": df,
        "faiss_index": faiss_index,
        "retrieval_model": retrieval_model,
        "tokenizer": tokenizer,
        "rte_model": rte_model
    }