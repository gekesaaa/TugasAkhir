# =====================================================
# UI ANALISIS KOMPARASI: RAG+RTE VS QWEN 
# DENGAN KALKULASI COSINE SIMILARITY SECARA REAL-TIME
# =====================================================

# =====================================================
# IMPORT LIBRARY
# =====================================================
import streamlit as st
import numpy as np
import time
from groq import Groq
from utils.resources import load_resources
import re


# =====================================================
# LOAD CSS
# =====================================================
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Perbandingan Hasil Jawaban",
    layout="wide"
)


# =====================================================
# LOAD RESOURCES & LOGIC MANAGEMENT
# =====================================================
resources = load_resources()
retrieval_model = resources["retrieval_model"]


# =====================================================
# GROQ (Qwen) API SYSTEM
# =====================================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=GROQ_API_KEY
)

def generate_qwen_content(
    prompt,
    temperature=0.4,
    model="qwen/qwen3-32b"
):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    )

    return response.choices[0].message.content.strip()


def clean_qwen_response(text):
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    ).strip()

# =====================================================
# RESULT STATE
# =====================================================
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.set_page_config(layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.write("")
    if st.button(
        "⬅ Kembali ke Chat QA",
        use_container_width=True
    ):
        st.switch_page(
            "pages/chat-full.py"
        )

# =====================================================
# QA & SIMILARITY FUNCTIONS
# =====================================================
def calculate_cosine_similarity(text1, text2):
    try:
        vec1 = retrieval_model.encode([text1])
        vec2 = retrieval_model.encode([text2])
        sim = np.dot(vec1, vec2.T) / (
            np.linalg.norm(vec1) *
            np.linalg.norm(vec2)
        )
        return float(sim[0][0])
    except:
        return 0.0


# =====================================================
# CSS
# =====================================================
load_css("styles/compare.css")

# =====================================================
# HEADER
# =====================================================
if (
    st.session_state.last_result
    and
    "similarity" in st.session_state.last_result
):
    score_sim = st.session_state.last_result["similarity"]

    similarity_display = (
        f"🎯 Cosine Similarity: "
        f"<b style='color:#0a84ff; font-size:1.15rem;'>"
        f"{score_sim:.4f}"
        f"</b>"
    )
else:
    similarity_display = (
        "🎯 Cosine Similarity: <b>0.0000</b>"
    )

# Render Header
st.markdown(
    f'<div class="custom-header-container">'
    f'<div class="header-title-text">'
    f'Mental Health Assistant — Analisis Perbandingan'
    f'</div>'
    f'<div style="color: #ffffff; font-size: 0.95rem;">'
    f'{similarity_display}'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)

# =====================================================
# STATUS PLACEHOLDER (DI BAWAH HEADER)
# =====================================================
status_placeholder = st.empty()

# =====================================================
# LOAD HASIL DARI SISTEM UTAMA
# =====================================================
result = st.session_state.get(
    "last_result",
    None
)

# =====================================================
# GENERATE QWEN + SIMILARITY
# =====================================================
if (
    result is not None
    and (
        "qwen" not in result
        or
        "similarity" not in result
    )
):
    status_placeholder.info(
    "🤖 Sedang Proses Perhitungan Kesamaan"
    )

    question = result["question"]
    answer_predict = result["predict"]

    qwen_prompt = f"""
    Bertindaklah sebagai psikolog klinis ahli.

    Berikan jawaban:
    - empatik
    - profesional
    - suportif
    - menenangkan
    - singkat namun jelas
    - terstruktur
    - maksimal 150 kata

    Pertanyaan:
    {question}
    """

    try:

        answer_qwen = generate_qwen_content(
            qwen_prompt,
            temperature=0.4
        )

        answer_qwen = clean_qwen_response(
            answer_qwen
        )

    except Exception as e:

        answer_qwen = (
            f"Error Qwen API: {str(e)}"
        )

    similarity = calculate_cosine_similarity(
        answer_predict,
        answer_qwen
    )

    st.session_state.last_result.update({
        "qwen": answer_qwen,
        "similarity": similarity
    })

    status_placeholder.empty()

    st.rerun()


# =====================================================
# AREA WORKSPACE UTAMA
# =====================================================
col_predict, col_qwen = st.columns(2, gap="large")


# =====================================================
# KOLOM PREDICT
# =====================================================
with col_predict:

    st.markdown(
        '''
        <div class="column-title-label label-predict">
        Hasil Sistem
        </div>
        ''',
        unsafe_allow_html=True
    )

    result = st.session_state.last_result
    with st.container(border=True):
        if result is None:
            st.markdown(
                "<center style='color:#636366;'>"
                "Silakan kirim keluhan."
                "</center>",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div style='
                color:#8e8e93;
                font-size:0.85rem;
                margin-bottom:8px;
                '>
                <b>User:</b> {result['question']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style='
                background-color:#2b2b2c;
                padding:2px;
                border-radius:8px;
                margin-bottom:12px;
                '>
                """,
                unsafe_allow_html=True
            )

            st.write(result["predict"])
            st.markdown("</div>", unsafe_allow_html=True)
            if result["url"]:
                st.markdown(
                    f"""
                    <div style='margin-top:10px;'>
                    🔗 <a href='{result["url"]}'
                    target='_blank'
                    style='
                    color:#34c759;
                    text-decoration:none;
                    '>
                    Rujukan Dokumen
                    </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =====================================================
# KOLOM QWEN
# =====================================================
with col_qwen:

    st.markdown(
        '''
        <div class="column-title-label label-gemini">
        Hasil QWEN
        </div>
        ''',
        unsafe_allow_html=True
    )

    result = st.session_state.last_result

    with st.container(border=True):
        if result is None:
            st.markdown(
                "<center style='color:#636366;'>"
                "Silakan kirim keluhan."
                "</center>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='
                color:#8e8e93;
                font-size:0.85rem;
                margin-bottom:8px;
                '>
                <b>User:</b> {result['question']}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style='
                background-color:#1a233a;
                padding:2px;
                border-radius:8px;
                border-left:3px solid #0a84ff;
                margin-bottom:12px;
                '>
                """,
                unsafe_allow_html=True
            )
            st.write(result["qwen"])
            st.markdown("</div>", unsafe_allow_html=True)