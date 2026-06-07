# =====================================================
# UI ANALISIS KOMPARASI: RAG+RTE (PREDICT) VS GEMINI (GROUND TRUTH)
# DENGAN KALKULASI COSINE SIMILARITY SECARA REAL-TIME
# =====================================================

# =====================================================
# IMPORT LIBRARY
# =====================================================
import streamlit as st
import numpy as np
import time

from google import genai
from google.genai import types

from utils.resources import load_resources


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
# GEMINI FALLBACK API SYSTEM
# =====================================================
GEMINI_KEYS = st.secrets["GEMINI_API_KEYS"]
if "gemini_key_index" not in st.session_state:
    st.session_state.gemini_key_index = 0


def get_gemini_client():
    current_index = st.session_state.gemini_key_index
    api_key = GEMINI_KEYS[current_index]
    return genai.Client(api_key=api_key)


def generate_gemini_content(
    prompt,
    temperature=0.1,
    model="gemini-2.5-flash"
):
    total_keys = len(GEMINI_KEYS)
    max_retry = total_keys * 2
    for attempt in range(max_retry):
        try:
            client = get_gemini_client()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                )
            )
            return response.text.strip()

        except Exception as e:
            error_text = str(e).lower()
            if (
                "quota" in error_text
                or
                "429" in error_text
                or
                "resource exhausted" in error_text
                or
                "rate limit" in error_text
                or
                "503" in error_text
                or
                "unavailable" in error_text
                or
                "high demand" in error_text
            ):

                st.session_state.gemini_key_index = (
                    st.session_state.gemini_key_index + 1
                ) % total_keys

                time.sleep(2)

                continue

            raise e

    raise Exception(
        "Semua API Gemini habis quota."
    )

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
# GENERATE GEMINI + SIMILARITY
# =====================================================
if (
    result is not None
    and (
        "gemini" not in result
        or
        "similarity" not in result
    )
):

    # status_placeholder.info(
    # "🤖 Gemini sedang memproses jawaban..."
    # )

    status_placeholder.info(
    "🤖 Sedang Proses Perhitungan Kesamaan"
    )

    question = result["question"]
    answer_predict = result["predict"]

    gemini_prompt = f"""
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

        answer_gemini = generate_gemini_content(
            gemini_prompt,
            temperature=0.4
        )

    except Exception as e:

        answer_gemini = (
            f"Error Gemini API: {str(e)}"
        )

    similarity = calculate_cosine_similarity(
        answer_predict,
        answer_gemini
    )

    st.session_state.last_result.update({
        "gemini": answer_gemini,
        "similarity": similarity
    })

    status_placeholder.empty()

    st.rerun()


# =====================================================
# AREA WORKSPACE UTAMA
# =====================================================
col_predict, col_gemini = st.columns(2, gap="large")


# =====================================================
# KOLOM PREDICT
# =====================================================
with col_predict:

    st.markdown(
        '''
        <div class="column-title-label label-predict">
        Hasil Sistem (Predict)
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
# KOLOM GEMINI
# =====================================================
with col_gemini:

    st.markdown(
        '''
        <div class="column-title-label label-gemini">
        Hasil Gemini (Ground Truth)
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
            st.write(result["gemini"])
            st.markdown("</div>", unsafe_allow_html=True)