# kode yang sedang diusahakan agar dapat memahami konteks percakapan sebelumnya dengan lebih baik. Sehingga jawaban yang diberikan bisa lebih relevan dan tepat sasaran.
# ditambah halaman supaya bisa langsung akses ke compare

# =====================================================
# UI UTAMA SISTEM HYBRID RAG + RTE + GEMINI PARAPHRASE
# =====================================================

# =====================================================
# IMPORT LIBRARY
# =====================================================
import streamlit as st
import faiss
import json
import numpy as np
import nltk
import hashlib
import torch
import pandas as pd
import time

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from google import genai
from google.genai import types

from supabase import create_client


# =====================================================
# LOAD CSS
# =====================================================
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# =====================================================
# AUTH FUNCTIONS
# =====================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register(username, password):

    try:

        supabase.table("users").insert({
            "username": username,
            "password": hash_password(password)
        }).execute()

        return True

    except Exception:
        return False


def login(username, password):

    result = (
        supabase
        .table("users")
        .select("*")
        .eq("username", username)
        .execute()
    )

    if not result.data:
        return "NOT_FOUND"

    user = result.data[0]

    if user["password"] != hash_password(password):
        return "WRONG_PASSWORD"

    return user

# =====================================================
# CONVERSATION FUNCTIONS
# =====================================================
def create_conversation(user_id):

    result = (
        supabase
        .table("conversations")
        .insert({
            "user_id": user_id,
            "title": "Chat Baru"
        })
        .execute()
    )

    return result.data[0]["id"]


def get_conversations(user_id):

    result = (
        supabase
        .table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return result.data


def load_chat(conv_id):

    if conv_id is None:
        return []

    result = (
        supabase
        .table("chat_history")
        .select("*")
        .eq("conversation_id", conv_id)
        .order("created_at")
        .execute()
    )

    rows = result.data

    chat = []

    for r in rows:

        chat.append({
            "role": "user",
            "content": r["question"]
        })

        ans = r["answer"]

        if r["source_url"]:
            ans += f"\n\n🔗 {r['source_url']}"

        chat.append({
            "role": "assistant",
            "content": ans
        })

    return chat


def save_chat(user_id, conv_id, q, a, url):

    (
        supabase
        .table("chat_history")
        .insert({
            "user_id": user_id,
            "conversation_id": conv_id,
            "question": q,
            "answer": a,
            "source_url": url
        })
        .execute()
    )


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Mental Health Assistant",
    layout="wide"
)

hide_streamlit_style = """
<style>

/* Hilangkan navigation multipage Streamlit */
[data-testid="stSidebarNav"] {
    display: none;
}

/* Hilangkan tombol collapse bawaan */
[data-testid="collapsedControl"] {
    display: none;
}

</style>
"""

st.markdown(
    hide_streamlit_style,
    unsafe_allow_html=True
)


# =====================================================
# LOGIN SESSION
# =====================================================
if "user" not in st.session_state:
    st.session_state.user = None

if "active_page" not in st.session_state:
    st.session_state.active_page = "LOGIN"


# =====================================================
# LOGIN UI
# =====================================================
if st.session_state.user is None:
    query_params = st.query_params
    if "page" in query_params:
        st.session_state.active_page = query_params["page"]
        st.query_params.clear()
    load_css("styles/login.css")
    col1, col_center, col3 = st.columns([1, 2, 1])
    with col_center:
        st.write("<br><br>", unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                btn_log = st.button(
                    "LOGIN",
                    use_container_width=True,
                    type="primary" if st.session_state.active_page == "LOGIN" else "secondary"
                )
                if btn_log:
                    st.session_state.active_page = "LOGIN"
                    st.rerun()
            with c2:
                btn_reg = st.button(
                    "DAFTAR",
                    use_container_width=True,
                    type="primary" if st.session_state.active_page == "DAFTAR" else "secondary"
                )
                if btn_reg:
                    st.session_state.active_page = "DAFTAR"
                    st.rerun()

            # LOGIN PAGE
            if st.session_state.active_page == "LOGIN":
                st.write("")
                u_login = st.text_input(
                    "Username",
                    key="login_user",
                    placeholder="Masukan username"
                )
                p_login = st.text_input(
                    "Password",
                    type="password",
                    key="login_pass",
                    placeholder="Masukan password"
                )

                st.write("")
                if st.button(
                    "LOGIN",
                    use_container_width=True,
                    type="primary",
                    key="btn_execute_login"
                ):
                    result = login(u_login, p_login)

                    if result == "NOT_FOUND":
                        st.error("Akun belum terdaftar.")

                    elif result == "WRONG_PASSWORD":
                        st.error("Username atau Password yang Anda masukkan salah.")

                    else:
                        st.session_state.user = result
                        st.success("Login berhasil!")
                        st.rerun()
                
                st.markdown(
                    "<div style='margin-top: 15px; padding-bottom: 15px;'> "
                    "<center style='font-size: 0.9rem; color: #8e8e93;'>"
                    "Belum punya akun? <a href='/?page=DAFTAR' target='_self' style='color: #ff3b30; text-decoration: none; font-weight: bold;'>Daftar Sekarang</a>"
                    "</center>"
                    "</div>", 
                    unsafe_allow_html=True
                )

            # REGISTER PAGE
            else:
                st.write("")
                u_reg = st.text_input(
                    "Username",
                    key="reg_user",
                    placeholder="Masukan username"
                )
                p_reg = st.text_input(
                    "Password",
                    type="password",
                    key="reg_pass",
                    placeholder="Masukan password"
                )

                st.write("")
                if st.button(
                    "DAFTAR",
                    use_container_width=True,
                    type="primary",
                    key="btn_execute_reg"
                ):
                    if not u_reg or not p_reg:
                        st.warning("Username dan Password tidak boleh kosong.")
                    elif register(u_reg, p_reg):
                        st.success("Berhasil daftar!")
                        st.session_state.active_page = "LOGIN"
                        st.rerun()
                    else:
                        st.error("Username sudah terdaftar.")

                st.markdown(
                    "<div style='margin-top: 15px; padding-bottom: 15px;'>"
                    "<center style='font-size: 0.9rem; color: #8e8e93;'>"
                    "Sudah punya akun? <a href='/?page=LOGIN' target='_self' style='color: #ff3b30; text-decoration: none; font-weight: bold;'>Login Sekarang</a>"
                    "</center>"
                    "</div>", 
                    unsafe_allow_html=True
                )

    st.stop()


# =====================================================
# LOAD RESOURCES
# =====================================================
@st.cache_resource
def get_supabase():

    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]

        return create_client(url, key)

    except Exception as e:
        st.error(f"Supabase Error: {e}")
        return None

supabase = get_supabase()

def load_resources():

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    # =========================
    # KNOWLEDGE BASE
    # =========================
    df = pd.read_json("knowledge-base.json")
    embeddings = np.load("embeddingsDown.npy")
    index = faiss.read_index("faissDown.index")

    

    # =========================
    # RETRIEVAL MODEL
    # =========================
    retrieval_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # =========================
    # RTE MODEL
    # =========================
    # rte_model_path = "checkpoint-343355"
    rte_model_path = "model-rte"
    tokenizer = AutoTokenizer.from_pretrained("gekesa/rte")
    rte_model = AutoModelForSequenceClassification.from_pretrained(
        "gekesa/rte"
    )

    rte_model.eval()

    return (
        df,
        embeddings,
        index,
        retrieval_model,
        tokenizer,
        rte_model
    )

(
    df,
    embeddings,
    faiss_index,
    retrieval_model,
    tokenizer,
    rte_model
) = load_resources()

# =====================================================
# GEMINI FALLBACK API SYSTEM
# =====================================================
GEMINI_KEYS = st.secrets["GEMINI_API_KEYS"]

# index key aktif
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
            # jika quota habis / rate limit
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
                # pindah ke key berikutnya
                st.session_state.gemini_key_index = (
                    st.session_state.gemini_key_index + 1
                ) % total_keys

                time.sleep(2)

                continue
            # error lain langsung raise
            raise e

    # semua key gagal
    raise Exception(
        "Semua API Gemini sedang habis quota."
    )


# =====================================================
# LABEL MAP
# =====================================================
label_map = rte_model.config.id2label

keterlibatan_index = [
    k for k, v in label_map.items()
    if v.lower() == "keterlibatan"
][0]

kontradiksi_index = [
    k for k, v in label_map.items()
    if v.lower() == "kontradiksi"
][0]


# =====================================================
# QA FUNCTIONS
# =====================================================
def contextualize_question(chat_history, latest_question):

    if not chat_history:
        return latest_question

    prompt = f"""
    Anda adalah AI yang bertugas memahami konteks percakapan.

    Berdasarkan riwayat percakapan dan pertanyaan terbaru user,
    ubah pertanyaan terbaru menjadi pertanyaan lengkap dan jelas.

    Aturan:
    - Pertahankan konteks percakapan sebelumnya.
    - Jika user memberikan koreksi atau klarifikasi,
    gabungkan dengan topik sebelumnya.
    - Jangan mengubah maksud pertanyaan.
    - HANYA tulis hasil pertanyaan final.
    - Jangan memberi penjelasan tambahan.

    Riwayat:
    {chat_history}

    Pertanyaan:
    {latest_question}
    """

    try:
        return generate_gemini_content(
            prompt,
            temperature=0.1
        )

    except:
        return latest_question


def is_feedback_correction(text):

    feedback_patterns = [
        "bukan itu maksud saya",
        "maksud saya",
        "yang saya tanyakan",
        "kurang tepat",
        "bukan itu",
        "salah",
        "yang benar",
        "yang saya maksud",
        "bukan seperti itu"
    ]

    text = text.lower()

    return any(p in text for p in feedback_patterns)


def rebuild_query_from_feedback(
    previous_question,
    previous_answer,
    feedback_text
):

    prompt = f"""
    Anda adalah AI yang bertugas memahami koreksi user.

    User sebelumnya bertanya:
    "{previous_question}"

    Sistem menjawab:
    "{previous_answer}"

    Tetapi user memberikan koreksi:
    "{feedback_text}"

    Tugas:
    Buat ulang pertanyaan user menjadi
    pertanyaan lengkap dan jelas.

    Aturan:
    - Fokus pada maksud terbaru user
    - Pertahankan konteks utama
    - Jangan menambahkan informasi baru
    - HANYA tampilkan pertanyaan final
    """

    try:

        return generate_gemini_content(
            prompt,
            temperature=0.1
        )

        return res.text.strip()

    except:
        return feedback_text


def generate_hypothesis(question: str) -> str:
    q = question.strip().lower()
    q = q.replace("?", "")

    # Apa itu X
    if q.startswith("apa itu"):
        x = q.replace("apa itu", "").strip()
        return f"{x} adalah"

    # Apa yang dimaksud dengan X
    if "apa yang dimaksud dengan" in q:
        x = q.replace("apa yang dimaksud dengan", "").strip()
        return f"{x} adalah"

    # Apa penyebab X
    if "apa penyebab" in q:
        x = q.replace("apa penyebab", "").strip()
        return f"penyebab {x} adalah"

    # Apa fungsi X
    if "apa fungsi" in q:
        x = q.replace("apa fungsi", "").strip()
        return f"fungsi {x} adalah"

    # fallback (AMAN)
    return q


def compute_rte_score(premise, hypothesis):

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        logits = rte_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)

    keterlibatan = probs[0][keterlibatan_index].item()
    kontradiksi = probs[0][kontradiksi_index].item()

    return keterlibatan - kontradiksi


def get_rte_label(premise, hypothesis):

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        logits = rte_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    label_id = torch.argmax(probs, dim=-1).item()

    return label_map[label_id]


def extract_best_answer(query, chunk_text, window=2):

    sents = sent_tokenize(chunk_text)

    if not sents:
        return chunk_text

    q_emb = retrieval_model.encode([query])
    s_emb = retrieval_model.encode(sents)
    scores = np.dot(s_emb, q_emb.T).squeeze()
    best_idx = int(np.argmax(scores))
    start = max(0, best_idx - window)
    end = min(len(sents), best_idx + window + 1)

    return " ".join(sents[start:end])


def paraphrase_answer(question, answer_text):

    prompt = f"""
    Anda adalah asisten kesehatan mental.

    Tugas Anda adalah merapikan jawaban agar:
    - natural,
    - profesional,
    - mudah dipahami,
    - singkat,
    - tidak mengubah makna,
    - tidak menambahkan informasi baru.

    PENTING:
    - JANGAN mengatakan:
    "berikut parafrase",
    "tentu",
    "jawaban yang telah dirapikan",
    atau kalimat pembuka lainnya.

    Pertanyaan:
    {question}

    Jawaban:
    {answer_text}
    """

    try:

        return generate_gemini_content(
            prompt,
            temperature=0.2
        )

    except:
        return answer_text


def is_mental_health_question(query):

    prompt = f"""
Anda adalah classifier domain.

Tentukan apakah pertanyaan berikut berkaitan dengan kesehatan mental.

Jawab HANYA dengan:
- YA
- TIDAK

Pertanyaan:
{query}
"""

    try:

        result = generate_gemini_content(
            prompt,
            temperature=0
        ).upper()

        return result == "YA"

    except:
        return True

# =====================================================
# SESSION CHAT
# =====================================================
if "current_conv" not in st.session_state:
    st.session_state.current_conv = None

if "chat" not in st.session_state:
    st.session_state.chat = load_chat(
        st.session_state.current_conv
    )


# =====================================================
# LOAD MAIN CSS
# =====================================================
load_css("styles/main.css")


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.write("📂 **Recent Conversations**")

    if st.button(
        "➕ Percakapan baru",
        use_container_width=True
    ):

        st.session_state.current_conv = None
        st.session_state.chat = []
        st.rerun()

    st.write("---")

    convs = get_conversations(
        st.session_state.user["id"]
    )
    if not convs:
        st.caption(
            "<center>Belum ada riwayat</center>",
            unsafe_allow_html=True
        )
    else:
        for c in convs:
            title = c["title"] if c["title"] else "Chat Baru"
            display_title = (
                title[:24] + "..."
                if len(title) > 24
                else title
            )
            is_active = (
                st.session_state.current_conv == c["id"]
            )
            if st.button(
                "💬 " + display_title,
                key=f"sidebar_c_{c['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):

                st.session_state.current_conv = c["id"]
                st.session_state.chat = load_chat(
                    c["id"]
                )
                st.session_state.confirm_logout = False
                st.rerun()

    st.write("<br><br>", unsafe_allow_html=True)
    st.write("---")

    if "confirm_logout" not in st.session_state:
        st.session_state.confirm_logout = False

    if not st.session_state.confirm_logout:
        if st.button("📤 Logout", use_container_width=True, key="sidebar_logout_trigger"):
            st.session_state.confirm_logout = True
            st.rerun()
    else:
        st.markdown("<p style='color:#ff453a; font-size:0.85rem; text-align:center; margin-bottom:5px;'>⚠️ Anda yakin logout?</p>", unsafe_allow_html=True)
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Ya, Keluar", use_container_width=True, key="btn_logout_yes", type="primary"):
                st.session_state.user = None
                st.session_state.chat = []
                st.session_state.current_conv = None
                st.session_state.confirm_logout = False
                st.rerun()
        with col_no:
            if st.button("Batal", use_container_width=True, key="btn_logout_no", type="secondary"):
                st.session_state.confirm_logout = False
                st.rerun()


# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="sticky-header">
    <h1>🧠 Mental Health Assistant</h1>
</div>
""", unsafe_allow_html=True)


# =====================================================
# RENDER CHAT
# =====================================================
if (
    st.session_state.current_conv is None
    and
    not st.session_state.chat
):

    st.markdown(
        f"""
        <div style='text-align:center;
                    margin-top:100px;
                    color:#8e8e93;'>

        <h2>Halo, {st.session_state.user['username']}.</h2>

        <h4>
        Ada keluhan kesehatan mental
        yang ingin kamu ceritakan hari ini?
        </h4>

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    for msg in st.session_state.chat:

        if msg["role"] == "user":

            st.markdown(
                f"""
                <div class="chat-row-user">
                    <div class="bubble-user-item">
                        {msg["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            full_text = msg["content"]
            main_answer = full_text
            markdown_debug_content = ""

            if "⚙️ Debug Info System:" in full_text:
                parts = full_text.split("---")
                main_answer = parts[0].strip()
                markdown_debug_content = (
                    parts[1]
                    .replace("⚙️ Debug Info System:", "")
                    .strip()
                )

            st.markdown(
                f"""
                <div class="chat-row-assistant">
                    <div class="bubble-assistant-container">
                        <div>{main_answer}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if markdown_debug_content:

                with st.expander("⚙️ Debug Info"):
                    st.markdown(markdown_debug_content)

# =====================================================
# TOMBOL ANALISIS HANYA UNTUK JAWABAN TERAKHIR
# =====================================================
            is_last_message = (
                msg == st.session_state.chat[-1]
            )

            if (
                msg["role"] == "assistant"
                and is_last_message
            ):

                if st.button(
                    "🔬 Analisis Perbandingan",
                    key="analyze_last",
                    use_container_width=True
                ):

                    last_question = ""

                    for m in reversed(st.session_state.chat):

                        if m["role"] == "user":
                            last_question = m["content"]
                            break

                    # Ambil jawaban assistant terakhir
                    last_answer = ""

                    for m in reversed(st.session_state.chat):

                        if m["role"] == "assistant":

                            last_answer = m["content"]

                            # buang debug info
                            if "⚙️ Debug Info System:" in last_answer:
                                last_answer = (
                                    last_answer
                                    .split("---")[0]
                                    .strip()
                                )

                            break

                    st.session_state.last_result = {
                        "question": last_question,
                        "predict": last_answer,
                        "url": None
                    }

                    st.switch_page(
                        "pages/perbandingan-hasil.py"
                    )


# =====================================================
# CHAT INPUT
# =====================================================
if question := st.chat_input("Tulis pertanyaan..."):

    # =========================
    # AUTO CREATE CONVERSATION
    # =========================
    if st.session_state.current_conv is None:

        conv_id = create_conversation(
            st.session_state.user["id"]
        )

        conn.execute(
            """
            UPDATE conversations
            SET title=?
            WHERE id=?
            """,
            (question[:50], conv_id)
        )

        conn.commit()

        st.session_state.current_conv = conv_id

    # =========================
    # USER BUBBLE
    # =========================
    st.markdown(
        f"""
        <div class="chat-row-user">
            <div class="bubble-user-item">
                {question}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.session_state.chat.append({
        "role": "user",
        "content": question
    })

    # =========================
    # MAIN QA PIPELINE
    # =========================
    with st.spinner("🧠 Mencari jawaban terbaik..."):
        total_start_time = time.time()

        history_text = ""

        for m in st.session_state.chat[-6:]:
            role = "User" if m["role"] == "user" else "Assistant"
            content = m["content"]
            # hapus debug info dari history
            if "⚙️ Debug Info System:" in content:
                content = content.split("---")[0].strip()

            history_text += f"{role}: {content}\n"

        # ======================================
        # FEEDBACK-AWARE CONTEXTUALIZATION
        # ======================================
        if (
            is_feedback_correction(question)
            and
            len(st.session_state.chat) >= 2
        ):
            
            previous_question = ""
            previous_answer = ""

            # Cari pertanyaan terakhir user
            for msg in reversed(st.session_state.chat[:-1]):

                if msg["role"] == "user":
                    previous_question = msg["content"]
                    break

            # Cari jawaban terakhir assistant
            for msg in reversed(st.session_state.chat[:-1]):

                if msg["role"] == "assistant":
                    previous_answer = msg["content"]
                    break

            query = rebuild_query_from_feedback(
                previous_question,
                previous_answer,
                question
            )
        else:

            context_start = time.time()

            query = contextualize_question(
                history_text,
                question
            )

            context_time = time.time() - context_start

        # =========================
        # TOKEN INFO
        # =========================
        question_tokens = len(question.split())
        query_tokens = len(query.split())


        print("\n" + "="*70)
        print("DEBUG QUERY")
        print("="*70)

        print(f"\nPertanyaan Asli:")
        print(question)

        print(f"\nQuery Setelah Contextualization:")
        print(query)


        # =========================
        # DOMAIN VALIDATION
        # =========================
        is_valid_domain = is_mental_health_question(query)

        if not is_valid_domain:

            final_answer = (
                f"Maaf, pertanyaan mengenai "
                f"'{question}' tidak termasuk "
                f"dalam topik kesehatan mental "
                f"yang tersedia pada basis data."
            )

            full_assistant_response = final_answer

            save_chat(
                st.session_state.user["id"],
                st.session_state.current_conv,
                question,
                full_assistant_response,
                None
            )

            st.session_state.chat = load_chat(
                st.session_state.current_conv
            )

            st.rerun()

        # =========================
        # QUERY EMBEDDING
        # =========================
        query_emb = retrieval_model.encode(
            [query],
            normalize_embeddings=True
        )

        # =========================
        # HYPOTHESIS
        # =========================
        hypothesis = generate_hypothesis(query)

        print(f"\nHypothesis RTE:")
        print(hypothesis)

        # =========================
        # RETRIEVAL
        # =========================
        top_k_retrieval = 15
        top_k_final = 5

        retrieval_start = time.time()

        faiss_scores, indices = faiss_index.search(
            query_emb,
            k=top_k_retrieval
        )

        retrieval_time = time.time() - retrieval_start

        print("\n" + "="*70)
        print("HASIL RETRIEVAL AWAL (FAISS)")
        print("="*70)

        results = []

        retrieval_rank_map = {}

        rte_start = time.time()

        for i, idx in enumerate(indices[0]):

            chunk_text = df.iloc[idx]["text"]
            faiss_score = faiss_scores[0][i]

            retrieval_rank_map[df.iloc[idx]["chunk_id"]] = i + 1

            print(f"\nRetrieval Rank #{i+1}")
            print(f"Chunk ID     : {df.iloc[idx]['chunk_id']}")
            print(f"FAISS Score  : {faiss_score:.4f}")
            print(f"Preview Text :")
            print(chunk_text[:300])

            rte_time = time.time() - rte_start

            # =========================
            # RTE SCORE
            # =========================
            rte_score = compute_rte_score(
                chunk_text,
                hypothesis
            )

            rte_label = get_rte_label(
                chunk_text,
                hypothesis
            )

            # =========================
            # NORMALIZATION
            # =========================
            faiss_norm = (faiss_score + 1) / 2
            rte_norm = (rte_score + 1) / 2

            # =========================
            # HYBRID SCORE
            # =========================
            final_score = (
                0.6 * faiss_norm
                +
                0.4 * rte_norm
            )

            results.append({
                "chunk_id": df.iloc[idx]["chunk_id"],
                "text_result": chunk_text,
                "link": df.iloc[idx].get("link", None),
                "faiss_score": faiss_score,
                "rte_score": rte_score,
                "rte_label": rte_label,
                "final_score": final_score
            })


            print("\n" + "-"*70)
            print("HASIL PER CHUNK SEBELUM SORTING")
            print("-"*70)

            print(f"Chunk ID        : {df.iloc[idx]['chunk_id']}")
            print(f"FAISS Score (α) : {faiss_score:.4f}")
            print(f"RTE Score (β)   : {rte_score:.4f}")
            print(f"RTE Label       : {rte_label}")
            print(f"FAISS Normal    : {faiss_norm:.4f}")
            print(f"RTE Normal      : {rte_norm:.4f}")
            print(f"Hybrid Score    : {final_score:.4f}")

        # =========================
        # SORT RESULTS
        # =========================
        results = sorted(
            results,
            key=lambda x: x["final_score"],
            reverse=True
        )

        final_results = results[:top_k_final]

        print("\n" + "="*70)
        print("HASIL SETELAH RE-RANKING")
        print("="*70)

        for rank, item in enumerate(results, start=1):

            old_rank = retrieval_rank_map.get(item["chunk_id"], "-")

            print(f"\nFinal Rank #{rank}")
            print(f"Previous Retrieval Rank : {old_rank}")
            print(f"Chunk ID                : {item['chunk_id']}")
            print(f"Final Score             : {item['final_score']:.4f}")
            print(f"FAISS Score (α)         : {item['faiss_score']:.4f}")
            print(f"RTE Score (β)           : {item['rte_score']:.4f}")
            print(f"RTE Label               : {item['rte_label']}")

            print("Preview:")
            print(item['text_result'][:300])

            print("-"*70)

        # =========================
        # BEST RESULT
        # =========================
        best_result = final_results[0]
        raw_text = best_result["text_result"]
        url = best_result.get("link", None)

        # =========================
        # ANSWER EXTRACTION
        # =========================
        extract_start = time.time()

        extracted_answer = extract_best_answer(
            question,
            raw_text
        )

        extract_time = time.time() - extract_start

        # =========================
        # GEMINI PARAPHRASE
        # =========================
        paraphrase_start = time.time()

        final_answer = paraphrase_answer(
            question,
            extracted_answer
        )

        st.session_state.last_result = {
            "question": question,
            "predict": final_answer,
            "url": url
}

        paraphrase_time = time.time() - paraphrase_start

        # =========================
        # DEBUG INFO
        # =========================
        total_time = time.time() - total_start_time
        debug_text = ""

        debug_text += "**⏱️ Processing Time**\n"
        debug_text += f"- Retrieval Time (FAISS): {retrieval_time:.4f} detik\n"
        debug_text += f"- RTE Re-ranking Time: {rte_time:.4f} detik\n"
        debug_text += f"- Answer Extraction Time: {extract_time:.4f} detik\n"
        debug_text += f"- Total Response Time {total_time:.4f} detik\n\n"

        debug_text += " **🧠 Token Information**\n"
        debug_text += f"- Original Question Tokens: {question_tokens}\n"
        debug_text += f"- Contextualized Query Tokens: {query_tokens}\n\n"

        

        for rank, item in enumerate(
            final_results,
            start=1
        ):

            debug_text += f"**Rank** {rank}\n"

            debug_text += f"- Chunk ID: {item['chunk_id']}\n"
            debug_text += f"- Final Score: {item['final_score']:.4f}\n"
            debug_text += f"- FAISS Score (α) : {item['faiss_score']:.4f}\n"
            debug_text += f"- RTE Score (β) : {item['rte_score']:.4f}\n"
            debug_text += f"- Label: {item['rte_label']}\n\n"

            debug_text += "Preview:\n"
            debug_text += f"{item['text_result'][:200]}...\n\n"


        full_assistant_response = final_answer

        if url:
            full_assistant_response += (
                f"\n\n🔗 "
                f"[Baca informasi lengkapnya di sini]({url})"
            )

        full_assistant_response += (
            f"\n\n---\n"
            f"⚙️ Debug Info System:\n\n"
            f"{debug_text}"
)

        # =========================
        # SAVE CHAT
        # =========================
        save_chat(
            st.session_state.user["id"],
            st.session_state.current_conv,
            question,
            full_assistant_response,
            None
        )

        # =========================
        # REFRESH CHAT
        # =========================
        st.session_state.chat = load_chat(
            st.session_state.current_conv
        )

        st.rerun()