import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import re
import nltk
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# --- Page config ---
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="magnifying_glass",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f0ede6;
    color: #111111;
}
.stApp { background: #f0ede6; }
.main .block-container { max-width: 500px; padding: 2rem 1.5rem 4rem; }

/* Badge */
.badge-circle {
    width: 68px; height: 68px; border-radius: 50%;
    background: linear-gradient(140deg, #1a1a2e 0%, #0f3460 60%, #533483 100%);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    margin: 0 auto 1.2rem; gap: 2px;
}
.badge-circle .tag {
    font-size: 8px; font-weight: 600; letter-spacing: 0.07em;
    color: rgba(255,255,255,0.85); line-height: 1;
}
.badge-dot { width: 2.5px; height: 2.5px; border-radius: 50%; background: rgba(255,255,255,0.3); }

/* Heading - force Libre Baskerville by overriding ALL Streamlit h1 selectors */
h1, .main-title,
[data-testid="stMarkdownContainer"] h1,
.stMarkdown h1,
div[data-testid="stMarkdownContainer"] h1 {
    font-family: 'Libre Baskerville', Georgia, serif !important;
    font-size: 2rem !important; font-weight: 700 !important; color: #0a0a0a !important;
    text-align: center !important; letter-spacing: -0.01em !important;
    line-height: 1.2 !important; margin-bottom: 0.5rem !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.12) !important;
}
.subtitle {
    font-size: 0.82rem; color: #2c2c2c; font-weight: 500;
    text-align: center; line-height: 1.65; margin-bottom: 1.4rem;
}

/* Reset button - smaller, right-aligned, same style as Analyze */
.reset-btn-row {
    display: flex;
    justify-content: flex-end;
    margin-top: 0.4rem;
    margin-bottom: 0.2rem;
}
.reset-btn-row .stButton > button {
    width: auto !important;
    padding: 0.45rem 1.2rem !important;
    font-size: 0.82rem !important;
    background: #37474f !important;
    border-radius: 10px !important;
    margin-top: 0 !important;
}
.reset-btn-row .stButton > button:hover { background: #546e7a !important; }
.reset-btn-row .stButton > button:active { background: #b71c1c !important; }

/* Stats */
.stats-bar {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 7px; margin-bottom: 1rem;
}
.stat-item { border-radius: 11px; padding: 0.55rem 0.3rem; text-align: center; }
.stat-item:nth-child(1) { background: #e8eaf6; }
.stat-item:nth-child(2) { background: #e8f5e9; }
.stat-item:nth-child(3) { background: #e8eaf6; }
.stat-item:nth-child(4) { background: #fce4ec; }
.stat-value {
    font-family: 'Libre Baskerville', serif;
    font-size: 1rem; font-weight: 700; line-height: 1.1; margin-bottom: 3px;
}
.stat-item:nth-child(1) .stat-value { color: #283593; }
.stat-item:nth-child(2) .stat-value { color: #1b5e20; }
.stat-item:nth-child(3) .stat-value { color: #283593; }
.stat-item:nth-child(4) .stat-value { color: #880e4f; }
.stat-label {
    font-size: 7px; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: #999; line-height: 1.2;
}

/* Tip box */
.tip-box {
    display: flex; align-items: flex-start; gap: 7px;
    background: #fffde7; border: 1px solid #ffe082;
    border-radius: 9px; padding: 0.45rem 0.7rem;
    margin-bottom: 1.2rem;
    font-size: 0.7rem; color: #5a4a00; line-height: 1.5;
}
.tip-icon { font-size: 12px; flex-shrink: 0; margin-top: 1px; }
.tip-box strong { color: #3949ab; font-weight: 600; }

/* Section labels - dark and visible */
.section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #1a1a2e;
    margin-bottom: 0.3rem; margin-top: 0.8rem;
}

/* Input placeholder color */
input[type="text"]::placeholder,
.stTextInput input::placeholder {
    color: #1a1a2e !important;
    opacity: 1 !important;
}
textarea::placeholder {
    color: #1a1a2e !important;
    opacity: 1 !important;
}

/* Input fields */
input[type="text"], .stTextInput input {
    background: #ffffff !important;
    border: 1.5px solid #ddd9d0 !important;
    border-radius: 11px !important; color: #111111 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.87rem !important; padding: 0.6rem 0.85rem !important;
    transition: border-color 0.2s !important;
    caret-color: #1a1a2e !important;
    cursor: text !important;
}
input[type="text"]:focus, .stTextInput input:focus {
    border-color: #5c6bc0 !important;
    box-shadow: 0 0 0 3px rgba(92,107,192,0.11) !important;
    caret-color: #1a1a2e !important;
}
textarea {
    background: #ffffff !important;
    border: 1.5px solid #ddd9d0 !important;
    border-radius: 11px !important; color: #111111 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.87rem !important; line-height: 1.6 !important;
    padding: 0.6rem 0.85rem !important;
    transition: border-color 0.2s !important;
    caret-color: #1a1a2e !important;
    cursor: text !important;
}
textarea:focus {
    border-color: #5c6bc0 !important;
    box-shadow: 0 0 0 3px rgba(92,107,192,0.11) !important;
    caret-color: #1a1a2e !important;
}

/* Analyze button */
.stButton > button {
    width: 100%; background: #212121 !important;
    color: #ffffff !important; border: none !important;
    border-radius: 12px !important; padding: 0.88rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important; cursor: pointer !important;
    transition: background 0.15s !important; margin-top: 0.3rem !important;
}
.stButton > button:hover { background: #333333 !important; }
.stButton > button:active { background: #00796b !important; }

/* Result cards */
.result-real {
    padding: 1rem 1.1rem; border-radius: 14px;
    background: #f1f8e9; border: 1.5px solid #aed581; margin-top: 1rem;
}
.result-fake {
    padding: 1rem 1.1rem; border-radius: 14px;
    background: #fce4ec; border: 1.5px solid #f48fb1; margin-top: 1rem;
}
.result-top { display: flex; align-items: center; gap: 10px; margin-bottom: 0.45rem; }
.result-icon {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 600; flex-shrink: 0;
}
.result-label {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.3rem; font-weight: 700;
}
.result-real .result-label { color: #33691e; }
.result-fake .result-label { color: #880e4f; }
.result-score { font-size: 0.76rem; color: #888; margin-bottom: 0.6rem; }
.confidence-bar-bg {
    height: 5px; border-radius: 100px;
    background: rgba(0,0,0,0.07); overflow: hidden;
}
.confidence-bar-fill-real {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #8bc34a, #4caf50);
}
.confidence-bar-fill-fake {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #e91e63, #f06292);
}

/* Footer */
.footer-note {
    font-size: 0.7rem; color: #bbb; text-align: center;
    margin-top: 1.4rem; line-height: 1.6;
    padding-top: 1rem; border-top: 1px dashed #e0ddd5;
}

/* Streamlit warning box - high visibility orange */
div[data-testid="stAlert"] {
    background-color: #e65100 !important;
    border: 2px solid #bf360c !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
    color: #ffffff !important;
}
div[data-testid="stAlert"] svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
}

/* Also target by alert role */
[role="alert"] {
    background-color: #e65100 !important;
    border: 2px solid #bf360c !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
MAX_WORDS = 10000
MAX_LEN = 200

# --- Load model and tokenizer ---
@st.cache_resource
def load_resources():
    model = load_model("Fake_News_Predictor.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# --- Text preprocessing ---
def clean_text(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_news(text: str, model, tokenizer):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "REAL" if score > 0.5 else "FAKE"
    conf = score if score > 0.5 else 1 - score
    return label, conf, score

# --- Render result card + instant page blink in ONE call (no lag) ---
def render_result(label, pct, raw_score):
    if label == "REAL":
        card_html = f"""
        <div style="padding:1rem 1.1rem;border-radius:14px;background:#f1f8e9;
                    border:1.5px solid #aed581;margin-top:1rem;font-family:sans-serif;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.45rem;">
                <div style="width:32px;height:32px;border-radius:50%;background:#aed581;
                            color:#33691e;display:flex;align-items:center;justify-content:center;
                            font-size:14px;font-weight:600;">&#10003;</div>
                <div style="font-family:'Libre Baskerville',Georgia,serif;
                            font-size:1.3rem;font-weight:700;color:#33691e;">Real News</div>
            </div>
            <div style="font-size:0.76rem;color:#888;margin-bottom:0.6rem;">
                Confidence: {pct}% &nbsp;&middot;&nbsp; Raw score: {raw_score:.4f}
            </div>
            <div style="height:5px;border-radius:100px;background:rgba(0,0,0,0.07);overflow:hidden;">
                <div style="height:100%;width:{pct}%;border-radius:100px;
                            background:linear-gradient(90deg,#8bc34a,#4caf50);"></div>
            </div>
        </div>"""
        blink_colors = "['#f0ede6','#4caf50','#a5d6a7','#4caf50','#f0ede6']"
    else:
        card_html = f"""
        <div style="padding:1rem 1.1rem;border-radius:14px;background:#fce4ec;
                    border:1.5px solid #f48fb1;margin-top:1rem;font-family:sans-serif;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.45rem;">
                <div style="width:32px;height:32px;border-radius:50%;background:#f48fb1;
                            color:#880e4f;display:flex;align-items:center;justify-content:center;
                            font-size:14px;font-weight:600;">&#10005;</div>
                <div style="font-family:'Libre Baskerville',Georgia,serif;
                            font-size:1.3rem;font-weight:700;color:#880e4f;">Fake News</div>
            </div>
            <div style="font-size:0.76rem;color:#888;margin-bottom:0.6rem;">
                Confidence: {pct}% &nbsp;&middot;&nbsp; Raw score: {raw_score:.4f}
            </div>
            <div style="height:5px;border-radius:100px;background:rgba(0,0,0,0.07);overflow:hidden;">
                <div style="height:100%;width:{pct}%;border-radius:100px;
                            background:linear-gradient(90deg,#e91e63,#f06292);"></div>
            </div>
        </div>"""
        blink_colors = "['#f0ede6','#e53935','#ef9a9a','#e53935','#f0ede6']"

    # card + blink in one single components.html — fires simultaneously, zero lag
    components.html(f"""
    {card_html}
    <script>
        (function() {{
            var app = window.parent.document.querySelector('.stApp');
            if (!app) return;
            var colors = {blink_colors};
            var i = 0;
            function step() {{
                if (i >= colors.length) {{
                    app.style.transition = '';
                    app.style.backgroundColor = '#f0ede6';
                    return;
                }}
                app.style.transition = 'background-color 0.35s ease';
                app.style.backgroundColor = colors[i];
                i++;
                setTimeout(step, 350);
            }}
            step();
        }})();
    </script>
    """, height=120)

# --- Session state for reset ---
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# --- UI ---
st.markdown("""
<div class="badge-circle" aria-hidden="true">
  <span class="tag">AI</span>
  <div class="badge-dot"></div>
  <span class="tag">NLP</span>
  <div class="badge-dot"></div>
  <span class="tag">LSTM</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Paste any news article or headline below and our LSTM model will analyse whether it\'s real or fabricated in seconds.</p>', unsafe_allow_html=True)

st.markdown("""
<div class="stats-bar">
  <div class="stat-item"><div class="stat-value">96%</div><div class="stat-label">Accuracy</div></div>
  <div class="stat-item"><div class="stat-value">44K+</div><div class="stat-label">Articles</div></div>
  <div class="stat-item"><div class="stat-value">LSTM</div><div class="stat-label">Model</div></div>
  <div class="stat-item"><div class="stat-value">Binary</div><div class="stat-label">Classification</div></div>
</div>
""", unsafe_allow_html=True)

try:
    model, tokenizer = load_resources()
    resources_ok = True
except Exception as e:
    resources_ok = False
    st.error(f"Could not load model or tokenizer: {e}\n\nMake sure Fake_News_Predictor.h5 and tokenizer.pkl are in the same folder as app.py.")

st.markdown("""
<div class="tip-box">
  <span class="tip-icon">&#128161;</span>
  <span>For best accuracy, provide <strong>both the title and article body</strong>.
  The model was trained on full article text - headlines alone may give less reliable results.</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">News Title / Headline</div>', unsafe_allow_html=True)
title_input = st.text_input(
    label="title_hidden",
    placeholder="Enter news headline...",
    label_visibility="collapsed",
    key=f"title_val_{st.session_state.reset_counter}",
)

st.markdown('<div class="section-label">Article Body / Description</div>', unsafe_allow_html=True)
text_input = st.text_area(
    label="text_hidden",
    placeholder="Paste full news article...",
    height=180,
    label_visibility="collapsed",
    key=f"body_val_{st.session_state.reset_counter}",
)

# Reset button — bottom-right of description box
st.markdown('<div class="reset-btn-row">', unsafe_allow_html=True)
if st.button("Reset", key="reset_btn"):
    st.session_state.reset_counter += 1
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Analyze"):
    if not resources_ok:
        st.warning("Model not loaded. Please check the error above.")
    elif not title_input.strip() and not text_input.strip():
        st.warning("Please enter at least a title or article body.")
    else:
        combined = ""
        if title_input.strip():
            combined += (title_input.strip() + " ") * 3
        if text_input.strip():
            combined += text_input.strip()

        if len(combined.split()) < 5:
            st.warning("Please provide more text for accurate results.")
        else:
            with st.spinner("Analysing..."):
                label, confidence, raw_score = predict_news(combined, model, tokenizer)

            pct = int(confidence * 100)

            if label == "REAL":
                render_result("REAL", pct, raw_score)
            else:
                render_result("FAKE", pct, raw_score)

st.markdown("""
<p class="footer-note">
  <strong>Note:</strong> This model was trained on political news data (2016-2018).
  Results may vary for other domains or recent events.
</p>
""", unsafe_allow_html=True)