import streamlit as st
import openai
import requests
import os
import io
import time
from groq import Groq
from pydub import AudioSegment

st.set_page_config(page_title="Ελληνική Μεταγραφή Ήχου", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d0f14; color: #e8e3d8; }
section[data-testid="stSidebar"] { background-color: #13161e; border-right: 1px solid #2a2d38; }
h1 { font-size: 2.4rem !important; font-weight: 800 !important; letter-spacing: -1px; color: #e8e3d8 !important; }
h2 { font-size: 1.4rem !important; font-weight: 700 !important; color: #b8b2a6 !important; }
.accent { color: #c8f55a; }
[data-testid="stFileUploader"] { border: 2px dashed #2e3240 !important; border-radius: 12px !important; background: #13161e !important; }
[data-testid="stFileUploader"]:hover { border-color: #c8f55a !important; }
.stButton > button { background: #c8f55a !important; color: #0d0f14 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1rem !important; border: none !important; border-radius: 8px !important; padding: 0.6rem 2rem !important; }
.stButton > button:hover { opacity: 0.85 !important; }
.stTextArea textarea { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.88rem !important; background: #13161e !important; color: #e8e3d8 !important; border: 1px solid #2a2d38 !important; border-radius: 10px !important; line-height: 1.7 !important; }
.stProgress > div > div { background: #c8f55a !important; }
[data-testid="stMetric"] { background: #13161e; border: 1px solid #2a2d38; border-radius: 10px; padding: 1rem 1.2rem; }
[data-testid="stMetricValue"] { color: #c8f55a !important; font-size: 1.6rem !important; font-weight: 700 !important; }
hr { border-color: #2a2d38 !important; }
.stDownloadButton > button { background: transparent !important; color: #c8f55a !important; border: 1.5px solid #c8f55a !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
CHUNK_DURATION_MS  = 10 * 60 * 1000
MAX_BYTES          = 24 * 1024 * 1024

CLEANUP_SYSTEM = """Είσαι ειδικός διορθωτής ελληνικών κειμένων.
Σου δίνεται αυτόματη μεταγραφή από ελληνικό ηχητικό αρχείο.
Κάνε τα εξής:
1. Αφαίρεσε watermarks/artifacts χωρίς νόημα (π.χ. "AUTHORWAVE", "Υπότιτλοι" κ.λπ.)
2. Διόρθωσε λάθη από παρακοή, βάσει πλαισίου
3. Βελτίωσε στίξη και παραγράφους
4. ΜΗΝ αλλάξεις το νόημα - αν δεν είσαι σίγουρος, άφησε ως έχει
5. Επέστρεψε ΜΟΝΟ το διορθωμένο κείμενο, χωρίς σχόλια."""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Keys")

    el_key   = st.text_input("ElevenLabs API Key",  value=st.secrets.get("ELEVENLABS_API_KEY", ""), type="password", placeholder="xi-…")
    groq_key = st.text_input("Groq API Key ⚡",      value=st.secrets.get("GROQ_API_KEY", ""),       type="password", placeholder="gsk_…")
    oai_key  = st.text_input("OpenAI API Key",       value=st.secrets.get("OPENAI_API_KEY", ""),     type="password", placeholder="sk-…")

    st.markdown("---")
    st.markdown("## 🧹 2ο Πέρασμα LLM")

    # Use checkbox instead of toggle for wider Streamlit version compatibility
    do_cleanup = st.checkbox("Αυτόματη διόρθωση κειμένου", value=True)

    # Always define these variables — avoids NameError if do_cleanup is False
    cleanup_engine  = "Groq llama-3.3-70b ⚡"
    custom_context  = ""

    if do_cleanup:
        cleanup_engine = st.radio(
            "Engine διόρθωσης",
            ["Groq llama-3.3-70b ⚡ (γρήγορο)", "OpenAI GPT-4o (ακριβέστερο)"],
        )
        custom_context = st.text_area(
            "Ειδικοί όροι (προαιρετικό)",
            placeholder="π.χ. Στοά, Τέκτων, Σεβάσμιος Διδάσκαλος, Επόπτης...",
            height=80,
        )

    st.markdown("---")
    st.markdown(
        "<small style='font-family:\"IBM Plex Mono\",monospace;color:#555;'>"
        "Fallback: ElevenLabs → Groq → OpenAI</small>",
        unsafe_allow_html=True,
    )

# ── Audio helpers ──────────────────────────────────────────────────────────────

def format_duration(ms):
    s = ms // 1000
    return f"{s // 60}m {s % 60}s"

def compress_to_mp3(audio, bitrate="64k"):
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()

def split_audio(audio, chunk_ms=CHUNK_DURATION_MS):
    return [audio[s:s+chunk_ms] for s in range(0, len(audio), chunk_ms)]

def safe_mp3(chunk):
    data = compress_to_mp3(chunk, "64k")
    if len(data) > MAX_BYTES:
        data = compress_to_mp3(chunk, "32k")
    return data

# ── Transcription backends ─────────────────────────────────────────────────────

def transcribe_elevenlabs(audio_bytes, filename, api_key):
    headers = {"xi-api-key": api_key}
    files   = {"audio": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    data    = {"model_id": "scribe_v1", "language_code": "el"}
    resp = requests.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=600)
    resp.raise_for_status()
    return resp.json().get("text", "")

def transcribe_groq_chunks(audio, api_key, prog):
    client = Groq(api_key=api_key)
    chunks = split_audio(audio)
    n, parts = len(chunks), []
    for i, chunk in enumerate(chunks):
        prog.progress(int(i/n*100), text=f"⚡ Groq — chunk {i+1}/{n}…")
        mp3 = safe_mp3(chunk)
        r = client.audio.transcriptions.create(file=(f"c{i}.mp3", mp3), model="whisper-large-v3", language="el", response_format="text")
        parts.append(r.strip())
        time.sleep(0.2)
    prog.progress(100, text="Groq done ✅")
    return " ".join(parts)

def transcribe_openai_chunks(audio, api_key, prog):
    client = openai.OpenAI(api_key=api_key)
    chunks = split_audio(audio)
    n, parts = len(chunks), []
    for i, chunk in enumerate(chunks):
        prog.progress(int(i/n*100), text=f"🟡 OpenAI — chunk {i+1}/{n}…")
        mp3 = safe_mp3(chunk)
        buf = io.BytesIO(mp3); buf.name = f"c{i}.mp3"
        r = client.audio.transcriptions.create(model="whisper-1", file=buf, language="el", response_format="text")
        parts.append(r.strip())
        time.sleep(0.3)
    prog.progress(100, text="OpenAI done ✅")
    return " ".join(parts)

# ── LLM cleanup ────────────────────────────────────────────────────────────────

def _build_system(context):
    s = CLEANUP_SYSTEM
    if context:
        s += f"\n\nΕιδικοί όροι:\n{context}"
    return s

def _split_words(text, size=3000):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def llm_cleanup_groq(text, api_key, context=""):
    client = Groq(api_key=api_key)
    sys = _build_system(context)
    parts = []
    for chunk in _split_words(text):
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":sys},{"role":"user","content":f"Διόρθωσε:\n\n{chunk}"}],
            max_tokens=4096, temperature=0.2)
        parts.append(r.choices[0].message.content.strip())
    return "\n\n".join(parts)

def llm_cleanup_openai(text, api_key, context=""):
    client = openai.OpenAI(api_key=api_key)
    sys = _build_system(context)
    parts = []
    for chunk in _split_words(text):
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":sys},{"role":"user","content":f"Διόρθωσε:\n\n{chunk}"}],
            max_tokens=4096, temperature=0.2)
        parts.append(r.choices[0].message.content.strip())
    return "\n\n".join(parts)

# ── SRT ────────────────────────────────────────────────────────────────────────

def _ts(s):
    h,r=divmod(s,3600); m,sc=divmod(r,60)
    return f"{h:02d}:{m:02d}:{sc:02d},000"

def make_srt(text):
    words=text.split(); lines,block,t=[],[],0
    for i,w in enumerate(words):
        block.append(w)
        if len(block)>=20 or i==len(words)-1:
            lines.append(f"{len(lines)+1}\n{_ts(t)} --> {_ts(t+10)}\n{' '.join(block)}\n")
            block,t=[],t+10
    return "\n".join(lines)

# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown("<h1>🎙️ Greek Audio <span class='accent'>Transcriber</span></h1>", unsafe_allow_html=True)
st.markdown("<h2>Upload a recording → get your Greek transcript.</h2>", unsafe_allow_html=True)
st.markdown("---")

uploaded = st.file_uploader("Drop your audio file here", type=["wav","mp3","m4a","ogg"])

if uploaded:
    raw_bytes = uploaded.read()
    c1,c2,c3 = st.columns(3)
    c1.metric("File size", f"{len(raw_bytes)/1024/1024:.1f} MB")
    with st.spinner("Reading audio…"):
        audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
    c2.metric("Duration", format_duration(len(audio)))
    c3.metric("Sample rate", f"{audio.frame_rate} Hz")
    st.markdown("---")

    if st.button("▶ Start Transcription", use_container_width=True):

        if not el_key and not groq_key and not oai_key:
            st.error("Προσθέστε τουλάχιστον ένα API key στο sidebar.")
            st.stop()

        raw_transcript = ""
        method_used = ""

        if el_key and not raw_transcript:
            try:
                st.info("🔵 ElevenLabs Scribe…")
                prog = st.progress(0, text="Uploading…")
                raw_transcript = transcribe_elevenlabs(raw_bytes, uploaded.name, el_key)
                method_used = "ElevenLabs Scribe v1"
                prog.progress(100, text="✅")
            except Exception as e:
                st.warning(f"ElevenLabs failed: {e}")

        if groq_key and not raw_transcript:
            try:
                n = max(1, len(audio)//CHUNK_DURATION_MS+1)
                st.info(f"⚡ Groq Whisper ({n} chunks)…")
                prog = st.progress(0)
                raw_transcript = transcribe_groq_chunks(audio, groq_key, prog)
                method_used = f"Groq Whisper large-v3 ({n} chunk{'s' if n>1 else ''})"
            except Exception as e:
                st.warning(f"Groq failed: {e}")

        if oai_key and not raw_transcript:
            try:
                n = max(1, len(audio)//CHUNK_DURATION_MS+1)
                st.info(f"🟡 OpenAI Whisper ({n} chunks)…")
                prog = st.progress(0)
                raw_transcript = transcribe_openai_chunks(audio, oai_key, prog)
                method_used = f"OpenAI Whisper-1 ({n} chunk{'s' if n>1 else ''})"
            except Exception as e:
                st.error(f"Όλα απέτυχαν: {e}")
                st.stop()

        if not raw_transcript:
            st.error("Δεν επιστράφηκε μεταγραφή.")
            st.stop()

        # ── 2nd pass ──────────────────────────────────────────────────────────
        cleaned_transcript = None

        if do_cleanup:
            use_groq = "Groq" in cleanup_engine
            key_for_cleanup = groq_key if use_groq else oai_key
            if not key_for_cleanup:
                st.warning("⚠️ Δεν υπάρχει key για LLM cleanup — παρακάμπτεται.")
            else:
                try:
                    engine_name = "Groq llama-3.3-70b" if use_groq else "GPT-4o"
                    st.info(f"🧹 2ο Πέρασμα με {engine_name}…")
                    prog2 = st.progress(0, text="Επεξεργασία…")
                    if use_groq:
                        cleaned_transcript = llm_cleanup_groq(raw_transcript, groq_key, custom_context)
                    else:
                        cleaned_transcript = llm_cleanup_openai(raw_transcript, oai_key, custom_context)
                    prog2.progress(100, text="Διόρθωση ολοκληρώθηκε ✅")
                except Exception as e:
                    st.warning(f"LLM cleanup απέτυχε: {e}")

        # ── Results ────────────────────────────────────────────────────────────
        label = method_used + (" + LLM διόρθωση" if cleaned_transcript else "")
        st.success(f"✅ {label}")

        final = cleaned_transcript or raw_transcript
        cc1,cc2 = st.columns(2)
        cc1.metric("Λέξεις", f"{len(final.split()):,}")
        cc2.metric("Χαρακτήρες", f"{len(final):,}")

        if cleaned_transcript:
            tab1, tab2 = st.tabs(["✅ Διορθωμένο", "📄 Αρχικό (raw)"])
            with tab1:
                st.text_area(" ", cleaned_transcript, height=420, label_visibility="collapsed", key="t1")
            with tab2:
                st.text_area(" ", raw_transcript, height=420, label_visibility="collapsed", key="t2")
        else:
            st.text_area(" ", raw_transcript, height=420, label_visibility="collapsed", key="t0")

        base = os.path.splitext(uploaded.name)[0]
        st.markdown("### 💾 Download")
        d1,d2,d3 = st.columns(3)
        with d1:
            st.download_button("⬇ Διορθωμένο .TXT", data=final.encode("utf-8"), file_name=f"{base}_transcript.txt", mime="text/plain", use_container_width=True)
        with d2:
            st.download_button("⬇ Αρχικό .TXT", data=raw_transcript.encode("utf-8"), file_name=f"{base}_raw.txt", mime="text/plain", use_container_width=True)
        with d3:
            st.download_button("⬇ .SRT", data=make_srt(final).encode("utf-8"), file_name=f"{base}.srt", mime="text/plain", use_container_width=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;color:#333;'>
        <div style='font-size:4rem'>🎙️</div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:0.9rem;margin-top:1rem;color:#555;'>
            Waiting for audio file…<br>WAV · MP3 · M4A · OGG
        </div>
    </div>""", unsafe_allow_html=True)
