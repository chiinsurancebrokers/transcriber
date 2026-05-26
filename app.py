import streamlit as st
import openai
import requests
import os
import io
import time
from groq import Groq
from pydub import AudioSegment

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ελληνική Μεταγραφή Ήχου",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d0f14; color: #e8e3d8; }
section[data-testid="stSidebar"] { background-color: #13161e; border-right: 1px solid #2a2d38; }
h1 { font-size: 2.4rem !important; font-weight: 800 !important; letter-spacing: -1px; color: #e8e3d8 !important; }
h2 { font-size: 1.4rem !important; font-weight: 700 !important; color: #b8b2a6 !important; }
h3 { font-size: 1.1rem !important; color: #9b9490 !important; }
.accent { color: #c8f55a; }
[data-testid="stFileUploader"] { border: 2px dashed #2e3240 !important; border-radius: 12px !important; background: #13161e !important; }
[data-testid="stFileUploader"]:hover { border-color: #c8f55a !important; }
.stButton > button { background: #c8f55a !important; color: #0d0f14 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1rem !important; border: none !important; border-radius: 8px !important; padding: 0.6rem 2rem !important; }
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }
.stTextArea textarea { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.88rem !important; background: #13161e !important; color: #e8e3d8 !important; border: 1px solid #2a2d38 !important; border-radius: 10px !important; line-height: 1.7 !important; }
.stProgress > div > div { background: #c8f55a !important; }
.stAlert { border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important; }
[data-testid="stMetric"] { background: #13161e; border: 1px solid #2a2d38; border-radius: 10px; padding: 1rem 1.2rem; }
[data-testid="stMetricValue"] { color: #c8f55a !important; font-size: 1.6rem !important; font-weight: 700 !important; }
.stRadio label { color: #b8b2a6 !important; }
hr { border-color: #2a2d38 !important; }
.stDownloadButton > button { background: transparent !important; color: #c8f55a !important; border: 1.5px solid #c8f55a !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; border-radius: 8px !important; }
.stDownloadButton > button:hover { background: #c8f55a22 !important; }
.stTabs [data-baseweb="tab"] { color: #888 !important; font-family: 'IBM Plex Mono', monospace !important; }
.stTabs [aria-selected="true"] { color: #c8f55a !important; border-bottom-color: #c8f55a !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
CHUNK_DURATION_MS  = 10 * 60 * 1000
MAX_BYTES          = 24 * 1024 * 1024

# LLM cleanup prompt
CLEANUP_SYSTEM = """Είσαι ειδικός διορθωτής ελληνικών κειμένων. 
Σου δίνεται μια αυτόματη μεταγραφή (transcript) από ελληνικό ηχητικό αρχείο.
Η μεταγραφή έχει λάθη τυπικά για αναγνώριση φωνής.

Κάνε τα εξής:
1. Αφαίρεσε ΠΛΗΡΩΣ οποιαδήποτε watermarks, artifacts ή επαναλαμβανόμενες φράσεις χωρίς νόημα (π.χ. "AUTHORWAVE", "Υπότιτλοι", κλπ.)
2. Διόρθωσε λάθη στις λέξεις που προέκυψαν από παρακοή (π.χ. "φτέχτον" → "Ελεύθερος Τέκτων", "αξιωθήτε" → ό,τι ταιριάζει στο πλαίσιο)
3. Βελτίωσε στίξη και παραγράφους για αναγνωσιμότητα
4. ΜΕΓΑΛΗ ΠΡΟΣΟΧΗ: ΜΗΝ αλλάξεις το νόημα. Αν δεν είσαι σίγουρος για μια λέξη, άφησέ τη ως έχει.
5. Επέστρεψε ΜΟΝΟ το διορθωμένο κείμενο, χωρίς σχόλια ή εξηγήσεις."""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Keys")
    st.caption("Στην ανάπτυξη, χρησιμοποιήστε Streamlit Secrets.")

    el_key = st.text_input("ElevenLabs API Key", value=st.secrets.get("ELEVENLABS_API_KEY", ""), type="password", placeholder="xi-…")
    groq_key = st.text_input("Groq API Key ⚡", value=st.secrets.get("GROQ_API_KEY", ""), type="password", placeholder="gsk_…")
    oai_key = st.text_input("OpenAI API Key", value=st.secrets.get("OPENAI_API_KEY", ""), type="password", placeholder="sk-…")

    st.markdown("---")
    st.markdown("## 🧹 2ο Πέρασμα (LLM)")
    do_cleanup = st.toggle("Αυτόματη διόρθωση με LLM", value=True, help="Καθαρίζει artifacts, διορθώνει λέξεις, βελτιώνει στίξη")

    if do_cleanup:
        cleanup_engine = st.radio(
            "Engine διόρθωσης",
            ["Groq llama-3.3-70b ⚡ (γρήγορο)", "OpenAI GPT-4o (ακριβέστερο)"],
            help="Χρησιμοποιείται μόνο για το 2ο πέρασμα"
        )
        custom_context = st.text_area(
            "Ειδικοί όροι / context (προαιρετικό)",
            placeholder="π.χ. Τεκτονική ορολογία: Στοά, Τέκτων, Σεβάσμιος Διδάσκαλος...",
            height=80,
            help="Βοηθά το LLM να διορθώσει εξειδικευμένους όρους"
        )

    st.markdown("---")
    st.markdown("## ⚙️ Μεταγραφή")
    st.markdown(
        "<small style='font-family:\"IBM Plex Mono\",monospace; color:#555;'>"
        "1 → ElevenLabs (full file)<br>"
        "2 → Groq Whisper ⚡ (chunked)<br>"
        "3 → OpenAI Whisper (chunked)</small>",
        unsafe_allow_html=True,
    )

# ── Helpers ────────────────────────────────────────────────────────────────────

def format_duration(ms):
    s = ms // 1000
    return f"{s // 60}m {s % 60}s"

def compress_to_mp3(audio, bitrate="64k"):
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()

def split_audio(audio, chunk_ms=CHUNK_DURATION_MS):
    return [audio[s: s + chunk_ms] for s in range(0, len(audio), chunk_ms)]

def safe_mp3(chunk):
    data = compress_to_mp3(chunk, "64k")
    if len(data) > MAX_BYTES:
        data = compress_to_mp3(chunk, "32k")
    return data

# ── Transcription ──────────────────────────────────────────────────────────────

def transcribe_elevenlabs(audio_bytes, filename, api_key):
    headers = {"xi-api-key": api_key}
    files   = {"audio": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    data    = {"model_id": "scribe_v1", "language_code": "el"}
    resp = requests.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=600)
    resp.raise_for_status()
    return resp.json().get("text", "")

def transcribe_groq_chunks(audio, api_key, progress_bar):
    client = Groq(api_key=api_key)
    chunks = split_audio(audio)
    n, parts = len(chunks), []
    for i, chunk in enumerate(chunks):
        progress_bar.progress(int((i / n) * 100), text=f"⚡ Groq Whisper — chunk {i+1} / {n}…")
        mp3 = safe_mp3(chunk)
        result = client.audio.transcriptions.create(
            file=(f"chunk_{i:03d}.mp3", mp3),
            model="whisper-large-v3",
            language="el",
            response_format="text",
        )
        parts.append(result.strip())
        time.sleep(0.2)
    progress_bar.progress(100, text="Groq Whisper done ✅")
    return " ".join(parts)

def transcribe_openai_chunks(audio, api_key, progress_bar):
    client = openai.OpenAI(api_key=api_key)
    chunks = split_audio(audio)
    n, parts = len(chunks), []
    for i, chunk in enumerate(chunks):
        progress_bar.progress(int((i / n) * 100), text=f"🟡 OpenAI Whisper — chunk {i+1} / {n}…")
        mp3 = safe_mp3(chunk)
        buf = io.BytesIO(mp3); buf.name = f"chunk_{i:03d}.mp3"
        result = client.audio.transcriptions.create(model="whisper-1", file=buf, language="el", response_format="text")
        parts.append(result.strip())
        time.sleep(0.3)
    progress_bar.progress(100, text="OpenAI Whisper done ✅")
    return " ".join(parts)

# ── LLM Cleanup ────────────────────────────────────────────────────────────────

def llm_cleanup_groq(raw_text, api_key, context=""):
    client = Groq(api_key=api_key)
    system = CLEANUP_SYSTEM
    if context:
        system += f"\n\nΕιδικοί όροι που μπορεί να εμφανιστούν:\n{context}"

    # Split into chunks of ~3000 words to stay within context
    words = raw_text.split()
    chunk_size = 3000
    word_chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
    cleaned_parts = []

    for i, wc in enumerate(word_chunks):
        chunk_text = " ".join(wc)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Διόρθωσε αυτό το κείμενο:\n\n{chunk_text}"}
            ],
            max_tokens=4096,
            temperature=0.2,
        )
        cleaned_parts.append(response.choices[0].message.content.strip())

    return "\n\n".join(cleaned_parts)

def llm_cleanup_openai(raw_text, api_key, context=""):
    client = openai.OpenAI(api_key=api_key)
    system = CLEANUP_SYSTEM
    if context:
        system += f"\n\nΕιδικοί όροι που μπορεί να εμφανιστούν:\n{context}"

    words = raw_text.split()
    chunk_size = 3000
    word_chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
    cleaned_parts = []

    for i, wc in enumerate(word_chunks):
        chunk_text = " ".join(wc)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Διόρθωσε αυτό το κείμενο:\n\n{chunk_text}"}
            ],
            max_tokens=4096,
            temperature=0.2,
        )
        cleaned_parts.append(response.choices[0].message.content.strip())

    return "\n\n".join(cleaned_parts)

# ── SRT ────────────────────────────────────────────────────────────────────────

def make_srt(transcript):
    words = transcript.split()
    lines, block, t = [], [], 0
    for i, word in enumerate(words):
        block.append(word)
        if len(block) >= 20 or i == len(words) - 1:
            end = t + 10
            lines.append(f"{len(lines)+1}\n{_ts(t)} --> {_ts(end)}\n{' '.join(block)}\n")
            block, t = [], end
    return "\n".join(lines)

def _ts(s):
    h, r = divmod(s, 3600); m, sc = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sc:02d},000"

# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown("<h1>🎙️ Greek Audio <span class='accent'>Transcriber</span></h1>", unsafe_allow_html=True)
st.markdown("<h2>Upload a WAV recording → get your Greek transcript in seconds.</h2>", unsafe_allow_html=True)
st.markdown("---")

uploaded = st.file_uploader("Drop your WAV file here", type=["wav", "mp3", "m4a", "ogg"])

if uploaded:
    raw_bytes = uploaded.read()
    file_mb   = len(raw_bytes) / (1024 * 1024)

    col1, col2, col3 = st.columns(3)
    col1.metric("File size", f"{file_mb:.1f} MB")
    with st.spinner("Reading audio…"):
        audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
    col2.metric("Duration", format_duration(len(audio)))
    col3.metric("Sample rate", f"{audio.frame_rate} Hz")

    st.markdown("---")

    if st.button("▶ Start Transcription", use_container_width=True):

        if not el_key and not groq_key and not oai_key:
            st.error("Προσθέστε τουλάχιστον ένα API key στο sidebar.")
            st.stop()

        raw_transcript = ""
        method_used = ""

        # ── 1. ElevenLabs ──────────────────────────────────────────────────────
        if el_key and not raw_transcript:
            try:
                st.info("🔵 ElevenLabs Scribe (full file)…")
                prog = st.progress(0, text="Uploading…")
                raw_transcript = transcribe_elevenlabs(raw_bytes, uploaded.name, el_key)
                method_used = "ElevenLabs Scribe v1"
                prog.progress(100, text="ElevenLabs done ✅")
            except Exception as e:
                st.warning(f"ElevenLabs failed: {e} → Groq…")

        # ── 2. Groq Whisper ────────────────────────────────────────────────────
        if groq_key and not raw_transcript:
            try:
                n = max(1, len(audio) // CHUNK_DURATION_MS + 1)
                st.info(f"⚡ Groq Whisper ({n} chunk(s))…")
                prog = st.progress(0)
                raw_transcript = transcribe_groq_chunks(audio, groq_key, prog)
                method_used = f"Groq Whisper large-v3 ({n} chunk{'s' if n>1 else ''})"
            except Exception as e:
                st.warning(f"Groq failed: {e} → OpenAI…")

        # ── 3. OpenAI Whisper ──────────────────────────────────────────────────
        if oai_key and not raw_transcript:
            try:
                n = max(1, len(audio) // CHUNK_DURATION_MS + 1)
                st.info(f"🟡 OpenAI Whisper ({n} chunk(s))…")
                prog = st.progress(0)
                raw_transcript = transcribe_openai_chunks(audio, oai_key, prog)
                method_used = f"OpenAI Whisper-1 ({n} chunk{'s' if n>1 else ''})"
            except Exception as e:
                st.error(f"Όλα απέτυχαν: {e}")
                st.stop()

        if not raw_transcript:
            st.error("Δεν επιστράφηκε μεταγραφή. Ελέγξτε τα API keys.")
            st.stop()

        # ── 2nd Pass: LLM Cleanup ──────────────────────────────────────────────
        cleaned_transcript = None

        if do_cleanup:
            use_groq_cleanup  = "Groq" in cleanup_engine
            use_openai_cleanup = "OpenAI" in cleanup_engine
            context_text = custom_context if custom_context else ""

            cleanup_key = groq_key if use_groq_cleanup else oai_key
            if not cleanup_key:
                st.warning("⚠️ Δεν υπάρχει key για το LLM cleanup. Παρακάμπτεται.")
            else:
                try:
                    st.info(f"🧹 2ο Πέρασμα: LLM διόρθωση με {'Groq llama-3.3-70b' if use_groq_cleanup else 'GPT-4o'}…")
                    prog2 = st.progress(0, text="Επεξεργασία κειμένου…")

                    if use_groq_cleanup:
                        cleaned_transcript = llm_cleanup_groq(raw_transcript, groq_key, context_text)
                    else:
                        cleaned_transcript = llm_cleanup_openai(raw_transcript, oai_key, context_text)

                    prog2.progress(100, text="Διόρθωση ολοκληρώθηκε ✅")
                except Exception as e:
                    st.warning(f"LLM cleanup failed: {e}. Εμφανίζεται η αρχική μεταγραφή.")

        # ── Results ────────────────────────────────────────────────────────────
        st.success(f"✅ Μεταγραφή: **{method_used}**" + (" + LLM διόρθωση" if cleaned_transcript else ""))

        mc1, mc2 = st.columns(2)
        display = cleaned_transcript or raw_transcript
        mc1.metric("Λέξεις", f"{len(display.split()):,}")
        mc2.metric("Χαρακτήρες", f"{len(display):,}")

        # Tabs: show both versions if cleanup ran
        if cleaned_transcript:
            tab1, tab2 = st.tabs(["✅ Διορθωμένο κείμενο", "📄 Αρχική μεταγραφή (raw)"])
            with tab1:
                st.text_area("", cleaned_transcript, height=420, label_visibility="collapsed", key="clean")
            with tab2:
                st.text_area("", raw_transcript, height=420, label_visibility="collapsed", key="raw")
        else:
            st.markdown("### 📄 Transcript")
            st.text_area("", raw_transcript, height=420, label_visibility="collapsed")

        # Downloads
        st.markdown("### 💾 Download")
        base_name = os.path.splitext(uploaded.name)[0]
        final = cleaned_transcript or raw_transcript

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.download_button("⬇ Διορθωμένο .TXT", data=final.encode("utf-8"),
                file_name=f"{base_name}_transcript.txt", mime="text/plain", use_container_width=True)
        with dc2:
            st.download_button("⬇ Αρχικό .TXT", data=raw_transcript.encode("utf-8"),
                file_name=f"{base_name}_raw.txt", mime="text/plain", use_container_width=True)
        with dc3:
            st.download_button("⬇ .SRT", data=make_srt(final).encode("utf-8"),
                file_name=f"{base_name}_transcript.srt", mime="text/plain", use_container_width=True)

else:
    st.markdown("""
        <div style='text-align:center; padding:3rem 0; color:#333;'>
            <div style='font-size:4rem'>🎙️</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.9rem; margin-top:1rem; color:#555;'>
                Waiting for audio file…<br>WAV · MP3 · M4A · OGG supported
            </div>
        </div>""", unsafe_allow_html=True)
