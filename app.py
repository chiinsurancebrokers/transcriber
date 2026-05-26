import streamlit as st
import anthropic
import requests
import os
import io
import time
import tempfile
import subprocess
from groq import Groq

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
CHUNK_DURATION_S   = 10 * 60
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
    el_key      = st.text_input("ElevenLabs API Key",  value=st.secrets.get("ELEVENLABS_API_KEY", ""),  type="password", placeholder="xi-…")
    groq_key    = st.text_input("Groq API Key ⚡",      value=st.secrets.get("GROQ_API_KEY", ""),        type="password", placeholder="gsk_…")
    claude_key  = st.text_input("Anthropic (Claude) Key 🤖", value=st.secrets.get("ANTHROPIC_API_KEY", ""), type="password", placeholder="sk-ant-…")

    st.markdown("---")
    st.markdown("## 🧹 2ο Πέρασμα LLM")
    do_cleanup = st.checkbox("Αυτόματη διόρθωση κειμένου", value=True)

    cleanup_engine = "Groq llama-3.3-70b ⚡"
    custom_context = ""
    if do_cleanup:
        cleanup_engine = st.radio(
            "Engine διόρθωσης",
            ["Groq llama-3.3-70b ⚡ (γρήγορο)", "Claude Haiku 🤖 (ακριβέστερο)"],
        )
        custom_context = st.text_area(
            "Ειδικοί όροι (προαιρετικό)",
            placeholder="π.χ. Στοά, Τέκτων, Σεβάσμιος Διδάσκαλος...",
            height=80,
        )

    st.markdown("---")
    st.markdown(
        "<small style='color:#555;font-family:\"IBM Plex Mono\",monospace;'>"
        "Transcription: ElevenLabs → Groq Whisper<br>"
        "Cleanup: Groq llama / Claude</small>",
        unsafe_allow_html=True,
    )

# ── Audio helpers ──────────────────────────────────────────────────────────────

def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

def get_audio_info(raw_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext in ['.wav', '.flac', '.ogg']:
            import soundfile as sf
            with sf.SoundFile(io.BytesIO(raw_bytes)) as f:
                return len(f) / f.samplerate, f.samplerate
        else:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(raw_bytes); tmp_path = tmp.name
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', tmp_path],
                    capture_output=True, text=True, timeout=10)
                return float(result.stdout.strip()), 48000
            except Exception:
                return len(raw_bytes) / (1024*1024) * 2.5 * 60, 48000
            finally:
                os.unlink(tmp_path)
    except Exception:
        return len(raw_bytes) / (1024*1024) * 2.5 * 60, 44100

def split_bytes_into_chunks(raw_bytes, filename, chunk_s=CHUNK_DURATION_S):
    ext = os.path.splitext(filename)[1].lower()
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
            tmp_in.write(raw_bytes); tmp_in_path = tmp_in.name
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', tmp_in_path],
            capture_output=True, text=True, timeout=10)
        total = float(result.stdout.strip())
        n = max(1, int(total / chunk_s) + (1 if total % chunk_s else 0))
        chunks = []
        for i in range(n):
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_out:
                tmp_out_path = tmp_out.name
            subprocess.run(['ffmpeg', '-y', '-ss', str(i*chunk_s), '-t', str(chunk_s),
                '-i', tmp_in_path, '-q:a', '5', '-map', 'a', tmp_out_path],
                capture_output=True, timeout=120)
            with open(tmp_out_path, 'rb') as f:
                data = f.read()
            os.unlink(tmp_out_path)
            if data: chunks.append(data)
        os.unlink(tmp_in_path)
        return chunks or [raw_bytes]
    except Exception:
        return [raw_bytes]

# ── Transcription ──────────────────────────────────────────────────────────────

def transcribe_elevenlabs(raw_bytes, filename, api_key):
    ext  = os.path.splitext(filename)[1].lower()
    mime = {"wav":"audio/wav","mp3":"audio/mpeg","m4a":"audio/mp4","ogg":"audio/ogg"}.get(ext, "audio/wav")
    headers = {"xi-api-key": api_key}
    files   = {"audio": (filename, io.BytesIO(raw_bytes), mime)}
    data    = {"model_id": "scribe_v1", "language_code": "el"}
    resp = requests.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=600)
    resp.raise_for_status()
    return resp.json().get("text", "")

def transcribe_groq_chunks(raw_bytes, filename, api_key, prog):
    client = Groq(api_key=api_key)
    chunks = split_bytes_into_chunks(raw_bytes, filename)
    n, parts = len(chunks), []
    for i, chunk in enumerate(chunks):
        prog.progress(int(i/n*100), text=f"⚡ Groq Whisper — chunk {i+1}/{n}…")
        buf = io.BytesIO(chunk); buf.name = f"chunk_{i}.mp3"
        r = client.audio.transcriptions.create(file=buf, model="whisper-large-v3", language="el", response_format="text")
        parts.append(r.strip())
        time.sleep(0.2)
    prog.progress(100, text="Groq Whisper done ✅")
    return " ".join(parts)

# ── LLM Cleanup ────────────────────────────────────────────────────────────────

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
    sys_prompt = _build_system(context)
    parts = []
    for chunk in _split_words(text):
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":f"Διόρθωσε:\n\n{chunk}"}],
            max_tokens=4096, temperature=0.2)
        parts.append(r.choices[0].message.content.strip())
    return "\n\n".join(parts)

def llm_cleanup_claude(text, api_key, context=""):
    client = anthropic.Anthropic(api_key=api_key)
    sys_prompt = _build_system(context)
    parts = []
    for chunk in _split_words(text):
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=sys_prompt,
            messages=[{"role":"user","content":f"Διόρθωσε:\n\n{chunk}"}])
        parts.append(r.content[0].text.strip())
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
    c1, c2, c3 = st.columns(3)
    c1.metric("File size", f"{len(raw_bytes)/1024/1024:.1f} MB")
    with st.spinner("Reading audio…"):
        duration, samplerate = get_audio_info(raw_bytes, uploaded.name)
    c2.metric("Duration", format_duration(duration))
    c3.metric("Sample rate", f"{samplerate} Hz")
    st.markdown("---")

    if st.button("▶ Start Transcription", use_container_width=True):

        if not el_key and not groq_key:
            st.error("Προσθέστε ElevenLabs ή Groq API key για μεταγραφή.")
            st.stop()

        raw_transcript = ""
        method_used = ""

        # 1. ElevenLabs
        if el_key and not raw_transcript:
            try:
                st.info("🔵 ElevenLabs Scribe (full file)…")
                prog = st.progress(0, text="Uploading…")
                raw_transcript = transcribe_elevenlabs(raw_bytes, uploaded.name, el_key)
                method_used = "ElevenLabs Scribe v1"
                prog.progress(100, text="✅")
            except Exception as e:
                st.warning(f"ElevenLabs failed: {e}")

        # 2. Groq Whisper
        if groq_key and not raw_transcript:
            try:
                st.info("⚡ Groq Whisper large-v3…")
                prog = st.progress(0)
                raw_transcript = transcribe_groq_chunks(raw_bytes, uploaded.name, groq_key, prog)
                method_used = "Groq Whisper large-v3"
            except Exception as e:
                st.error(f"Groq Whisper failed: {e}")
                st.stop()

        if not raw_transcript:
            st.error("Δεν επιστράφηκε μεταγραφή.")
            st.stop()

        # 2nd pass LLM cleanup
        cleaned_transcript = None
        if do_cleanup:
            use_groq_cleanup   = "Groq" in cleanup_engine
            use_claude_cleanup = "Claude" in cleanup_engine
            key_for_cleanup    = groq_key if use_groq_cleanup else claude_key

            if not key_for_cleanup:
                missing = "Groq" if use_groq_cleanup else "Anthropic Claude"
                st.warning(f"⚠️ Δεν υπάρχει {missing} key — παρακάμπτεται το cleanup.")
            else:
                try:
                    engine_label = "Groq llama-3.3-70b" if use_groq_cleanup else "Claude Haiku"
                    st.info(f"🧹 2ο Πέρασμα με {engine_label}…")
                    prog2 = st.progress(0, text="Επεξεργασία…")
                    if use_groq_cleanup:
                        cleaned_transcript = llm_cleanup_groq(raw_transcript, groq_key, custom_context)
                    else:
                        cleaned_transcript = llm_cleanup_claude(raw_transcript, claude_key, custom_context)
                    prog2.progress(100, text="Διόρθωση ολοκληρώθηκε ✅")
                except Exception as e:
                    st.warning(f"LLM cleanup απέτυχε: {e}")

        label = method_used + (" + LLM διόρθωση" if cleaned_transcript else "")
        st.success(f"✅ {label}")

        final = cleaned_transcript or raw_transcript
        cc1, cc2 = st.columns(2)
        cc1.metric("Λέξεις", f"{len(final.split()):,}")
        cc2.metric("Χαρακτήρες", f"{len(final):,}")

        if cleaned_transcript:
            tab1, tab2 = st.tabs(["✅ Διορθωμένο", "📄 Αρχικό (raw)"])
            with tab1:
                st.text_area("Διορθωμένο", cleaned_transcript, height=420, label_visibility="collapsed", key="t1")
            with tab2:
                st.text_area("Αρχικό", raw_transcript, height=420, label_visibility="collapsed", key="t2")
        else:
            st.text_area("Transcript", raw_transcript, height=420, label_visibility="collapsed", key="t0")

        base = os.path.splitext(uploaded.name)[0]
        st.markdown("### 💾 Download")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("⬇ Διορθωμένο .TXT", data=final.encode("utf-8"),
                file_name=f"{base}_transcript.txt", mime="text/plain", use_container_width=True)
        with d2:
            st.download_button("⬇ Αρχικό .TXT", data=raw_transcript.encode("utf-8"),
                file_name=f"{base}_raw.txt", mime="text/plain", use_container_width=True)
        with d3:
            st.download_button("⬇ .SRT", data=make_srt(final).encode("utf-8"),
                file_name=f"{base}.srt", mime="text/plain", use_container_width=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;'>
        <div style='font-size:4rem'>🎙️</div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:0.9rem;margin-top:1rem;color:#555;'>
            Waiting for audio file…<br>WAV · MP3 · M4A · OGG
        </div>
    </div>""", unsafe_allow_html=True)
