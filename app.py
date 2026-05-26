import streamlit as st
import openai
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

CHUNK_DURATION_S = 10 * 60

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
    st.caption("Χρησιμοποίησε personal keys (όχι project keys)")

    groq_key   = st.text_input("Groq API Key ⚡",            value=st.secrets.get("GROQ_API_KEY", ""),        type="password", placeholder="gsk_…")
    oai_key    = st.text_input("OpenAI API Key",              value=st.secrets.get("OPENAI_API_KEY", ""),      type="password", placeholder="sk-… (όχι sk-proj-)")
    claude_key = st.text_input("Anthropic (Claude) Key 🤖",  value=st.secrets.get("ANTHROPIC_API_KEY", ""),   type="password", placeholder="sk-ant-…")

    st.markdown("---")
    st.markdown("## 🎙️ Μεταγραφή")
    transcribe_engine = st.radio(
        "Engine μεταγραφής",
        ["Groq Whisper ⚡ (γρήγορο)", "OpenAI Whisper (καλύτερη ποιότητα)"],
    )

    st.markdown("---")
    st.markdown("## 🧹 2ο Πέρασμα LLM")
    do_cleanup = st.checkbox("Αυτόματη διόρθωση κειμένου", value=True)

    cleanup_engine = "Claude Sonnet 🤖"
    custom_context = ""
    if do_cleanup:
        cleanup_engine = st.radio(
            "Engine διόρθωσης",
            ["Claude Sonnet 🤖 (καλύτερο, συνιστάται)", "Groq llama-3.3-70b ⚡ (γρήγορο)", "OpenAI GPT-4o ✨ (αν έχεις key)"],
        )
        custom_context = st.text_area(
            "Ειδικοί όροι (προαιρετικό)",
            placeholder="π.χ. Στοά, Τέκτων, Σεβάσμιος Διδάσκαλος...",
            height=80,
        )

    st.markdown("---")
    st.markdown(
        "<small style='color:#444;font-family:\"IBM Plex Mono\",monospace;'>"
        "⚠️ OpenAI: χρειάζεσαι personal key<br>"
        "(platform.openai.com → API keys)<br>"
        "όχι project key (sk-proj-...)</small>",
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
            subprocess.run(
                ['ffmpeg', '-y', '-ss', str(i*chunk_s), '-t', str(chunk_s),
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

def transcribe_groq_chunks(raw_bytes, filename, api_key, prog):
    client = Groq(api_key=api_key)
    chunks = split_bytes_into_chunks(raw_bytes, filename)
    n, parts = len(chunks), []
    for i, chunk in enumerate(chunks):
        prog.progress(int(i/n*100), text=f"⚡ Groq Whisper — chunk {i+1}/{n}…")
        buf = io.BytesIO(chunk); buf.name = f"chunk_{i}.mp3"
        r = client.audio.transcriptions.create(
            file=buf, model="whisper-large-v3", language="el", response_format="text")
        parts.append(r.strip())
        time.sleep(0.2)
    prog.progress(100, text="Groq Whisper done ✅")
    return " ".join(parts)

def transcribe_openai_chunks(raw_bytes, filename, api_key, prog):
    client = openai.OpenAI(api_key=api_key)
    chunks = split_bytes_into_chunks(raw_bytes, filename)
    n, parts = len(chunks), []
    for i, chunk in enumerate(chunks):
        prog.progress(int(i/n*100), text=f"🟡 OpenAI Whisper — chunk {i+1}/{n}…")
        buf = io.BytesIO(chunk); buf.name = f"chunk_{i}.mp3"
        r = client.audio.transcriptions.create(
            model="whisper-1", file=buf, language="el", response_format="text")
        parts.append(r.strip())
        time.sleep(0.3)
    prog.progress(100, text="OpenAI Whisper done ✅")
    return " ".join(parts)

# ── LLM Cleanup ────────────────────────────────────────────────────────────────

def _build_system(context):
    return CLEANUP_SYSTEM + (f"\n\nΕιδικοί όροι:\n{context}" if context else "")

def _split_words(text, size=2000):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def llm_cleanup_openai(text, api_key, context=""):
    client = openai.OpenAI(api_key=api_key)
    sys_p  = _build_system(context)
    parts  = []
    for chunk in _split_words(text):
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":sys_p},
                      {"role":"user","content":f"Διόρθωσε:\n\n{chunk}"}],
            max_tokens=4096, temperature=0.2)
        parts.append(r.choices[0].message.content.strip())
    return "\n\n".join(parts)

def llm_cleanup_claude(text, api_key, context=""):
    client = anthropic.Anthropic(api_key=api_key)
    sys_p  = _build_system(context)
    parts  = []
    for chunk in _split_words(text):
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096, system=sys_p,
            messages=[{"role":"user","content":f"Διόρθωσε:\n\n{chunk}"}])
        parts.append(r.content[0].text.strip())
    return "\n\n".join(parts)

def llm_cleanup_groq(text, api_key, context=""):
    client = Groq(api_key=api_key)
    sys_p  = _build_system(context)
    parts  = []
    chunks = _split_words(text, size=1200)
    for i, chunk in enumerate(chunks):
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":sys_p},
                      {"role":"user","content":f"Διόρθωσε:\n\n{chunk}"}],
            max_tokens=2048, temperature=0.2)
        parts.append(r.choices[0].message.content.strip())
        if i < len(chunks)-1:
            time.sleep(5)
    return "\n\n".join(parts)

# ── SRT ────────────────────────────────────────────────────────────────────────

def _ts(s):
    h, r = divmod(s, 3600); m, sc = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sc:02d},000"

def make_srt(text):
    words = text.split(); lines, block, t = [], [], 0
    for i, w in enumerate(words):
        block.append(w)
        if len(block) >= 20 or i == len(words)-1:
            lines.append(f"{len(lines)+1}\n{_ts(t)} --> {_ts(t+10)}\n{' '.join(block)}\n")
            block, t = [], t+10
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

        use_groq_asr   = "Groq" in transcribe_engine
        use_openai_asr = "OpenAI" in transcribe_engine

        if use_groq_asr and not groq_key:
            st.error("Προσθέστε Groq API key.")
            st.stop()
        if use_openai_asr and not oai_key:
            st.error("Προσθέστε OpenAI API key (personal, όχι project key).")
            st.stop()

        raw_transcript = ""
        method_used    = ""

        # Transcription
        try:
            if use_groq_asr:
                st.info("⚡ Groq Whisper large-v3…")
                prog = st.progress(0)
                raw_transcript = transcribe_groq_chunks(raw_bytes, uploaded.name, groq_key, prog)
                method_used = "Groq Whisper large-v3"
            else:
                st.info("🟡 OpenAI Whisper-1…")
                prog = st.progress(0)
                raw_transcript = transcribe_openai_chunks(raw_bytes, uploaded.name, oai_key, prog)
                method_used = "OpenAI Whisper-1"
        except Exception as e:
            st.error(f"Μεταγραφή απέτυχε: {e}")
            st.stop()

        if not raw_transcript:
            st.error("Δεν επιστράφηκε μεταγραφή.")
            st.stop()

        # 2nd pass LLM cleanup
        cleaned_transcript = None
        if do_cleanup:
            use_oai_c    = "GPT" in cleanup_engine
            use_claude_c = "Claude" in cleanup_engine
            use_groq_c   = "Groq" in cleanup_engine

            key_map = {"oai": oai_key, "claude": claude_key, "groq": groq_key}
            key_c   = oai_key if use_oai_c else (claude_key if use_claude_c else groq_key)
            label_c = "GPT-4o" if use_oai_c else ("Claude Sonnet" if use_claude_c else "Groq llama")

            if not key_c:
                st.warning(f"⚠️ Δεν υπάρχει {label_c} key — παρακάμπτεται.")
            else:
                try:
                    st.info(f"🧹 2ο Πέρασμα με {label_c}…")
                    prog2 = st.progress(0, text="Επεξεργασία…")
                    if use_oai_c:
                        cleaned_transcript = llm_cleanup_openai(raw_transcript, oai_key, custom_context)
                    elif use_claude_c:
                        cleaned_transcript = llm_cleanup_claude(raw_transcript, claude_key, custom_context)
                    else:
                        cleaned_transcript = llm_cleanup_groq(raw_transcript, groq_key, custom_context)
                    prog2.progress(100, text="Διόρθωση ολοκληρώθηκε ✅")
                except Exception as e:
                    st.warning(f"LLM cleanup απέτυχε: {e}")

        label = method_used + (" + LLM διόρθωση" if cleaned_transcript else "")
        st.success(f"✅ {label}")

        final = cleaned_transcript or raw_transcript
        cc1, cc2 = st.columns(2)
        cc1.metric("Λέξεις",     f"{len(final.split()):,}")
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
