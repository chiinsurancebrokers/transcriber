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

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background-color: #0d0f14; color: #e8e3d8; }

section[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #2a2d38;
}

h1 { font-size: 2.4rem !important; font-weight: 800 !important; letter-spacing: -1px; color: #e8e3d8 !important; }
h2 { font-size: 1.4rem !important; font-weight: 700 !important; color: #b8b2a6 !important; }
h3 { font-size: 1.1rem !important; color: #9b9490 !important; }

.accent { color: #c8f55a; }

[data-testid="stFileUploader"] {
    border: 2px dashed #2e3240 !important;
    border-radius: 12px !important;
    background: #13161e !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #c8f55a !important; }

.stButton > button {
    background: #c8f55a !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    background: #13161e !important;
    color: #e8e3d8 !important;
    border: 1px solid #2a2d38 !important;
    border-radius: 10px !important;
    line-height: 1.7 !important;
}

.stProgress > div > div { background: #c8f55a !important; }

.stAlert {
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

[data-testid="stMetric"] {
    background: #13161e;
    border: 1px solid #2a2d38;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { color: #c8f55a !important; font-size: 1.6rem !important; font-weight: 700 !important; }

.stRadio label { color: #b8b2a6 !important; }
hr { border-color: #2a2d38 !important; }

.stDownloadButton > button {
    background: transparent !important;
    color: #c8f55a !important;
    border: 1.5px solid #c8f55a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.stDownloadButton > button:hover { background: #c8f55a22 !important; }

/* Badge pills for method indicator */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-el  { background: #1a3a6b; color: #6ab0f5; }
.badge-groq { background: #2a1a4a; color: #c89eff; }
.badge-oai { background: #1a3a2a; color: #7ecca8; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
CHUNK_DURATION_MS  = 10 * 60 * 1000   # 10-min chunks
MAX_BYTES          = 24 * 1024 * 1024  # 24 MB (safe for both Groq & OpenAI)

# ── Sidebar – API keys ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Keys")
    st.caption("Stored for this session only. Use Streamlit Secrets for permanent storage.")

    el_key = st.text_input(
        "ElevenLabs API Key",
        value=st.secrets.get("ELEVENLABS_API_KEY", ""),
        type="password", placeholder="xi-…",
    )
    groq_key = st.text_input(
        "Groq API Key ⚡",
        value=st.secrets.get("GROQ_API_KEY", ""),
        type="password", placeholder="gsk_…",
    )
    oai_key = st.text_input(
        "OpenAI API Key",
        value=st.secrets.get("OPENAI_API_KEY", ""),
        type="password", placeholder="sk-…",
    )

    st.markdown("---")
    st.markdown("## ⚙️ Settings")

    st.markdown("**Fallback order**")
    st.markdown(
        "<small style='font-family:\"IBM Plex Mono\",monospace; color:#666;'>"
        "1 → ElevenLabs (full file)<br>"
        "2 → Groq Whisper ⚡ (chunked)<br>"
        "3 → OpenAI Whisper (chunked)</small>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#444'>🇬🇷 Greek audio transcription<br>"
        "ElevenLabs Scribe · Groq Whisper · OpenAI Whisper</small>",
        unsafe_allow_html=True,
    )

# ── Helper functions ───────────────────────────────────────────────────────────

def format_duration(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}m {s % 60}s"


def compress_to_mp3(audio: AudioSegment, bitrate="64k") -> bytes:
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()


def split_audio(audio: AudioSegment, chunk_ms: int = CHUNK_DURATION_MS):
    return [audio[s: s + chunk_ms] for s in range(0, len(audio), chunk_ms)]


def safe_mp3(chunk: AudioSegment) -> bytes:
    """Compress chunk to MP3, re-compress harder if over limit."""
    data = compress_to_mp3(chunk, "64k")
    if len(data) > MAX_BYTES:
        data = compress_to_mp3(chunk, "32k")
    return data


# ── Transcription backends ─────────────────────────────────────────────────────

def transcribe_elevenlabs(audio_bytes: bytes, filename: str, api_key: str) -> str:
    """ElevenLabs Scribe v1 — sends full file, no chunking needed."""
    headers = {"xi-api-key": api_key}
    files   = {"audio": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    data    = {"model_id": "scribe_v1", "language_code": "el"}
    resp = requests.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=600)
    resp.raise_for_status()
    return resp.json().get("text", "")


def transcribe_groq_chunks(audio: AudioSegment, api_key: str, progress_bar) -> str:
    """Groq Whisper large-v3 — chunked, very fast."""
    client = Groq(api_key=api_key)
    chunks = split_audio(audio)
    n      = len(chunks)
    parts  = []

    for i, chunk in enumerate(chunks):
        progress_bar.progress(int((i / n) * 100), text=f"⚡ Groq — chunk {i+1} / {n}…")
        mp3 = safe_mp3(chunk)
        result = client.audio.transcriptions.create(
            file=(f"chunk_{i:03d}.mp3", mp3),
            model="whisper-large-v3",
            language="el",
            response_format="text",
        )
        parts.append(result.strip())
        time.sleep(0.2)

    progress_bar.progress(100, text="Groq done ✅")
    return " ".join(parts)


def transcribe_openai_chunks(audio: AudioSegment, api_key: str, progress_bar) -> str:
    """OpenAI Whisper-1 — chunked fallback."""
    client = openai.OpenAI(api_key=api_key)
    chunks = split_audio(audio)
    n      = len(chunks)
    parts  = []

    for i, chunk in enumerate(chunks):
        progress_bar.progress(int((i / n) * 100), text=f"🟡 OpenAI — chunk {i+1} / {n}…")
        mp3 = safe_mp3(chunk)
        buf = io.BytesIO(mp3)
        buf.name = f"chunk_{i:03d}.mp3"
        result = client.audio.transcriptions.create(
            model="whisper-1", file=buf, language="el", response_format="text",
        )
        parts.append(result.strip())
        time.sleep(0.3)

    progress_bar.progress(100, text="OpenAI done ✅")
    return " ".join(parts)


# ── SRT helper ─────────────────────────────────────────────────────────────────

def make_srt(transcript: str) -> str:
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
st.markdown(
    "<h1>🎙️ Greek Audio <span class='accent'>Transcriber</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h2>Upload a WAV recording → get your Greek transcript in seconds.</h2>",
    unsafe_allow_html=True,
)
st.markdown("---")

uploaded = st.file_uploader(
    "Drop your WAV file here",
    type=["wav", "mp3", "m4a", "ogg"],
    help="WAV · MP3 · M4A · OGG — up to 60+ minutes supported",
)

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
            st.error("Please enter at least one API key in the sidebar.")
            st.stop()

        transcript  = ""
        method_used = ""

        # ── 1. ElevenLabs ──────────────────────────────────────────────────────
        if el_key and not transcript:
            try:
                st.info("🔵 Trying ElevenLabs Scribe (full file, no chunking)…")
                prog = st.progress(0, text="Uploading to ElevenLabs…")
                transcript  = transcribe_elevenlabs(raw_bytes, uploaded.name, el_key)
                method_used = "ElevenLabs Scribe v1"
                prog.progress(100, text="ElevenLabs done ✅")
            except Exception as e:
                st.warning(f"ElevenLabs failed: {e}. Trying Groq…")
                transcript = ""

        # ── 2. Groq Whisper ────────────────────────────────────────────────────
        if groq_key and not transcript:
            try:
                n_chunks = max(1, len(audio) // CHUNK_DURATION_MS + 1)
                st.info(f"⚡ Using Groq Whisper large-v3 ({n_chunks} chunk(s) @ 10 min each)…")
                prog = st.progress(0, text="Starting Groq…")
                transcript  = transcribe_groq_chunks(audio, groq_key, prog)
                method_used = f"Groq Whisper large-v3 ({n_chunks} chunk{'s' if n_chunks>1 else ''})"
            except Exception as e:
                st.warning(f"Groq failed: {e}. Trying OpenAI…")
                transcript = ""

        # ── 3. OpenAI Whisper ──────────────────────────────────────────────────
        if oai_key and not transcript:
            try:
                n_chunks = max(1, len(audio) // CHUNK_DURATION_MS + 1)
                st.info(f"🟡 Using OpenAI Whisper-1 ({n_chunks} chunk(s) @ 10 min each)…")
                prog = st.progress(0, text="Starting OpenAI…")
                transcript  = transcribe_openai_chunks(audio, oai_key, prog)
                method_used = f"OpenAI Whisper-1 ({n_chunks} chunk{'s' if n_chunks>1 else ''})"
            except Exception as e:
                st.error(f"OpenAI also failed: {e}")
                st.stop()

        # ── Results ────────────────────────────────────────────────────────────
        if transcript:
            st.success(f"✅ Transcribed with **{method_used}**")
            word_count = len(transcript.split())
            char_count = len(transcript)

            mc1, mc2 = st.columns(2)
            mc1.metric("Words", f"{word_count:,}")
            mc2.metric("Characters", f"{char_count:,}")

            st.markdown("### 📄 Transcript")
            st.text_area("", transcript, height=420, label_visibility="collapsed")

            st.markdown("### 💾 Download")
            dc1, dc2 = st.columns(2)
            base_name = os.path.splitext(uploaded.name)[0]

            with dc1:
                st.download_button(
                    "⬇ Download .TXT",
                    data=transcript.encode("utf-8"),
                    file_name=f"{base_name}_transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dc2:
                st.download_button(
                    "⬇ Download .SRT",
                    data=make_srt(transcript).encode("utf-8"),
                    file_name=f"{base_name}_transcript.srt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.error("No transcript returned. Check your API keys and try again.")

else:
    st.markdown(
        """
        <div style='text-align:center; padding:3rem 0; color:#333;'>
            <div style='font-size:4rem'>🎙️</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.9rem; margin-top:1rem; color:#555;'>
                Waiting for audio file…<br>WAV · MP3 · M4A · OGG supported
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
