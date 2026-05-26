import streamlit as st
import openai
import requests
import os
import io
import math
import time
import tempfile
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

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0f14;
    color: #e8e3d8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #2a2d38;
}

/* Headers */
h1 { font-size: 2.4rem !important; font-weight: 800 !important; letter-spacing: -1px; color: #e8e3d8 !important; }
h2 { font-size: 1.4rem !important; font-weight: 700 !important; color: #b8b2a6 !important; }
h3 { font-size: 1.1rem !important; color: #9b9490 !important; }

/* Accent color */
.accent { color: #c8f55a; }

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #2e3240 !important;
    border-radius: 12px !important;
    background: #13161e !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #c8f55a !important;
}

/* Buttons */
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
.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

/* Text area (transcript output) */
.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    background: #13161e !important;
    color: #e8e3d8 !important;
    border: 1px solid #2a2d38 !important;
    border-radius: 10px !important;
    line-height: 1.7 !important;
}

/* Progress / status */
.stProgress > div > div {
    background: #c8f55a !important;
}

/* Info / warning / success boxes */
.stAlert {
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #13161e;
    border: 1px solid #2a2d38;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    color: #c8f55a !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Radio buttons */
.stRadio label { color: #b8b2a6 !important; }

/* Divider */
hr { border-color: #2a2d38 !important; }

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    color: #c8f55a !important;
    border: 1.5px solid #c8f55a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.stDownloadButton > button:hover {
    background: #c8f55a22 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
CHUNK_DURATION_MS  = 10 * 60 * 1000   # 10-minute chunks for OpenAI
OPENAI_MAX_BYTES   = 24 * 1024 * 1024  # 24 MB safety margin (limit is 25 MB)

# ── Sidebar – API keys ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Keys")
    st.caption("Stored only for this session. For permanent storage use Streamlit secrets.")

    el_key = st.text_input(
        "ElevenLabs API Key",
        value=st.secrets.get("ELEVENLABS_API_KEY", ""),
        type="password",
        placeholder="xi-…",
    )
    oai_key = st.text_input(
        "OpenAI API Key",
        value=st.secrets.get("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-…",
    )

    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    export_format = st.radio("Export format", ["Plain text (.txt)", "Subtitles (.srt)"])
    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>🇬🇷 Greek audio transcription<br>"
        "Powered by ElevenLabs Scribe + OpenAI Whisper</small>",
        unsafe_allow_html=True,
    )

# ── Helper functions ───────────────────────────────────────────────────────────

def format_duration(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}m {s % 60}s"


def compress_to_mp3(audio: AudioSegment, bitrate="64k") -> bytes:
    """Export AudioSegment to MP3 bytes."""
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()


def split_audio(audio: AudioSegment, chunk_ms: int = CHUNK_DURATION_MS):
    """Split audio into chunks of chunk_ms milliseconds."""
    chunks = []
    for start in range(0, len(audio), chunk_ms):
        chunks.append(audio[start : start + chunk_ms])
    return chunks


def transcribe_elevenlabs(audio_bytes: bytes, filename: str, api_key: str) -> str:
    """Send full audio to ElevenLabs Scribe (handles large files natively)."""
    headers = {"xi-api-key": api_key}
    files   = {"audio": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    data    = {"model_id": "scribe_v1", "language_code": "el"}

    resp = requests.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=600)
    resp.raise_for_status()
    result = resp.json()
    # ElevenLabs returns {"text": "...", ...}
    return result.get("text", "")


def transcribe_openai_chunk(mp3_bytes: bytes, client: openai.OpenAI, chunk_index: int) -> str:
    """Transcribe a single MP3 chunk with OpenAI Whisper."""
    audio_file = io.BytesIO(mp3_bytes)
    audio_file.name = f"chunk_{chunk_index:03d}.mp3"
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="el",
        response_format="text",
    )
    return result.strip()


def make_srt(transcript: str) -> str:
    """Rough SRT: split transcript into ~10-second subtitle blocks."""
    words = transcript.split()
    lines, block, t = [], [], 0
    wps = 2  # approx words per second in Greek speech
    duration = 10  # seconds per subtitle block

    for i, word in enumerate(words):
        block.append(word)
        if len(block) >= wps * duration or i == len(words) - 1:
            start_s, end_s = t, t + duration
            lines.append(
                f"{len(lines)+1}\n"
                f"{_srt_ts(start_s)} --> {_srt_ts(end_s)}\n"
                f"{' '.join(block)}\n"
            )
            block = []
            t = end_s
    return "\n".join(lines)


def _srt_ts(s: int) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d},000"


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
    help="Works with WAV, MP3, M4A, OGG. Up to 60+ minutes supported.",
)

if uploaded:
    raw_bytes = uploaded.read()
    file_mb   = len(raw_bytes) / (1024 * 1024)

    col1, col2, col3 = st.columns(3)
    col1.metric("File size", f"{file_mb:.1f} MB")

    with st.spinner("Reading audio metadata…"):
        audio = AudioSegment.from_file(io.BytesIO(raw_bytes))

    col2.metric("Duration",  format_duration(len(audio)))
    col3.metric("Channels",  "Stereo" if audio.channels == 2 else "Mono")

    st.markdown("---")

    if st.button("▶ Start Transcription", use_container_width=True):

        # ── Validate keys ──────────────────────────────────────────────────────
        if not el_key and not oai_key:
            st.error("Please enter at least one API key in the sidebar.")
            st.stop()

        transcript = ""
        method_used = ""

        # ── Attempt 1: ElevenLabs ──────────────────────────────────────────────
        if el_key:
            try:
                st.info("🔵 Trying ElevenLabs Scribe (handles the full file at once)…")
                progress = st.progress(0, text="Uploading to ElevenLabs…")

                transcript  = transcribe_elevenlabs(raw_bytes, uploaded.name, el_key)
                method_used = "ElevenLabs Scribe v1"
                progress.progress(100, text="ElevenLabs done ✅")

            except Exception as e:
                st.warning(f"ElevenLabs failed ({e}). Falling back to OpenAI Whisper…")
                transcript = ""

        # ── Attempt 2: OpenAI Whisper (chunked) ────────────────────────────────
        if not transcript and oai_key:
            try:
                st.info("🟡 Using OpenAI Whisper with automatic chunking…")
                client = openai.OpenAI(api_key=oai_key)
                chunks = split_audio(audio)
                n      = len(chunks)
                prog   = st.progress(0, text=f"Processing chunk 1 / {n}…")
                parts  = []

                for i, chunk in enumerate(chunks):
                    prog.progress(int((i / n) * 100), text=f"Transcribing chunk {i+1} / {n}…")
                    mp3_bytes = compress_to_mp3(chunk)

                    # If still too big, re-compress harder
                    if len(mp3_bytes) > OPENAI_MAX_BYTES:
                        mp3_bytes = compress_to_mp3(chunk, bitrate="32k")

                    part = transcribe_openai_chunk(mp3_bytes, client, i)
                    parts.append(part)
                    time.sleep(0.3)  # gentle rate-limit buffer

                transcript  = " ".join(parts)
                method_used = f"OpenAI Whisper-1 ({n} chunk{'s' if n>1 else ''})"
                prog.progress(100, text="OpenAI Whisper done ✅")

            except Exception as e:
                st.error(f"OpenAI Whisper also failed: {e}")
                st.stop()

        # ── Display results ────────────────────────────────────────────────────
        if transcript:
            st.success(f"✅ Transcribed with **{method_used}**")
            word_count = len(transcript.split())
            st.metric("Words transcribed", f"{word_count:,}")

            st.markdown("### 📄 Transcript")
            st.text_area("", transcript, height=400, label_visibility="collapsed")

            # ── Downloads ──────────────────────────────────────────────────────
            st.markdown("### 💾 Download")
            dcol1, dcol2 = st.columns(2)

            with dcol1:
                st.download_button(
                    "⬇ Download as .TXT",
                    data=transcript.encode("utf-8"),
                    file_name=f"{os.path.splitext(uploaded.name)[0]}_transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dcol2:
                srt_data = make_srt(transcript)
                st.download_button(
                    "⬇ Download as .SRT",
                    data=srt_data.encode("utf-8"),
                    file_name=f"{os.path.splitext(uploaded.name)[0]}_transcript.srt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.error("No transcript was returned. Check your API keys and try again.")

else:
    st.markdown(
        """
        <div style='text-align:center; padding: 3rem 0; color:#444;'>
            <div style='font-size:4rem'>🎙️</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.9rem; margin-top:1rem;'>
                Waiting for audio file…<br>
                WAV · MP3 · M4A · OGG supported
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
