import io
import os
import time
import tempfile
import subprocess
from typing import List, Tuple, Optional

import streamlit as st

# Optional SDK imports are handled safely inside functions where possible
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import openai
except Exception:
    openai = None

try:
    import anthropic
except Exception:
    anthropic = None


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ελληνική Μεταγραφή Ήχου",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background-color: #0d0f14;
    color: #e8e3d8;
}

section[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #2a2d38;
}

h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    color: #e8e3d8 !important;
}

h2, h3 {
    color: #b8b2a6 !important;
}

.accent {
    color: #c8f55a;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #2e3240 !important;
    border-radius: 12px !important;
    background: #13161e !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #c8f55a !important;
}

.stButton > button {
    background: #c8f55a !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    background: #13161e !important;
    color: #e8e3d8 !important;
    border: 1px solid #2a2d38 !important;
    border-radius: 10px !important;
    line-height: 1.7 !important;
}

.stProgress > div > div {
    background: #c8f55a !important;
}

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

hr {
    border-color: #2a2d38 !important;
}

.stDownloadButton > button {
    background: transparent !important;
    color: #c8f55a !important;
    border: 1.5px solid #c8f55a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CHUNK_DURATION_SECONDS = 10 * 60

CLEANUP_SYSTEM_PROMPT = """Είσαι ειδικός διορθωτής ελληνικών κειμένων.
Σου δίνεται αυτόματη μεταγραφή από ελληνικό ηχητικό αρχείο.

Οδηγίες:
1. Αφαίρεσε watermarks/artifacts χωρίς νόημα, όπως "AUTHORWAVE", "Υπότιτλοι" κ.λπ.
2. Διόρθωσε λάθη από παρακοή, βάσει πλαισίου.
3. Βελτίωσε στίξη, ορθογραφία και παραγράφους.
4. Μην αλλάξεις το νόημα.
5. Αν δεν είσαι σίγουρος για κάτι, άφησέ το όσο πιο κοντά γίνεται στο αρχικό.
6. Επέστρεψε μόνο το διορθωμένο κείμενο, χωρίς σχόλια, τίτλους ή εξηγήσεις."""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_secret(name: str, default: str = "") -> str:
    """Safely read Streamlit secrets without crashing locally."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def format_duration(seconds: float) -> str:
    seconds = int(seconds or 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def run_command(command: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ffmpeg_available() -> bool:
    try:
        result = run_command(["ffmpeg", "-version"], timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def ffprobe_duration(file_path: str) -> Optional[float]:
    try:
        result = run_command(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            timeout=15,
        )
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return float(value) if value else None
    except Exception:
        return None


def get_audio_info(raw_bytes: bytes, filename: str) -> Tuple[float, int]:
    """Return approximate duration and sample rate."""
    extension = os.path.splitext(filename)[1].lower()

    if extension in [".wav", ".flac", ".ogg"]:
        try:
            import soundfile as sf

            with sf.SoundFile(io.BytesIO(raw_bytes)) as audio_file:
                duration = len(audio_file) / float(audio_file.samplerate)
                return duration, int(audio_file.samplerate)
        except Exception:
            pass

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp_file:
            tmp_file.write(raw_bytes)
            tmp_path = tmp_file.name

        duration = ffprobe_duration(tmp_path)
        if duration:
            return duration, 48000
    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Fallback estimate when ffprobe is unavailable
    estimated_minutes = max(1.0, len(raw_bytes) / (1024 * 1024) * 2.5)
    return estimated_minutes * 60, 44100


def split_audio_into_chunks(
    raw_bytes: bytes,
    filename: str,
    chunk_seconds: int = CHUNK_DURATION_SECONDS,
) -> List[bytes]:
    """Split audio into MP3 chunks. If ffmpeg fails, return original bytes."""
    extension = os.path.splitext(filename)[1].lower() or ".mp3"

    input_path = None
    output_paths: List[str] = []

    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as input_file:
            input_file.write(raw_bytes)
            input_path = input_file.name

        total_duration = ffprobe_duration(input_path)
        if not total_duration or total_duration <= chunk_seconds:
            return [raw_bytes]

        chunk_count = int(total_duration // chunk_seconds)
        if total_duration % chunk_seconds:
            chunk_count += 1

        chunks: List[bytes] = []
        for index in range(chunk_count):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as output_file:
                output_path = output_file.name
                output_paths.append(output_path)

            command = [
                "ffmpeg",
                "-y",
                "-ss",
                str(index * chunk_seconds),
                "-t",
                str(chunk_seconds),
                "-i",
                input_path,
                "-vn",
                "-map",
                "a:0",
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "5",
                output_path,
            ]
            result = run_command(command, timeout=180)
            if result.returncode != 0:
                continue

            with open(output_path, "rb") as file:
                chunk_data = file.read()
                if chunk_data:
                    chunks.append(chunk_data)

        return chunks if chunks else [raw_bytes]

    except Exception:
        return [raw_bytes]

    finally:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        for path in output_paths:
            if os.path.exists(path):
                os.unlink(path)


def build_cleanup_prompt(custom_context: str = "") -> str:
    custom_context = (custom_context or "").strip()
    if not custom_context:
        return CLEANUP_SYSTEM_PROMPT
    return f"{CLEANUP_SYSTEM_PROMPT}\n\nΕιδικοί όροι / πλαίσιο που πρέπει να λάβεις υπόψη:\n{custom_context}"


def split_text_by_words(text: str, words_per_chunk: int = 1800) -> List[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


# ──────────────────────────────────────────────────────────────────────────────
# Transcription engines
# ──────────────────────────────────────────────────────────────────────────────
def transcribe_with_groq(raw_bytes: bytes, filename: str, api_key: str, progress) -> str:
    if Groq is None:
        raise RuntimeError("Το πακέτο groq δεν είναι εγκατεστημένο. Πρόσθεσε groq στο requirements.txt.")

    client = Groq(api_key=api_key)
    chunks = split_audio_into_chunks(raw_bytes, filename)
    parts: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        progress.progress(
            int((index - 1) / len(chunks) * 100),
            text=f"⚡ Groq Whisper — chunk {index}/{len(chunks)}…",
        )
        buffer = io.BytesIO(chunk)
        buffer.name = f"chunk_{index}.mp3"
        response = client.audio.transcriptions.create(
            file=buffer,
            model="whisper-large-v3",
            language="el",
            response_format="text",
        )
        parts.append(str(response).strip())
        time.sleep(0.2)

    progress.progress(100, text="Groq Whisper ολοκληρώθηκε ✅")
    return "\n\n".join(part for part in parts if part)


def transcribe_with_openai(raw_bytes: bytes, filename: str, api_key: str, progress) -> str:
    if openai is None:
        raise RuntimeError("Το πακέτο openai δεν είναι εγκατεστημένο. Πρόσθεσε openai στο requirements.txt.")

    client = openai.OpenAI(api_key=api_key)
    chunks = split_audio_into_chunks(raw_bytes, filename)
    parts: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        progress.progress(
            int((index - 1) / len(chunks) * 100),
            text=f"🟡 OpenAI Whisper — chunk {index}/{len(chunks)}…",
        )
        buffer = io.BytesIO(chunk)
        buffer.name = f"chunk_{index}.mp3"
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,
            language="el",
            response_format="text",
        )
        parts.append(str(response).strip())
        time.sleep(0.3)

    progress.progress(100, text="OpenAI Whisper ολοκληρώθηκε ✅")
    return "\n\n".join(part for part in parts if part)


# ──────────────────────────────────────────────────────────────────────────────
# Cleanup engines
# ──────────────────────────────────────────────────────────────────────────────
def cleanup_with_openai(text: str, api_key: str, context: str = "", progress=None) -> str:
    if openai is None:
        raise RuntimeError("Το πακέτο openai δεν είναι εγκατεστημένο. Πρόσθεσε openai στο requirements.txt.")

    client = openai.OpenAI(api_key=api_key)
    system_prompt = build_cleanup_prompt(context)
    chunks = split_text_by_words(text, words_per_chunk=1800)
    cleaned_parts: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if progress:
            progress.progress(
                int((index - 1) / len(chunks) * 100),
                text=f"✨ OpenAI cleanup — κομμάτι {index}/{len(chunks)}…",
            )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Διόρθωσε το παρακάτω κείμενο:\n\n{chunk}"},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        cleaned_parts.append(response.choices[0].message.content.strip())

    if progress:
        progress.progress(100, text="OpenAI cleanup ολοκληρώθηκε ✅")
    return "\n\n".join(cleaned_parts)


def cleanup_with_claude(text: str, api_key: str, context: str = "", progress=None) -> str:
    if anthropic is None:
        raise RuntimeError("Το πακέτο anthropic δεν είναι εγκατεστημένο. Πρόσθεσε anthropic στο requirements.txt.")

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = build_cleanup_prompt(context)
    chunks = split_text_by_words(text, words_per_chunk=1800)
    cleaned_parts: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if progress:
            progress.progress(
                int((index - 1) / len(chunks) * 100),
                text=f"🤖 Claude cleanup — κομμάτι {index}/{len(chunks)}…",
            )
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=4096,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Διόρθωσε το παρακάτω κείμενο:\n\n{chunk}"}
            ],
        )
        cleaned_parts.append(response.content[0].text.strip())

    if progress:
        progress.progress(100, text="Claude cleanup ολοκληρώθηκε ✅")
    return "\n\n".join(cleaned_parts)


def cleanup_with_groq(text: str, api_key: str, context: str = "", progress=None) -> str:
    if Groq is None:
        raise RuntimeError("Το πακέτο groq δεν είναι εγκατεστημένο. Πρόσθεσε groq στο requirements.txt.")

    client = Groq(api_key=api_key)
    system_prompt = build_cleanup_prompt(context)
    chunks = split_text_by_words(text, words_per_chunk=1200)
    cleaned_parts: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if progress:
            progress.progress(
                int((index - 1) / len(chunks) * 100),
                text=f"⚡ Groq cleanup — κομμάτι {index}/{len(chunks)}…",
            )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Διόρθωσε το παρακάτω κείμενο:\n\n{chunk}"},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        cleaned_parts.append(response.choices[0].message.content.strip())

        # Helpful for rate limits on free/low-tier Groq accounts
        if index < len(chunks):
            time.sleep(3)

    if progress:
        progress.progress(100, text="Groq cleanup ολοκληρώθηκε ✅")
    return "\n\n".join(cleaned_parts)


# ──────────────────────────────────────────────────────────────────────────────
# SRT export
# ──────────────────────────────────────────────────────────────────────────────
def seconds_to_srt_timestamp(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},000"


def make_srt(text: str, words_per_caption: int = 20, seconds_per_caption: int = 10) -> str:
    words = text.split()
    if not words:
        return ""

    captions: List[str] = []
    current_words: List[str] = []
    start_time = 0

    for index, word in enumerate(words, start=1):
        current_words.append(word)
        is_last_word = index == len(words)

        if len(current_words) >= words_per_caption or is_last_word:
            caption_number = len(captions) + 1
            end_time = start_time + seconds_per_caption
            captions.append(
                f"{caption_number}\n"
                f"{seconds_to_srt_timestamp(start_time)} --> {seconds_to_srt_timestamp(end_time)}\n"
                f"{' '.join(current_words)}\n"
            )
            current_words = []
            start_time = end_time

    return "\n".join(captions)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Keys")
    st.caption("Βάλε τα keys σου εδώ ή στα Streamlit Secrets.")

    groq_key = st.text_input(
        "Groq API Key ⚡",
        value=get_secret("GROQ_API_KEY"),
        type="password",
        placeholder="gsk_…",
    )
    openai_key = st.text_input(
        "OpenAI API Key",
        value=get_secret("OPENAI_API_KEY"),
        type="password",
        placeholder="sk-…",
    )
    claude_key = st.text_input(
        "Anthropic / Claude API Key 🤖",
        value=get_secret("ANTHROPIC_API_KEY"),
        type="password",
        placeholder="sk-ant-…",
    )

    st.markdown("---")
    st.markdown("## 🎙️ Μεταγραφή")
    transcription_engine = st.radio(
        "Engine μεταγραφής",
        ["Groq Whisper ⚡", "OpenAI Whisper"],
        index=0,
    )

    st.markdown("---")
    st.markdown("## 🧹 2ο Πέρασμα LLM")
    do_cleanup = st.checkbox("Αυτόματη διόρθωση κειμένου", value=True)

    cleanup_engine = "Claude Haiku 🤖"
    custom_context = ""
    if do_cleanup:
        cleanup_engine = st.radio(
            "Engine διόρθωσης",
            [
                "Claude Haiku 🤖",
                "Groq llama-3.3-70b ⚡",
                "OpenAI GPT-4o ✨",
            ],
            index=0,
        )
        custom_context = st.text_area(
            "Ειδικοί όροι / πλαίσιο (προαιρετικό)",
            placeholder="π.χ. Στοά, Τέκτων, Σεβάσμιος Διδάσκαλος, ασφαλιστικοί όροι...",
            height=90,
        )

    st.markdown("---")
    if ffmpeg_available():
        st.success("ffmpeg διαθέσιμο ✅")
    else:
        st.warning("Δεν βρέθηκε ffmpeg. Το app θα δουλέψει, αλλά ίσως χωρίς σωστό chunking.")


# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1>🎙️ Greek Audio <span class='accent'>Transcriber</span></h1>",
    unsafe_allow_html=True,
)
st.markdown("<h2>Upload audio → Greek transcript → optional LLM cleanup.</h2>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Drop your audio file here",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
)

if not uploaded_file:
    st.markdown(
        """
        <div style='text-align:center;padding:3rem 0;'>
            <div style='font-size:4rem'>🎙️</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:0.9rem;margin-top:1rem;color:#777;'>
                Waiting for audio file…<br>WAV · MP3 · M4A · OGG · FLAC
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

raw_bytes = uploaded_file.read()

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("File size", f"{len(raw_bytes) / 1024 / 1024:.1f} MB")

with st.spinner("Reading audio info…"):
    duration, sample_rate = get_audio_info(raw_bytes, uploaded_file.name)

metric_2.metric("Duration", format_duration(duration))
metric_3.metric("Sample rate", f"{sample_rate} Hz")
st.markdown("---")

if st.button("▶ Start Transcription", use_container_width=True):
    use_groq_transcription = transcription_engine.startswith("Groq")
    use_openai_transcription = transcription_engine.startswith("OpenAI")

    if use_groq_transcription and not groq_key:
        st.error("Προσθέστε Groq API key.")
        st.stop()

    if use_openai_transcription and not openai_key:
        st.error("Προσθέστε OpenAI API key.")
        st.stop()

    raw_transcript = ""
    method_used = ""

    try:
        if use_groq_transcription:
            st.info("⚡ Ξεκινάει μεταγραφή με Groq Whisper large-v3…")
            progress = st.progress(0, text="Προετοιμασία…")
            raw_transcript = transcribe_with_groq(raw_bytes, uploaded_file.name, groq_key, progress)
            method_used = "Groq Whisper large-v3"

        elif use_openai_transcription:
            st.info("🟡 Ξεκινάει μεταγραφή με OpenAI Whisper…")
            progress = st.progress(0, text="Προετοιμασία…")
            raw_transcript = transcribe_with_openai(raw_bytes, uploaded_file.name, openai_key, progress)
            method_used = "OpenAI Whisper"

    except Exception as error:
        st.error(f"Μεταγραφή απέτυχε: {error}")
        st.stop()

    if not raw_transcript.strip():
        st.error("Δεν επιστράφηκε μεταγραφή από το engine.")
        st.stop()

    cleaned_transcript: Optional[str] = None

    if do_cleanup:
        cleanup_key = None
        cleanup_label = ""

        if cleanup_engine.startswith("Claude"):
            cleanup_key = claude_key
            cleanup_label = "Claude Haiku"
        elif cleanup_engine.startswith("Groq"):
            cleanup_key = groq_key
            cleanup_label = "Groq llama-3.3-70b"
        elif cleanup_engine.startswith("OpenAI"):
            cleanup_key = openai_key
            cleanup_label = "OpenAI GPT-4o"

        if not cleanup_key:
            st.warning(f"Δεν υπάρχει API key για {cleanup_label}. Παρακάμπτεται το 2ο πέρασμα.")
        else:
            try:
                st.info(f"🧹 Ξεκινάει 2ο πέρασμα με {cleanup_label}…")
                cleanup_progress = st.progress(0, text="Επεξεργασία κειμένου…")

                if cleanup_engine.startswith("Claude"):
                    cleaned_transcript = cleanup_with_claude(
                        raw_transcript,
                        claude_key,
                        custom_context,
                        cleanup_progress,
                    )
                elif cleanup_engine.startswith("Groq"):
                    cleaned_transcript = cleanup_with_groq(
                        raw_transcript,
                        groq_key,
                        custom_context,
                        cleanup_progress,
                    )
                elif cleanup_engine.startswith("OpenAI"):
                    cleaned_transcript = cleanup_with_openai(
                        raw_transcript,
                        openai_key,
                        custom_context,
                        cleanup_progress,
                    )

            except Exception as error:
                st.warning(f"LLM cleanup απέτυχε: {error}")
                cleaned_transcript = None

    final_transcript = cleaned_transcript or raw_transcript
    final_label = method_used + (" + LLM διόρθωση" if cleaned_transcript else "")

    st.success(f"✅ Ολοκληρώθηκε: {final_label}")

    count_1, count_2 = st.columns(2)
    count_1.metric("Λέξεις", f"{len(final_transcript.split()):,}")
    count_2.metric("Χαρακτήρες", f"{len(final_transcript):,}")

    st.markdown("---")

    if cleaned_transcript:
        tab_clean, tab_raw = st.tabs(["✅ Διορθωμένο", "📄 Αρχικό raw"])
        with tab_clean:
            st.text_area(
                "Διορθωμένο",
                cleaned_transcript,
                height=450,
                label_visibility="collapsed",
            )
        with tab_raw:
            st.text_area(
                "Αρχικό raw",
                raw_transcript,
                height=450,
                label_visibility="collapsed",
            )
    else:
        st.text_area(
            "Transcript",
            raw_transcript,
            height=450,
            label_visibility="collapsed",
        )

    base_name = os.path.splitext(uploaded_file.name)[0]
    st.markdown("### 💾 Download")
    download_1, download_2, download_3 = st.columns(3)

    with download_1:
        st.download_button(
            "⬇ Τελικό .TXT",
            data=final_transcript.encode("utf-8"),
            file_name=f"{base_name}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with download_2:
        st.download_button(
            "⬇ Αρχικό .TXT",
            data=raw_transcript.encode("utf-8"),
            file_name=f"{base_name}_raw.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with download_3:
        st.download_button(
            "⬇ Υπότιτλοι .SRT",
            data=make_srt(final_transcript).encode("utf-8"),
            file_name=f"{base_name}.srt",
            mime="text/plain",
            use_container_width=True,
        )
