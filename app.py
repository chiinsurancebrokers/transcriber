"""
ASKLEPIOS — AI Nurse
Bilingual AI health assistant for the Greek market.
Standalone Streamlit app · Real data only · No placeholders.
"""

import streamlit as st
import os
import json
import io
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import io as _io, base64 as _b64
import hmac, hashlib, time, unicodedata

# "Stay signed in" via a browser cookie (persists login across reloads / new tabs,
# e.g. when returning from the external face scan). Degrades gracefully if missing.
try:
    import extra_streamlit_components as stx
    _STX_OK = True
except Exception:
    _STX_OK = False

# HEIC support for iPhone photos
try:
    import pillow_heif as _heif
    from PIL import Image as _Image
    _heif.register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

# ── SAFE SECRETS / ENV ACCESS ─────────────────────────────────────────────────
def _secret(name, default=""):
    """Read a config value from st.secrets, falling back to os.environ, then default.
    Safe on platforms (e.g. Railway) where no secrets.toml exists — accessing
    st.secrets there raises StreamlitSecretNotFoundError even with a default."""
    try:
        v = st.secrets.get(name, None)
        if v not in (None, ""):
            return v
    except Exception:
        pass
    v = os.environ.get(name, "")
    return v if v != "" else default

# ── PHOTO SCANNER FUNCTIONS ───────────────────────────────────────────────────
HUMAN_SCAN_PROMPTS = {
    "eye":    "Examine the eye carefully. Describe: sclera colour (white/red/yellow), pupil symmetry, conjunctiva, any discharge (colour, quantity), eyelid swelling, corneal clarity, third eyelid. Flag any urgent findings.",
    "skin":   "Examine the skin lesion or rash. Describe: colour, size (estimate), borders (regular/irregular), texture (flat/raised/scaly), distribution pattern, any ulceration, satellite lesions. Note ABCDE criteria if applicable (Asymmetry, Border, Colour, Diameter, Evolution).",
    "wound":  "Examine the wound. Describe: type (laceration/abrasion/puncture/burn), dimensions (estimate), depth, wound edges, signs of infection (redness/swelling/warmth/pus/odour), presence of foreign bodies, tissue viability.",
    "throat": "Examine the mouth and throat. Describe: tonsil size and appearance, pharyngeal wall, any exudate or white patches, uvula position, tongue coating, gum condition, mucosal lesions, petechiae on palate.",
    "nails":  "Examine the nails. Describe: colour (pale/yellow/blue/brown/white), shape (clubbing/koilonychia/normal), surface (ridges/pitting/onycholysis), subungual changes, surrounding skin.",
    "body":   "Describe the visible body area. Note: skin colour, visible swelling, asymmetry, rashes, bruising, oedema, muscle wasting, posture, any visible masses or lesions.",
}

def convert_heic_human(img_bytes):
    if not HEIC_OK: raise RuntimeError("pillow-heif not installed")
    img = _Image.open(_io.BytesIO(img_bytes))
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"

def florence2_human(image_b64, scan_type, api_key):
    workspace = _secret("ROBOFLOW_WORKSPACE","chriss-workspace-zk0ng")
    workflow  = _secret("ROBOFLOW_WORKFLOW","florence2-base-demo")
    url = f"https://serverless.roboflow.com/{workspace}/workflows/{workflow}"
    task_prompt = HUMAN_SCAN_PROMPTS.get(scan_type, HUMAN_SCAN_PROMPTS["skin"])
    body = json.dumps({
        "api_key": api_key,
        "inputs": {"image":{"type":"base64","value":image_b64},"task_prompt":task_prompt}
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
        outputs = result.get("outputs",[])
        if outputs:
            for key in ["output","caption","text","result","description"]:
                if key in outputs[0] and outputs[0][key]:
                    return {"ok":True,"description":str(outputs[0][key])}
        return {"ok":True,"description":str(result)}
    except Exception as e:
        return {"ok":False,"error":str(e)}

def claude_vision_human(image_b64, image_type, prompt, system=""):
    key = get_claude_key()
    if not key: return "⚠️ API key not set."
    body = json.dumps({
        "model":"claude-sonnet-4-6","max_tokens":3000,"system":system,
        "messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":image_type,"data":image_b64}},
            {"type":"text","text":prompt}
        ]}]
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",data=body,
        headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=60) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception as e: return f"⚠️ {e}"

def transcribe_audio(audio_bytes, lang="el", mime="audio/webm", filename="recording.webm"):
    """Transcribe a short voice recording → text. Groq Whisper large-v3 primary
    (fast, free tier, Greek-capable), OpenAI Whisper-1 fallback.
    
    Input expected from st.audio_input → WebM/Opus, small (~1MB per minute).
    Both APIs use multipart/form-data. We build it manually with urllib to
    avoid adding requests as a dep.
    
    Privacy: audio goes to the chosen STT API for processing, NEVER stored
    on our side. Only the resulting transcript text enters session state."""
    import uuid as _uuid
    boundary = f"----asklepios{_uuid.uuid4().hex}"
    
    def _multipart(parts):
        """parts: list of (name, value, filename_or_None, content_type_or_None)"""
        body = bytearray()
        for name, value, fn, ct in parts:
            body += f"--{boundary}\r\n".encode()
            if fn:
                body += f'Content-Disposition: form-data; name="{name}"; filename="{fn}"\r\n'.encode()
                body += f"Content-Type: {ct or 'application/octet-stream'}\r\n\r\n".encode()
                body += value
                body += b"\r\n"
            else:
                body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
                body += str(value).encode()
                body += b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        return bytes(body)
    
    # Try Groq Whisper large-v3 first
    groq_key = get_groq_key()
    if groq_key:
        try:
            body = _multipart([
                ("file",  audio_bytes, filename, mime),
                ("model", "whisper-large-v3", None, None),
                ("language", lang, None, None),
                ("response_format", "text", None, None),
            ])
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                data=body,
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode("utf-8", errors="replace").strip()
            if txt:
                return txt.strip('"'), "groq"
        except Exception:
            pass
    
    # Fallback: OpenAI Whisper-1
    openai_key = get_openai_key()
    if openai_key:
        try:
            body = _multipart([
                ("file",  audio_bytes, filename, mime),
                ("model", "whisper-1", None, None),
                ("language", lang, None, None),
                ("response_format", "text", None, None),
            ])
            req = urllib.request.Request(
                "https://api.openai.com/v1/audio/transcriptions",
                data=body,
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode("utf-8", errors="replace").strip()
            if txt:
                return txt.strip('"'), "openai"
        except Exception as e:
            return f"⚠️ Σφάλμα μεταγραφής: {e}", None
    
    return "⚠️ Καμία STT API key δεν είναι ρυθμισμένη (Groq ή OpenAI).", None


def claude_analyze_lab(file_bytes, mime_type, profile, conversation, lang, file_name=""):
    """Analyze lab results (PDF or image) via Claude with native document support.
    
    Lab results in Greece are typically PDFs from labs (e.g. Biocheck, Affidea) or
    phone photos of paper printouts. Claude's PDF support handles both text-based
    and image-based PDFs internally (built-in OCR). Result is interpreted WITHIN
    the conversation context — not as a standalone report — so findings tie back
    to the user's reported symptoms.
    
    Privacy: file is sent to Claude API for processing, NEVER stored on our side.
    """
    key = get_claude_key()
    if not key:
        return "⚠️ Claude API key not set."
    
    file_b64 = _b64.b64encode(file_bytes).decode()
    
    # Clinical context from the ongoing assessment
    convo_txt = "\n".join(
        f"{'Ασθενής' if m['role']=='user' else 'Asklepios'}: {m['content'][:400]}"
        for m in (conversation or [])[-6:]
    ) if conversation else ("Δεν έχει καταγραφεί συνομιλία ακόμη." if lang=="el" else "No conversation yet.")
    
    age = profile.get("age", "?")
    sex = profile.get("sex", "")
    history = profile.get("history", "") or "—"
    meds = profile.get("meds_raw", "") or "—"
    # Special-population flags affect reference ranges + drug warnings
    flags = []
    if profile.get("pregnancy"):
        flags.append("ΕΓΚΥΟΣ" if lang=="el" else "PREGNANT")
    if profile.get("for_whom") == "other":
        flags.append("Caregiver-mode" if lang=="el" else "Caregiver-mode")
    try:
        _aint = int(age)
        if _aint < 18:
            flags.append(f"ΠΑΙΔΙΑΤΡΙΚΟΣ {_aint}" if lang=="el" else f"PEDIATRIC {_aint}")
    except (TypeError, ValueError):
        pass
    flags_line = (" | ".join(flags)) if flags else ("—" if lang=="el" else "—")
    
    if lang == "el":
        system = ("Είσαι έμπειρος ιατρός νοσηλευτής που ερμηνεύει εργαστηριακές εξετάσεις "
                  "στα Ελληνικά. Είσαι ακριβής, σαφής, και κάνεις το κλινικό συμπέρασμα ΜΕΣΑ "
                  "στο πλαίσιο των συμπτωμάτων και του ιστορικού. ΔΕΝ κάνεις τελική διάγνωση — "
                  "επισημαίνεις ευρήματα και τι μπορεί να σημαίνουν.")
        prompt = f"""ΚΛΙΝΙΚΟ ΠΛΑΙΣΙΟ:
Ασθενής: {age} ετών, {sex}
Ιστορικό: {history}
Φάρμακα: {meds}

Συνομιλία μέχρι τώρα:
{convo_txt}

---

ΕΡΓΑΣΤΗΡΙΑΚΕΣ ΕΞΕΤΑΣΕΙΣ (επισυνάπτεται PDF/εικόνα):
Ανάλυσε τα αποτελέσματα σε αυτές τις ενότητες:

**1. ΕΥΡΗΜΑΤΑ ΕΚΤΟΣ ΟΡΙΩΝ**
Πίνακας ή λίστα με τους δείκτες που είναι ψηλά ή χαμηλά, με την τιμή, τα όρια αναφοράς, την κατεύθυνση (↑/↓). Αν όλα είναι εντός ορίων, πες το ξεκάθαρα.

**2. ΕΡΜΗΝΕΙΑ**
Τι μπορεί να σημαίνει αυτή η εικόνα κλινικά. Σύντομα, σε απλή γλώσσα.

**3. ΣΧΕΣΗ ΜΕ ΣΥΜΠΤΩΜΑΤΑ**
Συμβατά με όσα περιγράφει ο ασθενής στη συνομιλία; Υποστηρίζουν την τρέχουσα εκτίμηση ή την αλλάζουν;

**4. ΕΠΟΜΕΝΑ ΒΗΜΑΤΑ**
Τι θα ρωτούσε ο γιατρός. Επιπλέον εξετάσεις που ίσως χρειάζονται. Πότε είναι επείγον.

ΣΗΜΑΝΤΙΚΟ: ΜΗΝ κάνεις τελική διάγνωση. Πάντα συστήνεις επίσκεψη σε ιατρό για ερμηνεία.
Αναφέρε ΜΟΝΟ τα ευρήματα που πραγματικά βλέπεις στο έγγραφο — μην εφεύρεις δείκτες."""
    else:
        system = ("You are an expert clinical reviewer interpreting lab results. Be precise, "
                  "clear, and tie findings to the patient's reported symptoms. Do NOT make a "
                  "final diagnosis — surface findings and what they may indicate.")
        prompt = f"""CLINICAL CONTEXT:
Patient: {age} yo {sex}
History: {history}
Medications: {meds}

Conversation so far:
{convo_txt}

---

LAB RESULTS (PDF/image attached):
Analyse in these sections:

**1. OUT-OF-RANGE FINDINGS**
Table or list of indicators that are high or low, with value, reference range, direction (↑/↓). If all within range, say so clearly.

**2. INTERPRETATION**
What this clinical picture may indicate. Brief, plain language.

**3. RELATION TO SYMPTOMS**
Consistent with what the patient describes? Supports or changes the current assessment?

**4. NEXT STEPS**
What the doctor would ask. Additional tests possibly needed. When this is urgent.

IMPORTANT: Do NOT make a final diagnosis. Always recommend seeing a doctor for interpretation.
Only findings you actually see in the document — don't invent indicators."""
    
    # Build content block based on file type
    if mime_type == "application/pdf":
        content_block = {
            "type": "document",
            "source": {"type":"base64", "media_type":"application/pdf", "data":file_b64}
        }
    else:
        content_block = {
            "type": "image",
            "source": {"type":"base64", "media_type":mime_type, "data":file_b64}
        }
    
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 3000,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [content_block, {"type":"text","text":prompt}]
        }]
    }).encode()
    
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception as e:
        return f"⚠️ Σφάλμα ανάλυσης: {e}"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asklepios · AI Nurse",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F0F4FF 0%, #F8F0FF 100%);
}
[data-testid="stSidebar"] { display: none; }

.kira-hero {
    background: linear-gradient(135deg, #2D3FE7 0%, #7B2FE0 100%);
    border-radius: 20px;
    padding: 48px 40px;
    color: white;
    text-align: center;
    margin-bottom: 32px;
}
.kira-hero h1 { font-size: 52px; font-weight: 700; margin: 0; letter-spacing: -1px; }
.kira-hero p  { font-size: 18px; opacity: 0.85; margin: 12px 0 0; }
.kira-tagline { font-size: 13px; opacity: 0.65; margin-top: 8px; letter-spacing: 2px; text-transform: uppercase; }

.card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(45,63,231,0.07);
    border: 1px solid rgba(45,63,231,0.08);
}
.card h3 { font-size: 16px; font-weight: 600; margin: 0 0 16px; color: #1A1A2E; }

.vital-badge {
    background: #F4F6FF;
    border: 1px solid #E0E5FF;
    border-radius: 12px;
    padding: 14px 18px;
    min-width: 120px;
    text-align: center;
    flex: 1;
}
.vital-badge.green { background: #EDFBF0; border-color: #A3E6B5; }
.vital-badge.yellow { background: #FFFBEB; border-color: #FCD34D; }
.vital-badge.red { background: #FEF2F2; border-color: #FCA5A5; }
.vital-badge .vb-value { font-size: 22px; font-weight: 700; color: #1A1A2E; }
.vital-badge .vb-label { font-size: 11px; color: #6B7280; margin-top: 2px; }
.vital-badge .vb-unit  { font-size: 10px; color: #9CA3AF; }

.pill { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.pill-green  { background: #DCFCE7; color: #15803D; }
.pill-yellow { background: #FEF9C3; color: #A16207; }
.pill-red    { background: #FEE2E2; color: #B91C1C; }
.pill-blue   { background: #DBEAFE; color: #1D4ED8; }

.disclaimer {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #92400E;
    margin: 12px 0;
}
.disclaimer-red {
    background: #FEF2F2;
    border: 1px solid #FCA5A5;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #991B1B;
    margin: 12px 0;
}

.emergency {
    background: linear-gradient(90deg, #DC2626, #B91C1C);
    color: white; border-radius: 10px; padding: 16px 20px;
    font-weight: 600; font-size: 14px; margin: 12px 0;
}

.kira-stepper {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin: 0 0 28px; padding: 16px 0 0;
}
.kira-step {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    flex: 1; max-width: 120px;
}
.kira-step-circle {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; border: 2px solid #E0E5FF;
    background: white; color: #CBD5E1; position: relative; z-index: 1;
}
.kira-step.done   .kira-step-circle { background: #7B2FE0; border-color: #7B2FE0; color: white; }
.kira-step.active .kira-step-circle { background: #2D3FE7; border-color: #2D3FE7; color: white; box-shadow: 0 0 0 4px rgba(45,63,231,.15); }
.kira-step-label { font-size: 10px; color: #94A3B8; text-align: center; letter-spacing: .02em; }
.kira-step.done   .kira-step-label  { color: #7B2FE0; }
.kira-step.active .kira-step-label  { color: #2D3FE7; font-weight: 600; }
.kira-step-line {
    flex: 1; height: 2px; background: #E0E5FF; margin-bottom: 18px;
}
.kira-step-line.done { background: #7B2FE0; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px; }
.chip {
    padding: 6px 14px; border-radius: 20px; font-size: 13px; cursor: pointer;
    border: 1.5px solid #C4B5FD; color: #5B21B6; background: #F5F3FF;
    transition: all .15s; user-select: none;
}
.chip.selected { background: #7B2FE0; border-color: #7B2FE0; color: white; }

.wellness-wrap {
    display: flex; align-items: center; gap: 20px;
    background: linear-gradient(135deg,#2D3FE7,#7B2FE0);
    border-radius: 16px; padding: 20px 24px; margin-bottom: 20px; color: white;
}
.wellness-score { font-size: 48px; font-weight: 800; letter-spacing: -2px; }
.wellness-label { font-size: 12px; opacity: .7; text-transform: uppercase; letter-spacing: 1.5px; }
.wellness-desc  { font-size: 15px; opacity: .9; margin-top: 4px; }

.red-flags-urgent {
    background: linear-gradient(90deg,#DC2626,#B91C1C);
    color: white; border-radius: 12px; padding: 16px 20px; margin: 12px 0;
    animation: pulse-bg 2s ease-in-out infinite;
}
@keyframes pulse-bg { 0%,100%{opacity:1} 50%{opacity:.85} }

/* Mobile */
@media (max-width: 768px) {
    .kira-hero h1 { font-size: 32px !important; }
    .kira-hero { padding: 28px 20px !important; }
    .stChatMessage { font-size: 14px !important; }
    [data-testid="stChatMessageContent"] { max-width: 100% !important; overflow-wrap: break-word !important; }
    .main .block-container { padding-bottom: 120px !important; }
    .stButton button { white-space: normal !important; min-height: 44px !important; }
}
[data-testid="stMarkdownContainer"] { overflow-wrap: break-word !important; word-break: break-word !important; }
/* Markdown tables — clean column alignment. Auto layout (no fixed widths) so each
 * table sizes naturally: differential-diagnosis (3 cols) and treatment plans (2 cols)
 * both render correctly. Previously had table-layout:fixed + nth-child(2):64px which
 * was meant for the diagnosis %-column but accidentally squashed every 2-col table
 * into one-letter-per-line on mobile. */
[data-testid="stMarkdownContainer"] table {
    width: 100%; border-collapse: collapse;
    font-size: 12.5px; margin: 12px 0;
}
[data-testid="stMarkdownContainer"] thead th { background: #F4F6FF; font-weight: 600; }
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] td {
    border: 1px solid #E0E5FF; padding: 7px 9px;
    text-align: left; vertical-align: top;
    word-break: normal !important; overflow-wrap: break-word !important; hyphens: none;
}
</style>
""", unsafe_allow_html=True)

# ── KEYS ──────────────────────────────────────────────────────────────────────
def _key(name, fallback=""):
    for k in [name, name.lower(), name.upper()]:
        v = _secret(k, "")
        if v:
            return v
    return fallback

def get_claude_key():  return _key("Claude_API_Key")
def get_openai_key():  return _key("OPENAI_API_KEY")
def get_groq_key():    return _key("GROQ_API_KEY")
def get_ncbi_key():    return _key("NCBI_API_KEY")

# ── AUTH (Supabase email-OTP — gates the premium report) ──────────────────────
# Graceful degradation: if SUPABASE_URL / SUPABASE_ANON_KEY are not set (or the
# supabase package is missing), auth stays OFF and the whole app is open — so the
# demo keeps working. Set the secrets to switch the gate on automatically.
def _supabase_client():
    url = _secret("SUPABASE_URL", "")
    key = _secret("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None

def auth_enabled():
    return _supabase_client() is not None

def is_logged_in():
    return bool(st.session_state.get("auth_user"))

# ── PERSISTENT LOGIN (HMAC-signed cookie — cannot be forged) ──────────────────
CM = None  # CookieManager instance, created once per run in the router
COOKIE_NAME = "ak_session"

def _cookie_secret():
    return (_secret("AUTH_COOKIE_SECRET","") or _secret("SUPABASE_ANON_KEY","")
            or "asklepios-dev-cookie-secret")

def _make_token(email, days=14):
    exp = int(time.time()) + days*86400
    body = f"{email}|{exp}"
    sig = hmac.new(_cookie_secret().encode(), body.encode(), hashlib.sha256).hexdigest()[:32]
    return _b64.urlsafe_b64encode(f"{body}|{sig}".encode()).decode()

def _read_token(tok):
    try:
        raw = _b64.urlsafe_b64decode(str(tok).encode()).decode()
        email, exp, sig = raw.rsplit("|", 2)
        if int(exp) < time.time():
            return None
        good = hmac.new(_cookie_secret().encode(), f"{email}|{exp}".encode(), hashlib.sha256).hexdigest()[:32]
        if hmac.compare_digest(sig, good):
            return email
    except Exception:
        return None
    return None

def _save_login_cookie(email):
    cm = globals().get("CM")
    if not cm:
        return
    try:
        cm.set(COOKIE_NAME, _make_token(email), key="ak_set_auth",
               expires_at=datetime.now()+timedelta(days=14))
    except Exception:
        pass

def _clear_login_cookie():
    cm = globals().get("CM")
    if not cm:
        return
    try:
        cm.delete(COOKIE_NAME, key="ak_del_auth")
    except Exception:
        pass

# ── IN-PROGRESS PROFILE DRAFT (server-side, encrypted) ────────────────────────
# Returning from the external face scan opens a NEW browser tab → a fresh Streamlit
# session, so the profile held in session_state is gone and intake gets re-asked.
# We persist the profile server-side in Supabase, keyed by the user's email, and
# ENCRYPT it (Fernet symmetric, key derived from the app secret) so the stored row
# is ciphertext — readable only by the app, not by anyone who can see the DB.
try:
    from cryptography.fernet import Fernet
    _ENC_OK = True
except Exception:
    _ENC_OK = False

def _fernet():
    # 32-byte urlsafe key derived from the app secret (AUTH_COOKIE_SECRET / anon key)
    key = _b64.urlsafe_b64encode(hashlib.sha256(_cookie_secret().encode()).digest())
    return Fernet(key)

def save_draft(email, payload):
    sb = _supabase_client()
    if not sb or not email or not _ENC_OK:
        return
    try:
        blob = _fernet().encrypt(
            json.dumps(payload, ensure_ascii=False).encode()
        ).decode()
        sb.table("drafts").upsert({"user_email": email, "data": blob}, on_conflict="user_email").execute()
    except Exception:
        pass

def load_draft(email):
    sb = _supabase_client()
    if not sb or not email or not _ENC_OK:
        return None
    try:
        res = sb.table("drafts").select("data").eq("user_email", email).limit(1).execute()
        rows = res.data or []
        if rows and rows[0].get("data"):
            dec = _fernet().decrypt(rows[0]["data"].encode()).decode()
            return json.loads(dec)
    except Exception:
        return None
    return None

def delete_draft(email):
    sb = _supabase_client()
    if not sb or not email:
        return
    try:
        sb.table("drafts").delete().eq("user_email", email).execute()
    except Exception:
        pass

def _save_session_for_external_nav():
    """Save the current assessment to Supabase right before the user clicks an
    EXTERNAL navigation (face scan in a new tab). Called from the render that
    shows the scan link, so by click time the draft is in the DB and the new tab
    can restore it. Single-use, deleted immediately after restore."""
    if not (auth_enabled() and is_logged_in() and st.session_state.profile.get("name")):
        return
    payload = {
        "profile":         st.session_state.profile,
        "lang":            st.session_state.lang,
        "triage_chat":     st.session_state.triage_chat,
        "medications":     st.session_state.medications,
        "vitals_analysis": st.session_state.vitals_analysis,
    }
    save_draft(st.session_state.get("auth_user", ""), payload)


def send_otp(email):
    sb = _supabase_client()
    if not sb: return False, "Auth not configured."
    try:
        sb.auth.sign_in_with_otp({"email": email})
        return True, ""
    except Exception as e:
        return False, str(e)

def verify_otp(email, token):
    sb = _supabase_client()
    if not sb: return False, "Auth not configured."
    token = str(token).strip()
    last_err = "invalid"
    # New users (or with "Confirm email" on) get a 'signup' token; returning users get 'email'.
    for otp_type in ("email", "signup"):
        try:
            res = sb.auth.verify_otp({"email": email, "token": token, "type": otp_type})
            if getattr(res, "user", None):
                st.session_state["auth_user"] = email
                return True, ""
        except Exception as e:
            last_err = str(e)
    return False, last_err

def logout():
    sb = _supabase_client()
    if sb:
        try: sb.auth.sign_out()
        except Exception: pass
    delete_draft(st.session_state.get("auth_user", ""))
    _clear_login_cookie()
    # FULL RESET on exit — wipe assessment state and runtime flags. Keep language
    # preference (it's a UI choice, not assessment data).
    _lang_keep = st.session_state.get("lang", "el")
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state["lang"] = _lang_keep
    try:
        if "pe" in st.query_params: del st.query_params["pe"]
    except Exception:
        pass

def render_login_gate():
    """Inline email->OTP login. Returns True once the user is logged in.

    UX-hardened (tester report, Cyprus): when Supabase rate-limits or returns a
    transient error, the email STILL gets delivered — but the previous version
    showed an error and never revealed the code-entry field. Result: the user has
    a code in their inbox and nowhere to type it. This version always advances to
    the code-entry stage after the send button is clicked, regardless of the API
    return code. If no code arrives the user can press 'Resend' or 'Different email'."""
    lang = st.session_state.lang
    if is_logged_in():
        return True

    # Friendly header
    st.markdown(f'''<div style="background:rgba(45,63,231,0.06);border:1px solid rgba(45,63,231,0.15);border-radius:14px;padding:20px 22px;text-align:center;margin:10px 0">
        <div style="font-size:34px;margin-bottom:6px">🔒</div>
        <div style="font-size:16px;font-weight:700;color:#1A1A2E">{"Σύνδεση" if lang=="el" else "Sign in"}</div>
        <div style="font-size:13px;color:#6B7280;margin-top:4px">{"Email + κωδικός μίας χρήσης. Χωρίς password." if lang=="el" else "Email + one-time code. No password."}</div>
    </div>''', unsafe_allow_html=True)

    # Recover pending email across mobile reloads / fresh tabs
    sent_to = st.session_state.get("otp_sent_to")
    if not sent_to:
        pe = st.query_params.get("pe")
        if pe:
            st.session_state["otp_sent_to"] = pe
            sent_to = pe

    if not sent_to:
        # ── STAGE 1: enter email ────────────────────────────────────────────
        email = st.text_input("Email", key="otp_email", placeholder="you@example.com")
        if st.button(("📩 " + ("Στείλε μου τον κωδικό" if lang=="el" else "Send me the code")),
                     type="primary", use_container_width=True, key="otp_send"):
            if email and "@" in email:
                # Best-effort send: ADVANCE to stage 2 regardless of API result.
                # Email is usually delivered even when Supabase rate-limits the
                # response — the user just needs the code field to appear.
                ok, err = send_otp(email)
                st.session_state["otp_sent_to"] = email
                st.query_params["pe"] = email
                if not ok:
                    st.session_state["_otp_send_warning"] = (err or "")[:140]
                st.rerun()
            else:
                st.warning("Έγκυρο email, παρακαλώ." if lang=="el" else "Please enter a valid email.")
    else:
        # ── STAGE 2: enter code ─────────────────────────────────────────────
        warn = st.session_state.pop("_otp_send_warning", None)
        if warn:
            # Soft warning — DON'T block the code field. Email may still have arrived.
            st.warning(("⚠️ Πιθανό πρόβλημα στην αποστολή — αλλά ο κωδικός μπορεί να έχει φτάσει στο email σου. "
                        "Έλεγξε το inbox και το spam folder, και βάλε τον κωδικό παρακάτω. "
                        "Αν δεν λάβεις τίποτα σε 1 λεπτό, πάτα «Νέος κωδικός»."
                        if lang=="el" else
                        "⚠️ The send response had an issue — but the code may still have reached your email. "
                        "Check your inbox and spam folder, then enter the code below. "
                        "If nothing arrives within 1 minute, press 'New code'."))
        else:
            st.success(f"📧 " + (f"Σου στείλαμε κωδικό στο **{sent_to}**" if lang=="el"
                                  else f"We sent a code to **{sent_to}**"))
        st.caption(("Έλεγξε το inbox και το spam folder. Ο κωδικός φτάνει σε λίγα δευτερόλεπτα."
                    if lang=="el" else
                    "Check your inbox and spam folder. The code arrives within a few seconds."))

        code = st.text_input(
            ("Κωδικός από το email" if lang=="el" else "Code from your email"),
            key="otp_code",
            placeholder="12345678",
            max_chars=8,
        )
        if st.button(("✓ " + ("Επιβεβαίωση & Σύνδεση" if lang=="el" else "Verify & Sign in")),
                     type="primary", use_container_width=True, key="otp_verify"):
            _code_clean = str(code or "").strip().replace(" ", "")
            if not _code_clean.isdigit() or len(_code_clean) < 6:
                st.warning(("Βάλε τον κωδικό από το email (6-8 ψηφία)." if lang=="el"
                            else "Enter the code from your email (6-8 digits)."))
            else:
                ok, err = verify_otp(sent_to, _code_clean)
                if ok:
                    st.session_state.pop("otp_sent_to", None)
                    if "pe" in st.query_params: del st.query_params["pe"]
                    st.rerun()
                else:
                    st.error(("Λάθος ή ληγμένος κωδικός — δοκίμασε ξανά ή πάτα «Νέος κωδικός»."
                              if lang=="el" else
                              "Wrong or expired code — try again or press 'New code'."))

        c1, c2 = st.columns(2)
        with c1:
            if st.button(("📩 " + ("Νέος κωδικός" if lang=="el" else "New code")),
                         use_container_width=True, key="otp_resend"):
                ok2, err2 = send_otp(sent_to)
                # Always show user-friendly message — code may have arrived regardless
                if ok2:
                    st.success(("Νέος κωδικός στάλθηκε." if lang=="el" else "New code sent."))
                else:
                    st.info(("Αν δεν λάβεις νέο κωδικό σε 60'', χρησιμοποίησε τον προηγούμενο που έλαβες."
                             if lang=="el" else
                             "If no new code arrives in 60s, use the previous one you received."))
        with c2:
            if st.button(("Άλλο email" if lang=="el" else "Different email"),
                         use_container_width=True, key="otp_reset"):
                st.session_state.pop("otp_sent_to", None)
                if "pe" in st.query_params: del st.query_params["pe"]
                st.rerun()

    return is_logged_in()

def render_ad_banner(lang):
    """Editorial-style value-prop banner shown on the login screen. Uses pure
    st.markdown (no iframe → no deprecation, no JS). Editorial typography
    inspired by Cira but with HONEST claims only:
      • Heart rate yes (rPPG is reliable for HR)
      • NO blood pressure or "30+ vitals" promise — rPPG can't reliably do those
      • GDPR not HIPAA — we are EU-based
    Cards visualize the actual product: chat bubble (symptoms),
    vitals readout (HR + BP/SpO₂ entered by user), report checklist."""
    if lang == "en":
        d = {
            "pill_l":"ASKLEPIOS · AI NURSE", "pill_r":"🔒 GDPR · Encrypted",
            "h_l":"Symptoms.", "h_m":"Assessment.", "h_r":"In Greek.",
            "sub":"Describe what you're feeling. Get a clinical assessment with PubMed references. In a few minutes.",
            "s1_lbl":"YOU", "s1_text":"\"Headache and nausea for 3 days…\"",
            "s2_lbl":"VITALS",
            "s2_v1":"HR", "s2_v1v":"78 bpm",
            "s2_v2":"BP", "s2_v2v":"120/80",
            "s3_lbl":"REPORT",
            "s3_l1":"Clinical assessment",
            "s3_l2":"PubMed references",
            "s3_l3":"Drug interactions",
            "t1":"🇬🇷 Greek", "t2":"🔒 GDPR",
            "t3":"📚 PubMed", "t4":"🤖 Claude + GPT-4o", "t5":"⚡ Free",
        }
    else:
        d = {
            "pill_l":"ASKLEPIOS · AI ΝΟΣΗΛΕΥΤΗΣ", "pill_r":"🔒 GDPR · Κρυπτογράφηση",
            "h_l":"Συμπτώματα.", "h_m":"Εκτίμηση.", "h_r":"Στα Ελληνικά.",
            "sub":"Περίγραψε τι νιώθεις. Λάβε κλινική εκτίμηση με τεκμηρίωση από PubMed. Σε λίγα λεπτά.",
            "s1_lbl":"ΕΣΥ", "s1_text":"«Πονοκέφαλος και ναυτία 3 μέρες…»",
            "s2_lbl":"ΖΩΤΙΚΑ",
            "s2_v1":"HR", "s2_v1v":"78 bpm",
            "s2_v2":"BP", "s2_v2v":"120/80",
            "s3_lbl":"ΑΝΑΦΟΡΑ",
            "s3_l1":"Κλινική εκτίμηση",
            "s3_l2":"Αναφορές PubMed",
            "s3_l3":"Αλληλεπιδράσεις",
            "t1":"🇬🇷 Ελληνικά", "t2":"🔒 GDPR",
            "t3":"📚 PubMed", "t4":"🤖 Claude + GPT-4o", "t5":"⚡ Δωρεάν",
        }
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700;1,800;1,900&family=Inter:wght@400;500;600;700&display=swap');
.ad-hero {
  background: linear-gradient(180deg, #FAF7F2 0%, #F2EDE3 100%);
  border-radius: 28px; padding: 60px 40px 36px;
  margin: 12px 0 28px; text-align: center;
  font-family: 'Inter', system-ui, sans-serif;
  border: 1px solid rgba(122, 47, 224, 0.08);
}
.ad-pill {
  display: inline-flex; align-items: center; gap: 12px;
  background: white; border: 1px solid #E5E7EB;
  border-radius: 999px; padding: 8px 18px;
  font-size: 11.5px; font-weight: 700; letter-spacing: 0.1em;
  color: #7B2FE0; margin-bottom: 24px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.ad-pill .sep { color: #D1D5DB; font-weight: 400; }
.ad-pill .gdpr { color: #10B981; letter-spacing: 0.04em; }
.ad-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 60px; font-weight: 700; line-height: 1.02;
  letter-spacing: -2px; color: #1A1A2E; margin: 0 0 4px;
}
.ad-title .word { display: inline-block; }
.ad-title .accent {
  color: #D946EF; font-style: italic; font-weight: 900;
  letter-spacing: -2.5px;
}
.ad-sub {
  font-size: 16.5px; color: #4B5563;
  max-width: 580px; margin: 22px auto 38px;
  line-height: 1.6; font-weight: 400;
}

/* 3-card flow with mockup-style content */
.ad-flow {
  display: flex; align-items: stretch; justify-content: center;
  gap: 14px; margin: 36px 0 38px; flex-wrap: wrap;
}
.ad-card {
  background: white; border: 1px solid #ECEEF3;
  border-radius: 18px; padding: 18px 16px 18px;
  width: 210px; max-width: 230px; min-height: 130px;
  box-shadow: 0 3px 10px rgba(26, 26, 46, 0.05);
  display: flex; flex-direction: column;
  text-align: left;
}
.ad-card-label {
  font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
  color: #9CA3AF; text-transform: uppercase; margin-bottom: 10px;
  display: flex; align-items: center; gap: 6px;
}
.ad-card-label .dot {
  width: 6px; height: 6px; border-radius: 50%;
}
.ad-card-1 .ad-card-label .dot { background: #2D3FE7; }
.ad-card-2 .ad-card-label .dot { background: #DC2626; }
.ad-card-3 .ad-card-label .dot { background: #059669; }

/* Card 1: Chat bubble */
.ad-bubble {
  background: #F4F1FB; border-radius: 14px 14px 14px 4px;
  padding: 11px 13px; font-size: 13px;
  color: #1A1A2E; line-height: 1.45; font-style: italic;
  font-weight: 500;
}
/* Card 2: Vitals readout */
.ad-vitals { display: flex; flex-direction: column; gap: 8px; }
.ad-vital-row {
  display: flex; align-items: center; justify-content: space-between;
  background: #FAFBFC; border-radius: 9px;
  padding: 8px 11px; font-size: 12.5px;
}
.ad-vital-row .lbl { color: #6B7280; font-weight: 600; letter-spacing: 0.04em; }
.ad-vital-row .val { color: #1A1A2E; font-weight: 700; font-variant-numeric: tabular-nums; }
/* Card 3: Report checklist */
.ad-report { display: flex; flex-direction: column; gap: 7px; }
.ad-report-line {
  display: flex; align-items: center; gap: 9px;
  font-size: 13px; color: #1A1A2E; font-weight: 500;
}
.ad-report-line .check {
  width: 18px; height: 18px; border-radius: 50%;
  background: #ECFDF5; color: #059669;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; flex-shrink: 0;
}
.ad-arrow {
  display: flex; align-items: center;
  font-size: 22px; color: #D946EF; font-weight: 700; opacity: 0.5;
}

/* Trust badges — inline with dot separators (Cira-style) */
.ad-trust {
  display: flex; justify-content: center; align-items: center;
  gap: 10px; flex-wrap: wrap; font-size: 12.5px;
  color: #6B7280; font-weight: 500;
  padding-top: 14px; border-top: 1px solid rgba(0,0,0,0.05);
  margin-top: 20px;
}
.ad-trust .item { white-space: nowrap; }
.ad-trust .sep-dot {
  color: #D1D5DB; font-weight: 400; font-size: 14px;
  line-height: 1;
}

@media (max-width: 640px) {
  .ad-hero { padding: 36px 22px 28px; border-radius: 22px; }
  .ad-title { font-size: 36px; letter-spacing: -1.2px; }
  .ad-title .accent { letter-spacing: -1.5px; }
  .ad-sub { font-size: 14.5px; margin: 18px auto 28px; }
  .ad-arrow { display: none; }
  .ad-card { width: 100%; max-width: 340px; padding: 14px; min-height: auto; }
  .ad-flow { gap: 10px; margin: 24px 0 28px; }
  .ad-trust { gap: 6px; font-size: 11.5px; }
  .ad-pill { font-size: 10.5px; padding: 7px 14px; }
}
</style>
"""
    body = f"""
<div class="ad-hero">
  <div class="ad-pill">✦ {d["pill_l"]} <span class="sep">|</span> <span class="gdpr">{d["pill_r"]}</span></div>
  <h1 class="ad-title">
    <span class="word">{d["h_l"]}</span>
    <span class="word">{d["h_m"]}</span><br>
    <span class="word accent">{d["h_r"]}</span>
  </h1>
  <p class="ad-sub">{d["sub"]}</p>
  <div class="ad-flow">
    <div class="ad-card ad-card-1">
      <div class="ad-card-label"><span class="dot"></span>{d["s1_lbl"]}</div>
      <div class="ad-bubble">{d["s1_text"]}</div>
    </div>
    <div class="ad-arrow">→</div>
    <div class="ad-card ad-card-2">
      <div class="ad-card-label"><span class="dot"></span>{d["s2_lbl"]}</div>
      <div class="ad-vitals">
        <div class="ad-vital-row"><span class="lbl">❤️ {d["s2_v1"]}</span><span class="val">{d["s2_v1v"]}</span></div>
        <div class="ad-vital-row"><span class="lbl">💉 {d["s2_v2"]}</span><span class="val">{d["s2_v2v"]}</span></div>
      </div>
    </div>
    <div class="ad-arrow">→</div>
    <div class="ad-card ad-card-3">
      <div class="ad-card-label"><span class="dot"></span>{d["s3_lbl"]}</div>
      <div class="ad-report">
        <div class="ad-report-line"><span class="check">✓</span>{d["s3_l1"]}</div>
        <div class="ad-report-line"><span class="check">✓</span>{d["s3_l2"]}</div>
        <div class="ad-report-line"><span class="check">✓</span>{d["s3_l3"]}</div>
      </div>
    </div>
  </div>
  <div class="ad-trust">
    <span class="item">{d["t1"]}</span><span class="sep-dot">·</span>
    <span class="item">{d["t2"]}</span><span class="sep-dot">·</span>
    <span class="item">{d["t3"]}</span><span class="sep-dot">·</span>
    <span class="item">{d["t4"]}</span><span class="sep-dot">·</span>
    <span class="item">{d["t5"]}</span>
  </div>
</div>
"""
    st.markdown(css + body, unsafe_allow_html=True)


def render_explainer_video(lang):
    """Photo-slider style walkthrough: horizontal scrollable cards (one per step).
    Mobile: swipeable with snap. Desktop: 3-4 cards visible + scroll. No iframe,
    no JS, no tabs — visible immediately."""
    el = (lang == "el")
    if el:
        steps = [
            ("01", "🩺", "#EEF6FF", "ASKLEPIOS",
             "Ο ψηφιακός σου νοσηλευτής",
             "Αξιολόγηση συμπτωμάτων με τεχνητή νοημοσύνη — γρήγορα, στα Ελληνικά."),
            ("02", "✉️", "#F0EEFE", "ΣΥΝΔΕΣΗ",
             "Σύνδεση με email",
             "Email + κωδικός μίας χρήσης. Χωρίς password, χωρίς πολύπλοκη εγγραφή."),
            ("03", "👤", "#ECFDF5", "ΠΡΟΦΙΛ",
             "Συμπλήρωσε το προφίλ σου",
             "Όνομα, ηλικία, φύλο, ιατρικό ιστορικό, αλλεργίες, φάρμακα."),
            ("04", "💬", "#FFF7ED", "ΣΥΜΠΤΩΜΑΤΑ",
             "Περίγραψε τι νιώθεις",
             "Ο Asklepios κάνει στοχευμένες ερωτήσεις — μία κάθε φορά."),
            ("05", "❤️", "#FEF2F2", "ΖΩΤΙΚΑ",
             "Μέτρηση ζωτικών — 3 επιλογές",
             "Χειροκίνητα · συσκευή · σάρωση προσώπου (καρδιακός ρυθμός)."),
            ("06", "📷", "#F0FDFA", "ΦΩΤΟ",
             "Φωτογραφία — μόνο αν χρειαστεί",
             "Προτείνεται για ορατά συμπτώματα: δερματικά, τραύματα, εξογκώματα."),
            ("07", "📋", "#FDF4FF", "ΑΝΑΦΟΡΑ",
             "Αναλυτική αναφορά υγείας",
             "Κλινική εκτίμηση με PubMed + GPT-4o δεύτερη γνώμη. PDF για τον γιατρό σου."),
        ]
        header = "Πώς λειτουργεί"
        hint   = "← σύρε για περισσότερα →"
    else:
        steps = [
            ("01", "🩺", "#EEF6FF", "ASKLEPIOS",
             "Your digital nurse",
             "AI-powered symptom assessment — fast, in your language."),
            ("02", "✉️", "#F0EEFE", "SIGN-IN",
             "Sign in with email",
             "Email + one-time code. No password, no complex registration."),
            ("03", "👤", "#ECFDF5", "PROFILE",
             "Fill in your profile",
             "Name, age, sex, medical history, allergies, medications."),
            ("04", "💬", "#FFF7ED", "SYMPTOMS",
             "Describe what you're feeling",
             "Asklepios asks targeted questions — one at a time."),
            ("05", "❤️", "#FEF2F2", "VITALS",
             "Measure vitals — 3 options",
             "Manual entry · device · face scan (heart rate only)."),
            ("06", "📷", "#F0FDFA", "PHOTO",
             "Photo — only when needed",
             "Suggested for visible symptoms: skin, wounds, lumps."),
            ("07", "📋", "#FDF4FF", "REPORT",
             "Detailed health report",
             "Clinical assessment with PubMed + GPT-4o second opinion. PDF for your doctor."),
        ]
        header = "How it works"
        hint   = "← swipe for more →"
    cards = "".join(
        f"""<div class="exp-card" style="background:{tint};">
              <div class="exp-num">{num}</div>
              <div class="exp-icon">{icon}</div>
              <div class="exp-label">{label}</div>
              <div class="exp-title">{title}</div>
              <div class="exp-sub">{sub}</div>
            </div>"""
        for (num, icon, tint, label, title, sub) in steps
    )
    st.markdown(
        f"""
<style>
.exp-section {{
  margin: 32px 0 16px;
}}
.exp-header {{
  display: flex; justify-content: space-between; align-items: baseline;
  margin: 0 4px 12px;
  font-family: 'Inter', system-ui, sans-serif;
}}
.exp-header .ttl {{
  font-size: 18px; font-weight: 700; color: #1A1A2E;
  letter-spacing: -0.01em;
}}
.exp-header .hint {{
  font-size: 11px; color: #9CA3AF; font-weight: 500;
  letter-spacing: 0.02em;
}}
.exp-scroll {{
  display: flex; gap: 12px;
  overflow-x: auto; overflow-y: hidden;
  padding: 4px 4px 18px;
  scroll-snap-type: x mandatory;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
  scrollbar-color: #CBD5E1 transparent;
}}
.exp-scroll::-webkit-scrollbar {{ height: 6px; }}
.exp-scroll::-webkit-scrollbar-thumb {{
  background: #CBD5E1; border-radius: 3px;
}}
.exp-scroll::-webkit-scrollbar-track {{ background: transparent; }}
.exp-card {{
  flex: 0 0 250px; max-width: 250px;
  border-radius: 18px; padding: 22px 20px;
  scroll-snap-align: start;
  border: 1px solid rgba(0,0,0,0.04);
  text-align: left;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.exp-num {{
  font-size: 11px; font-weight: 800; letter-spacing: 0.14em;
  color: rgba(0,0,0,0.28); margin-bottom: 12px;
}}
.exp-icon {{
  font-size: 30px; line-height: 1; margin-bottom: 10px;
}}
.exp-label {{
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.14em;
  color: #9CA3AF; text-transform: uppercase; margin-bottom: 6px;
}}
.exp-title {{
  font-size: 15px; font-weight: 700; color: #1A1A2E;
  line-height: 1.35; margin-bottom: 8px;
}}
.exp-sub {{
  font-size: 12.5px; color: #4B5563; line-height: 1.55;
}}
@media (max-width: 640px) {{
  .exp-card {{ flex: 0 0 220px; padding: 18px 16px; }}
  .exp-icon {{ font-size: 26px; }}
  .exp-title {{ font-size: 14px; }}
  .exp-sub {{ font-size: 12px; }}
}}
</style>
<div class="exp-section">
  <div class="exp-header"><span class="ttl">{header}</span><span class="hint">{hint}</span></div>
  <div class="exp-scroll">{cards}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_login_screen():
    """Full-page login shown at the very start when auth is enabled."""
    lang = st.session_state.lang
    c1, c2 = st.columns([6,1])
    with c2:
        if st.button("🇬🇧 EN" if lang=="el" else "🇬🇷 ΕΛ", key="login_lang"):
            st.session_state.lang = "en" if lang=="el" else "el"; st.rerun()
    # Value-prop banner: tells new visitors what the app does at a glance.
    render_ad_banner(lang)
    # Login form
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        render_login_gate()
    # "How it works" photo-slider — visible inline, no click needed.
    # Horizontal scrollable cards, mobile-swipeable.
    render_explainer_video(lang)
    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>', unsafe_allow_html=True)

def save_feedback(rating, comment=""):
    """Store a minimal, non-medical feedback row in Supabase. No report/identifiers."""
    sb = _supabase_client()
    if not sb:
        return False  # demo mode: nothing stored
    try:
        sb.table("feedback").insert({
            "user_email": st.session_state.get("auth_user", ""),
            "rating": rating,
            "comment": (comment or "")[:1000],
            "lang": st.session_state.lang,
        }).execute()
        return True
    except Exception:
        return False

# ── NCBI HELPERS ──────────────────────────────────────────────────────────────
def pubmed_search(query, n=3):
    try:
        p = urllib.parse.urlencode({"db":"pubmed","term":query,"retmax":n,"retmode":"json","api_key":get_ncbi_key()})
        with urllib.request.urlopen(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{p}", timeout=8) as r:
            ids = json.loads(r.read()).get("esearchresult",{}).get("idlist",[])
        if not ids: return []
        p2 = urllib.parse.urlencode({"db":"pubmed","id":",".join(ids),"retmode":"json","api_key":get_ncbi_key()})
        with urllib.request.urlopen(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{p2}", timeout=8) as r:
            res = json.loads(r.read()).get("result",{})
        out = []
        for pmid in ids:
            a = res.get(pmid,{})
            out.append({
                "pmid": pmid, "title": a.get("title","—"),
                "authors": ", ".join(x.get("name","") for x in a.get("authors",[])[:2]),
                "journal": a.get("source",""), "date": a.get("pubdate",""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
        return out
    except: return []

# Pillar-targeted PubMed query: scopes results to *high-evidence* publication
# types (Practice Guideline, Systematic Review, Meta-Analysis, Review) crossed
# with the MeSH heading that matches the pillar — so each recommendation gets
# 1-2 references from guideline-quality literature rather than single studies.
_PILLAR_MESH = {
    "exercise":  '("Exercise Therapy"[MeSH] OR "Exercise"[MeSH] OR "Physical Activity"[MeSH:NoExp] OR "Exercise Movement Techniques"[MeSH])',
    "nutrition": '("Diet Therapy"[MeSH] OR "Diet"[MeSH] OR "Nutrition Therapy"[MeSH] OR "Diet, Healthy"[MeSH])',
    "lifestyle": '("Life Style"[MeSH] OR "Risk Reduction Behavior"[MeSH] OR "Health Behavior"[MeSH])',
}
_PILLAR_PTYPE = '(Practice Guideline[ptyp] OR Systematic Review[ptyp] OR Meta-Analysis[ptyp] OR Review[ptyp])'

def pubmed_pillar_search(condition, pillar, n=2):
    """High-evidence PubMed search for one of: 'exercise', 'nutrition', 'lifestyle'.
    Returns the same list-of-dicts shape as pubmed_search. Falls back to a
    broader keyword query if the strict MeSH+ptyp combo returns nothing."""
    if not condition: return []
    mesh = _PILLAR_MESH.get(pillar)
    if not mesh: return []
    cond_q = condition.strip()
    # Try strict (MeSH + high-evidence ptype) first
    strict = f"{cond_q} AND {mesh} AND {_PILLAR_PTYPE}"
    res = pubmed_search(strict, n=n)
    if res:
        return res
    # Fallback: drop ptype filter — still MeSH-scoped, just any pub type
    return pubmed_search(f"{cond_q} AND {mesh}", n=n)

def rxnorm_interactions(names):
    try:
        cuis = []
        for name in names:
            p = urllib.parse.urlencode({"name": name.split()[0]})
            with urllib.request.urlopen(f"https://rxnav.nlm.nih.gov/REST/rxcui.json?{p}", timeout=6) as r:
                ids = json.loads(r.read()).get("idGroup",{}).get("rxnormId",[])
                if ids: cuis.append(ids[0])
        if len(cuis) < 2: return None
        p2 = urllib.parse.urlencode({"rxcuis": " ".join(cuis)})
        with urllib.request.urlopen(f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?{p2}", timeout=8) as r:
            data = json.loads(r.read())
        pairs = data.get("fullInteractionTypeGroup",[])
        if not pairs: return "\u2705 RxNorm: No known interactions found."
        lines = []
        for g in pairs:
            src = g.get("sourceName","")
            for t2 in g.get("fullInteractionType",[]):
                for pair in t2.get("interactionPair",[]):
                    sev  = pair.get("severity","")
                    desc = pair.get("description","")
                    drugs = " + ".join(c.get("minConceptItem",{}).get("name","") for c in pair.get("interactionConcept",[]))
                    lines.append(f"- **{drugs}** [{sev}] \u2014 {desc} *({src})*")
        return "\n".join(lines) if lines else "\u2705 RxNorm: No known interactions found."
    except: return None

# ── GPT-4o ────────────────────────────────────────────────────────────────────
def gpt4o(prompt, system="", max_tokens=900):
    try:
        oai = get_openai_key()
        if not oai: return None
        body = json.dumps({
            "model": "gpt-4o",
            "max_tokens": max_tokens,
            "messages": [{"role":"system","content":system},{"role":"user","content":prompt}] if system
                        else [{"role":"user","content":prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Content-Type":"application/json","Authorization":f"Bearer {oai}"}
        )
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"GPT-4o unavailable: {e}"

# ── WHISPER (voice → text) ────────────────────────────────────────────────────
def whisper_transcribe(audio_bytes, filename="recording.webm", lang="el"):
    """Greek/English voice → text via OpenAI Whisper. Reuses the existing
    OPENAI_API_KEY — no new dependency. The transcribed text is shown to the
    user for review/edit BEFORE sending to chat (safety + privacy).
    
    Audio is sent to OpenAI for processing but NEVER stored on our side.
    Returns (text, error) where one of them is None.
    """
    key = get_openai_key()
    if not key:
        return None, "⚠️ OpenAI API key not set."
    try:
        import requests
        # Map common audio MIME types so Whisper recognises the format
        mime = "audio/webm"
        if filename.lower().endswith(".wav"):  mime = "audio/wav"
        elif filename.lower().endswith(".mp3"): mime = "audio/mpeg"
        elif filename.lower().endswith(".m4a"): mime = "audio/mp4"
        files = {"file": (filename, audio_bytes, mime)}
        data = {
            "model": "whisper-1",
            "language": lang if lang in ("el","en") else "el",
            "response_format": "text",
        }
        r = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files=files, data=data, timeout=60,
        )
        if r.status_code == 200:
            return r.text.strip(), None
        return None, f"⚠️ Whisper {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"⚠️ {e}"

# ── CLAUDE ────────────────────────────────────────────────────────────────────
def claude(messages, system="", max_tokens=1200, timeout=60):
    """Call Claude via raw HTTP."""
    key = get_claude_key()
    if not key:
        return "\u26a0\ufe0f Claude API key not set."
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data["content"][0]["text"]
    except urllib.error.URLError as e:
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            return "\u26a0\ufe0f Request timed out. Please try again."
        return f"\u26a0\ufe0f Claude error: {e}"
    except Exception as e:
        return f"\u26a0\ufe0f Claude error: {e}"

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "lang": "el",
    "screen": "home",
    "profile": {},
    "vitals": {},
    "vitals_analysis": "",
    "triage_chat": [],
    "triage_ready": False,
    "report": "",
    "report_pubmed": [],
    "report_gpt": "",
    "report_recs": None,  # {"exercise": "...", "nutrition": "...", "lifestyle": "..."} from Claude
    "report_recs_refs": {},  # {"exercise": [...refs...], "nutrition": [...], "lifestyle": [...]}
    "photo_findings": [],  # list of dicts — visual analyses added to assessment
    "lab_findings": [],    # list of dicts — lab PDF/image analyses added to assessment
    "medications": [],
    "med_inputs": [],
    "symptom_chips": [],
    "fb_rating": "",
    "fb_sent": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── TRANSLATIONS ──────────────────────────────────────────────────────────────
T = {
    "el": {
        "title": "Asklepios",
        "subtitle": "\u039f AI \u039d\u03bf\u03c3\u03b7\u03bb\u03b5\u03c5\u03c4\u03ae\u03c2 \u03c3\u03bf\u03c5",
        "tagline": "\u0388\u03b3\u03ba\u03c5\u03c1\u03b7 \u03b9\u03b1\u03c4\u03c1\u03b9\u03ba\u03ae \u03c0\u03bb\u03b7\u03c1\u03bf\u03c6\u03cc\u03c1\u03b7\u03c3\u03b7 \u00b7 \u03a0\u03ac\u03bd\u03c4\u03b1 \u03b4\u03af\u03c0\u03bb\u03b1 \u03c3\u03bf\u03c5",
        "start": "\u039e\u03b5\u03ba\u03af\u03bd\u03b1 \u0395\u03ba\u03c4\u03af\u03bc\u03b7\u03c3\u03b7",
        "disclaimer_main": "\u26a0\ufe0f \u039f Asklepios \u03c0\u03b1\u03c1\u03ad\u03c7\u03b5\u03b9 \u03c0\u03bb\u03b7\u03c1\u03bf\u03c6\u03bf\u03c1\u03af\u03b5\u03c2 \u03c5\u03b3\u03b5\u03af\u03b1\u03c2 \u03b1\u03c0\u03bf\u03ba\u03bb\u03b5\u03b9\u03c3\u03c4\u03b9\u03ba\u03ac \u03b3\u03b9\u03b1 \u03b5\u03bd\u03b7\u03bc\u03b5\u03c1\u03c9\u03c4\u03b9\u03ba\u03bf\u03cd\u03c2 \u03c3\u03ba\u03bf\u03c0\u03bf\u03cd\u03c2. \u0394\u03b5\u03bd \u03b1\u03bd\u03c4\u03b9\u03ba\u03b1\u03b8\u03b9\u03c3\u03c4\u03ac \u03b9\u03b1\u03c4\u03c1\u03b9\u03ba\u03ae \u03b4\u03b9\u03ac\u03b3\u03bd\u03c9\u03c3\u03b7 \u03ae \u03b8\u03b5\u03c1\u03b1\u03c0\u03b5\u03af\u03b1. \u03a3\u03b5 \u03b5\u03c0\u03b5\u03af\u03b3\u03bf\u03c5\u03c3\u03b1 \u03b1\u03bd\u03ac\u03b3\u03ba\u03b7 \u03ba\u03b1\u03bb\u03ad\u03c3\u03c4\u03b5 **166** (\u0395\u039a\u0391\u0392) \u03ae **112**.",
        "emergency": "\U0001f6a8 \u03a3\u0395 \u0395\u03a0\u0395\u0399\u0393\u039f\u03a5\u03a3\u0391 \u0391\u039d\u0391\u0393\u039a\u0397: \u039a\u0391\u039b\u0395\u03a3\u03a4\u0395 166 (\u0395\u039a\u0391\u0392) \u03ae 112",
        "name": "\u038c\u03bd\u03bf\u03bc\u03b1", "age": "\u0397\u03bb\u03b9\u03ba\u03af\u03b1", "sex": "\u03a6\u03cd\u03bb\u03bf",
        "male": "\u0386\u03bd\u03b4\u03c1\u03b1\u03c2", "female": "\u0393\u03c5\u03bd\u03b1\u03af\u03ba\u03b1", "other": "\u0386\u03bb\u03bb\u03bf",
        "history": "\u0399\u03b1\u03c4\u03c1\u03b9\u03ba\u03cc \u03b9\u03c3\u03c4\u03bf\u03c1\u03b9\u03ba\u03cc (\u03c0\u03c1\u03bf\u03b7\u03b3\u03bf\u03cd\u03bc\u03b5\u03bd\u03b5\u03c2 \u03c0\u03b1\u03b8\u03ae\u03c3\u03b5\u03b9\u03c2, \u03c7\u03b5\u03b9\u03c1\u03bf\u03c5\u03c1\u03b3\u03b5\u03af\u03b1)",
        "allergies": "\u0391\u03bb\u03bb\u03b5\u03c1\u03b3\u03af\u03b5\u03c2",
        "meds": "\u03a4\u03c1\u03ad\u03c7\u03bf\u03bd\u03c4\u03b1 \u03c6\u03ac\u03c1\u03bc\u03b1\u03ba\u03b1 / \u03c3\u03c5\u03bc\u03c0\u03bb\u03b7\u03c1\u03ce\u03bc\u03b1\u03c4\u03b1",
        "next": "\u0395\u03c0\u03cc\u03bc\u03b5\u03bd\u03bf \u2192",
        "back": "\u2190 \u03a0\u03af\u03c3\u03c9",
        "vitals_title": "\u0396\u03c9\u03c4\u03b9\u03ba\u03ad\u03c2 \u0395\u03bd\u03b4\u03b5\u03af\u03be\u03b5\u03b9\u03c2",
        "vitals_sub": "\u0395\u03b9\u03c3\u03ac\u03b3\u03b5\u03c4\u03b5 \u03c4\u03b9\u03c2 \u03bc\u03b5\u03c4\u03c1\u03ae\u03c3\u03b5\u03b9\u03c2 \u03c3\u03b1\u03c2.",
        "hr": "\u039a\u03b1\u03c1\u03b4\u03b9\u03b1\u03ba\u03cc\u03c2 \u03a1\u03c5\u03b8\u03bc\u03cc\u03c2 (bpm)",
        "bp_sys": "\u0391\u03c1\u03c4\u03b7\u03c1\u03b9\u03b1\u03ba\u03ae \u03a0\u03af\u03b5\u03c3\u03b7 \u2014 \u03a3\u03c5\u03c3\u03c4\u03bf\u03bb\u03b9\u03ba\u03ae (mmHg)",
        "bp_dia": "\u0391\u03c1\u03c4\u03b7\u03c1\u03b9\u03b1\u03ba\u03ae \u03a0\u03af\u03b5\u03c3\u03b7 \u2014 \u0394\u03b9\u03b1\u03c3\u03c4\u03bf\u03bb\u03b9\u03ba\u03ae (mmHg)",
        "br": "\u0391\u03bd\u03b1\u03c0\u03bd\u03b5\u03c5\u03c3\u03c4\u03b9\u03ba\u03cc\u03c2 \u03a1\u03c5\u03b8\u03bc\u03cc\u03c2 (/min)",
        "spo2": "SpO2 (%)",
        "temp": "\u0398\u03b5\u03c1\u03bc\u03bf\u03ba\u03c1\u03b1\u03c3\u03af\u03b1 (\u00b0C)",
        "weight": "\u0392\u03ac\u03c1\u03bf\u03c2 (kg)",
        "height": "\u038e\u03c8\u03bf\u03c2 (cm)",
        "analyse_vitals": "\u0391\u03bd\u03ac\u03bb\u03c5\u03c3\u03b7 \u0396\u03c9\u03c4\u03b9\u03ba\u03ce\u03bd",
        "triage_title": "\u0395\u03ba\u03c4\u03af\u03bc\u03b7\u03c3\u03b7 \u03a3\u03c5\u03bc\u03c0\u03c4\u03c9\u03bc\u03ac\u03c4\u03c9\u03bd",
        "triage_sub": "\u03a0\u03b5\u03c1\u03b9\u03b3\u03c1\u03ac\u03c8\u03c4\u03b5 \u03c4\u03b1 \u03c3\u03c5\u03bc\u03c0\u03c4\u03ce\u03bc\u03b1\u03c4\u03ac \u03c3\u03b1\u03c2. \u039f Asklepios \u03b8\u03b1 \u03c3\u03b1\u03c2 \u03ba\u03ac\u03bd\u03b5\u03b9 \u03ba\u03b1\u03c4\u03b5\u03c5\u03b8\u03c5\u03bd\u03cc\u03bc\u03b5\u03bd\u03b5\u03c2 \u03b5\u03c1\u03c9\u03c4\u03ae\u03c3\u03b5\u03b9\u03c2.",
        "triage_placeholder": "\u03a0.\u03c7. \u0388\u03c7\u03c9 \u03c0\u03bf\u03bd\u03bf\u03ba\u03ad\u03c6\u03b1\u03bb\u03bf \u03c4\u03c1\u03b9\u03ce\u03bd \u03b7\u03bc\u03b5\u03c1\u03ce\u03bd \u03bc\u03b5 \u03bd\u03b1\u03c5\u03c4\u03af\u03b1...",
        "generate_report": "\u0394\u03b7\u03bc\u03b9\u03bf\u03c5\u03c1\u03b3\u03af\u03b1 \u03a0\u03bb\u03ae\u03c1\u03bf\u03c5\u03c2 \u0391\u03bd\u03b1\u03c6\u03bf\u03c1\u03ac\u03c2",
        "report_title": "\u039b\u03b5\u03c0\u03c4\u03bf\u03bc\u03b5\u03c1\u03ae\u03c2 \u0395\u03ba\u03c4\u03af\u03bc\u03b7\u03c3\u03b7 \u03a5\u03b3\u03b5\u03af\u03b1\u03c2",
        "second_opinion": "\u0394\u03b5\u03cd\u03c4\u03b5\u03c1\u03b7 \u0393\u03bd\u03ce\u03bc\u03b7 GPT-4o",
        "pubmed": "\u0395\u03c0\u03b9\u03c3\u03c4\u03b7\u03bc\u03bf\u03bd\u03b9\u03ba\u03ad\u03c2 \u0391\u03bd\u03b1\u03c6\u03bf\u03c1\u03ad\u03c2 PubMed",
        "skip_vitals": "\u03a0\u03b1\u03c1\u03ac\u03bb\u03b5\u03b9\u03c8\u03b7 (\u03c7\u03c9\u03c1\u03af\u03c2 \u03bc\u03b5\u03c4\u03c1\u03ae\u03c3\u03b5\u03b9\u03c2)",
    },
    "en": {
        "title": "Asklepios",
        "subtitle": "Your AI Nurse",
        "tagline": "Evidence-based health guidance · Always by your side",
        "start": "Start Assessment",
        "disclaimer_main": "\u26a0\ufe0f Asklepios provides health information for informational purposes only. It does not replace medical diagnosis or treatment. In an emergency call **166** (EKAB) or **112**.",
        "emergency": "\U0001f6a8 EMERGENCY: CALL 166 (EKAB) or 112",
        "name": "Name", "age": "Age", "sex": "Biological Sex",
        "male": "Male", "female": "Female", "other": "Other",
        "history": "Medical history (conditions, surgeries)",
        "allergies": "Allergies",
        "meds": "Current medications / supplements",
        "next": "Next \u2192",
        "back": "\u2190 Back",
        "vitals_title": "Your Vitals",
        "vitals_sub": "Enter your measurements.",
        "hr": "Heart Rate (bpm)",
        "bp_sys": "Blood Pressure \u2014 Systolic (mmHg)",
        "bp_dia": "Blood Pressure \u2014 Diastolic (mmHg)",
        "br": "Breathing Rate (/min)",
        "spo2": "SpO2 (%)",
        "temp": "Temperature (\u00b0C)",
        "weight": "Weight (kg)",
        "height": "Height (cm)",
        "analyse_vitals": "Analyse Vitals",
        "triage_title": "Symptom Assessment",
        "triage_sub": "Describe your symptoms. Asklepios will ask targeted follow-up questions.",
        "triage_placeholder": "E.g. I have had a headache for three days with nausea...",
        "generate_report": "Generate Full Clinical Report",
        "report_title": "Detailed Health Assessment",
        "second_opinion": "GPT-4o Second Opinion",
        "pubmed": "PubMed Evidence",
        "skip_vitals": "Skip (no measurements)",
    }
}

def t(key): return T[st.session_state.lang].get(key, key)


def render_topbar():
    """Top-right bar visible on every post-login screen: language toggle + logout.
    Centralises both so each screen does not duplicate them."""
    lang = st.session_state.lang
    _t1, _t2, _t3 = st.columns([7, 1, 1])
    with _t2:
        if st.button(("🇬🇧 EN" if lang=="el" else "🇬🇷 ΕΛ"),
                     key="topbar_lang", use_container_width=True):
            st.session_state.lang = "en" if lang=="el" else "el"
            st.rerun()
    with _t3:
        if is_logged_in():
            if st.button("🚪 " + ("Έξοδος" if lang=="el" else "Logout"),
                         key="topbar_logout", use_container_width=True):
                logout()
                st.rerun()


def render_doc_header(title_el, title_en, *, icon="📋",
                      sub_el=None, sub_en=None, show_date=True):
    """Compact doc-template style header card for each screen.
    White card with blue circular logo, org caps, friendly title, optional subtitle
    and date. Establishes the medical-form aesthetic on intake/vitals/triage/report
    while keeping Streamlit widgets unchanged below."""
    lang = st.session_state.lang
    title = title_el if lang == "el" else title_en
    sub = (sub_el if lang == "el" else sub_en) or ""
    org = "ASKLEPIOS · AI ΝΟΣΗΛΕΥΤΗΣ" if lang == "el" else "ASKLEPIOS · AI NURSE"
    date_str = datetime.now().strftime("%d.%m.%Y")
    date_lbl = "ΗΜΕΡ." if lang == "el" else "DATE"
    date_html = (
        f'<div class="dph-date"><div class="dph-date-lbl">{date_lbl}</div>'
        f'<div class="dph-date-val">{date_str}</div></div>'
    ) if show_date else ""
    sub_html = f'<div class="dph-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
<style>
.doc-page-head {{
  display: flex; align-items: center; gap: 16px;
  padding: 18px 22px;
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  margin: 4px 0 22px;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}}
.dph-logo {{
  width: 50px; height: 50px; border-radius: 50%;
  background: #DBEAFE;
  display: flex; align-items: center; justify-content: center;
  font-size: 23px; flex-shrink: 0;
}}
.dph-text {{ flex: 1; min-width: 0; }}
.dph-org {{
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.14em;
  color: #6B7280; text-transform: uppercase; margin-bottom: 3px;
}}
.dph-title {{
  font-size: 19px; font-weight: 700; color: #111827;
  letter-spacing: -0.015em; line-height: 1.2;
}}
.dph-sub {{
  font-size: 12.5px; color: #6B7280; margin-top: 3px; font-weight: 500;
}}
.dph-date {{
  text-align: right; flex-shrink: 0;
  border-left: 1px solid #E5E7EB; padding-left: 14px;
}}
.dph-date-lbl {{
  font-size: 9px; font-weight: 700; letter-spacing: 0.14em;
  color: #9CA3AF; text-transform: uppercase;
}}
.dph-date-val {{
  font-size: 13px; font-weight: 700; color: #111827;
  font-variant-numeric: tabular-nums; margin-top: 2px;
}}
@media (max-width: 640px) {{
  .doc-page-head {{ padding: 14px 16px; gap: 12px; }}
  .dph-logo {{ width: 42px; height: 42px; font-size: 19px; }}
  .dph-title {{ font-size: 16px; }}
  .dph-sub {{ font-size: 11.5px; }}
  .dph-date {{ display: none; }}
}}
</style>
<div class="doc-page-head">
  <div class="dph-logo">{icon}</div>
  <div class="dph-text">
    <div class="dph-org">{org}</div>
    <div class="dph-title">{title}</div>
    {sub_html}
  </div>
  {date_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_stepper(current):
    steps_el = ["1 Στοιχεία","2 Ζωτικές","3 Συμπτώματα","4 Αναφορά"]
    steps_en = ["1 Profile","2 Vitals","3 Symptoms","4 Report"]
    steps = steps_el if st.session_state.lang=="el" else steps_en
    order = ["intake","vitals","triage","report"]
    cur_i = order.index(current) if current in order else 0
    html = '<div class="kira-stepper">'
    for i, label in enumerate(steps):
        cls = "done" if i < cur_i else ("active" if i == cur_i else "")
        icon = "✓" if i < cur_i else str(i+1)
        html += f'<div class="kira-step {cls}"><div class="kira-step-circle">{icon}</div><div class="kira-step-label">{label}</div></div>'
        if i < len(steps)-1:
            line_cls = "done" if i < cur_i else ""
            html += f'<div class="kira-step-line {line_cls}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def classify_vitals(v, age=None):
    """Classify vitals as green/yellow/red. Age-aware: pediatric ranges differ
    significantly from adult, especially HR and BR.
    
    Reference ranges (PALS/AHA + WHO):
      Infant <1y:   HR 100-160, BR 30-60, SBP >70+(age×2)
      Toddler 1-3:  HR 90-150,  BR 24-40
      Preschool 3-5: HR 80-140, BR 22-34
      School 6-12:  HR 70-120,  BR 18-30
      Adolescent 13-17: HR 60-100, BR 12-20
      Adult 18+:    HR 60-100,  BR 12-20
    
    Temp, SpO2, BMI are essentially the same across ages (pediatric BMI uses
    percentiles, but our coarse green/yellow/red still gives useful signal).
    """
    status = {}
    a = age if age is not None else 99  # treat unknown as adult
    
    # Heart rate — age-stratified
    hr = v.get("hr")
    if hr:
        if a < 1:
            g_lo, g_hi, y_lo, y_hi = 100, 160, 90, 180
        elif a < 3:
            g_lo, g_hi, y_lo, y_hi = 90, 150, 80, 170
        elif a < 6:
            g_lo, g_hi, y_lo, y_hi = 80, 140, 70, 160
        elif a < 13:
            g_lo, g_hi, y_lo, y_hi = 70, 120, 60, 140
        else:
            g_lo, g_hi, y_lo, y_hi = 60, 100, 50, 110
        if g_lo <= hr <= g_hi:  status["hr"] = "green"
        elif y_lo <= hr <= y_hi: status["hr"] = "yellow"
        else:                   status["hr"] = "red"
    
    # Blood pressure
    sys_ = v.get("bp_sys"); dia = v.get("bp_dia")
    if sys_ and dia:
        if a < 13:
            # Pediatric: rough rule "70 + 2×age" for hypotension threshold,
            # pediatric hypertension >95th percentile (~ 1.2× normal). Use
            # coarse ranges — recommend physician for accurate pediatric BP.
            expected_sys = 90 + (a * 2) if a >= 1 else 70 + (a * 2)
            if sys_ < expected_sys - 15 or sys_ > expected_sys + 25:
                status["bp"] = "red"
            elif sys_ < expected_sys - 5 or sys_ > expected_sys + 15:
                status["bp"] = "yellow"
            else:
                status["bp"] = "green"
        else:
            if sys_ < 120 and dia < 80:        status["bp"] = "green"
            elif sys_ < 130:                   status["bp"] = "yellow"
            elif sys_ < 140 or dia < 90:       status["bp"] = "yellow"
            else:                              status["bp"] = "red"
    
    # Breathing rate — age-stratified
    br = v.get("br")
    if br:
        if a < 1:
            g_lo, g_hi, y_lo, y_hi = 30, 60, 24, 70
        elif a < 3:
            g_lo, g_hi, y_lo, y_hi = 24, 40, 20, 50
        elif a < 6:
            g_lo, g_hi, y_lo, y_hi = 22, 34, 18, 40
        elif a < 13:
            g_lo, g_hi, y_lo, y_hi = 18, 30, 14, 36
        else:
            g_lo, g_hi, y_lo, y_hi = 12, 20, 10, 24
        if g_lo <= br <= g_hi:   status["br"] = "green"
        elif y_lo <= br <= y_hi: status["br"] = "yellow"
        else:                    status["br"] = "red"
    
    # SpO2 — same across ages
    spo2 = v.get("spo2")
    if spo2:
        if spo2 >= 95:   status["spo2"] = "green"
        elif spo2 >= 90: status["spo2"] = "yellow"
        else:            status["spo2"] = "red"
    
    # Temperature — same
    temp = v.get("temp")
    if temp:
        if 36.1 <= temp <= 37.2:  status["temp"] = "green"
        elif 37.3 <= temp <= 38.0: status["temp"] = "yellow"
        else:                      status["temp"] = "red"
    
    # BMI — only for adults (pediatric BMI requires percentile charts)
    w = v.get("weight"); h = v.get("height")
    if w and h:
        bmi = w / ((h/100)**2); v["bmi"] = round(bmi, 1)
        if a >= 18:
            if 18.5 <= bmi <= 24.9:  status["bmi"] = "green"
            elif 25 <= bmi <= 29.9:  status["bmi"] = "yellow"
            else:                    status["bmi"] = "red"
        # For pediatric, we don't classify — the value is recorded but no
        # green/yellow/red without percentile data
    
    return status

def demographic_bp_risk(age, bmi, hr, weight=None, height=None):
    """
    Evidence-based BP risk classification using demographic features.
    Based on: Chowdhury et al. (2020) - top ReliefF features for BP estimation.
    Returns: dict with risk_level, sbp_range, dbp_range, explanation
    """
    score = 0
    factors = []

    # Age — strongest demographic predictor (Feature #105 in paper)
    if age >= 70:   score += 4; factors.append("age ≥70" if True else "")
    elif age >= 60: score += 3; factors.append("age 60-69")
    elif age >= 50: score += 2; factors.append("age 50-59")
    elif age >= 40: score += 1; factors.append("age 40-49")

    # BMI — second strongest (Feature #107)
    if bmi:
        if bmi >= 35:   score += 3; factors.append("BMI ≥35 (obese II)")
        elif bmi >= 30: score += 2; factors.append("BMI 30-34 (obese I)")
        elif bmi >= 25: score += 1; factors.append("BMI 25-29 (overweight)")

    # Heart Rate — Feature #106
    if hr:
        if hr > 90:   score += 2; factors.append("elevated HR")
        elif hr > 80: score += 1; factors.append("high-normal HR")
        elif hr < 55: score -= 1; factors.append("low HR (fit/athletic)")

    # Weight/Height ratio proxy if BMI not computed yet
    if weight and height and not bmi:
        bmi_calc = weight / ((height/100)**2)
        if bmi_calc >= 30: score += 2
        elif bmi_calc >= 25: score += 1

    # Map score to risk level + estimated range
    if score <= 0:
        return {"level":"optimal","color":"#10B981","label_el":"Βέλτιστη","label_en":"Optimal",
                "sbp":"<115","dbp":"<75","note_el":"Εξαιρετικό καρδιαγγειακό προφίλ.","note_en":"Excellent cardiovascular profile.","score":score}
    elif score <= 2:
        return {"level":"normal","color":"#10B981","label_el":"Φυσιολογική","label_en":"Normal",
                "sbp":"115-129","dbp":"75-84","note_el":"Φυσιολογικά επίπεδα για το προφίλ σας.","note_en":"Normal levels for your profile.","score":score}
    elif score <= 4:
        return {"level":"elevated","color":"#F59E0B","label_el":"Ελαφρά Αυξημένη","label_en":"Elevated Risk",
                "sbp":"130-144","dbp":"85-89","note_el":"Σχετικά αυξημένος κίνδυνος. Μέτρηση πίεσης συνιστάται.","note_en":"Moderately elevated risk. BP measurement advised.","score":score}
    elif score <= 6:
        return {"level":"high","color":"#EF4444","label_el":"Υψηλός Κίνδυνος","label_en":"High Risk",
                "sbp":"140-159","dbp":"90-99","note_el":"Αυξημένος κίνδυνος υπέρτασης. Επισκεφθείτε γιατρό.","note_en":"Elevated hypertension risk. See a doctor.","score":score}
    else:
        return {"level":"very_high","color":"#DC2626","label_el":"Πολύ Υψηλός Κίνδυνος","label_en":"Very High Risk",
                "sbp":"≥160","dbp":"≥100","note_el":"Πολύ υψηλός κίνδυνος. Απαιτείται ιατρική αξιολόγηση.","note_en":"Very high risk. Medical evaluation required.","score":score}


KIRA_SYSTEM_EL = """Είσαι ο Asklepios — AI νοσηλευτής για Έλληνες χρήστες. Είσαι κλινικά ακριβής, άμεσος και υποστηρικτικός.
Ρόλος: Τριάζ συμπτωμάτων (μία ερώτηση κάθε φορά), ερμηνεία ζωτικών, φάρμακα, ελληνικό σύστημα υγείας (ΕΟΠΥΥ, ΕΟΔΥ, ΕΟΦ).
Φωτογραφία: Αν το σύμπτωμα είναι οπτικό (δέρμα/εξάνθημα, μάτι, τραύμα/πληγή, στόμα/λαιμός, νύχια, ορατή αλλοίωση), αφού κάνεις την αρχική σου εκτίμηση πρότεινε στον χρήστη να ανεβάσει φωτογραφία από την επιλογή «📷 Ανάλυση φωτογραφίας» πιο κάτω, για πιο ακριβή εκτίμηση. Για μη-οπτικά συμπτώματα (π.χ. πονοκέφαλος, ζάλη) ΜΗΝ ζητάς φωτογραφία. Η φωτογραφία είναι ΠΡΟΑΙΡΕΤΙΚΗ: αν ο χρήστης δεν ανεβάσει ή δεν θέλει, ΣΥΝΕΧΙΣΕ κανονικά την εκτίμηση χωρίς να σταματάς, να περιμένεις ή να επιμένεις.
Κανόνες: Πάντα συστήνεις επαγγελματία. Κόκκινες σημαίες → 166/112. Όταν έχεις αρκετά: "Έχω αρκετά στοιχεία — μπορούμε να δημιουργήσουμε πλήρη αναφορά." Μία ερώτηση κάθε φορά.
Ζωτικά: Αν τα συμπτώματα είναι καρδιακά/αυτόνομα (αίσθημα παλμών, ταχυπαλμία, πόνος/σφίξιμο στο στήθος, δύσπνοια, ζάλη, λιποθυμία, κρύος ιδρώτας/εφίδρωση), πρότεινε ήπια στον χρήστη να μετρήσει ζωτικά (καρδιακός ρυθμός/πίεση) — ΠΡΟΑΙΡΕΤΙΚΟ, συνέχισε κανονικά αν δεν το κάνει."""

KIRA_SYSTEM_EN = """You are Asklepios — an AI nurse for users in Greece. Clinically accurate, direct, supportive.
Role: Symptom triage (one question at a time), vitals interpretation, medications, Greek health system (EOPYY, EODY, EOF).
Photo: If the symptom is visual (skin/rash, eye, wound, mouth/throat, nails, any visible lesion), after giving your initial assessment, invite the user to upload a photo via the "📷 Photo analysis" option below for a more accurate assessment. For non-visual symptoms (e.g. headache, dizziness) do NOT ask for a photo. The photo is OPTIONAL: if the user doesn't upload one or declines, CONTINUE the assessment normally — do not stop, wait, or insist.
Rules: Always recommend a professional. Red flags → 166/112. When ready: "I have enough information — we can generate a full clinical report." One question at a time.
Vitals: If the symptoms are cardiac/autonomic (palpitations, racing heart, chest pain/tightness, shortness of breath, dizziness, fainting, cold sweat/sweating), gently suggest the user measure vitals (heart rate/blood pressure) — OPTIONAL, continue normally if they don't."""

def kira_system(): return KIRA_SYSTEM_EL if st.session_state.lang=="el" else KIRA_SYSTEM_EN

# Symptoms where measuring a specific vital genuinely adds value → surface the
# relevant measurement contextually instead of forcing vitals on everyone.
def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")
# Each category maps symptom roots → the vital that helps. "scan"=True only where
# the camera face-scan can actually produce the value (heart rate → cardiac only).
_VITAL_CATEGORIES = [
    {"key":"cardio","scan":True,
     "el":"καρδιακός ρυθμός / πίεση","en":"heart rate / blood pressure",
     # Cardiac symptoms only. We deliberately do NOT use generic "ιδρωτ/ιδρωσ/
     # ιδρων/εφιδρ" (sweating) — those false-positive on workout/fever/photo
     # descriptions. Cold sweat ("κρυος ιδρωτ" / "cold sweat") is cardiac and
     # is checked specifically below.
     "roots":["παλμ","ταχυκαρδ","αρρυθμ","στηθ","θωρακ","λιποθυμ","λιγοθυμ",
              "κρυος ιδρωτ","ζαλ",
              "palpit","racing heart","irregular heart","tachycard","arrhythm",
              "chest pain","chest tightness","faint","cold sweat","dizz","lightheaded","light-headed"]},
    {"key":"bp","scan":False,
     "el":"αρτηριακή πίεση","en":"blood pressure",
     "roots":["πιεση","υπερτασ","υποτασ","αρτηριακ",
              "blood pressure","hypertens","hypotens"]},
    {"key":"temp","scan":False,
     "el":"θερμοκρασία","en":"temperature",
     "roots":["πυρετ","θερμοκρασ","δεκατ","εμπυρετ","ριγος","ριγη","κρυαδ",
              "fever","febrile","chills","temperature","high temp"]},
    {"key":"resp","scan":False,
     "el":"οξυγόνο (SpO₂) & αναπνοές","en":"oxygen (SpO₂) & breathing",
     "roots":["δυσπν","βηχ","ασθμ","πνευμον","αναπν","λαχαν","συριγμ","βρογχ","κορον","covid",
              "cough","wheez","asthma","pneumonia","breathless","short of breath",
              "shortness of breath","respiratory","oxygen"]},
]
def _relevant_vitals():
    # Only consider what the USER actually reported — NOT photo-analysis text
    # we injected as user messages. Those are Claude's AI descriptions and
    # routinely list cardiac warning signs (sweating, palpitations) even for
    # unrelated cases like an elbow lump, which would falsely trigger nudges.
    _PHOTO_PREFIXES = ("Αποτέλεσμα φωτογραφικής", "Photo analysis result")
    user_msgs = [m["content"] for m in st.session_state.triage_chat
                 if m["role"] == "user"
                 and not str(m["content"]).startswith(_PHOTO_PREFIXES)]
    txt = _strip_accents(" ".join(user_msgs))
    return [c for c in _VITAL_CATEGORIES if any(_strip_accents(r) in txt for r in c["roots"])]

# A photo only helps for VISUAL complaints (skin/rash, eye, wound/swelling, mouth/
# throat, nails, lesions...). For non-visual ones (e.g. chest pain, dizziness) the
# camera adds nothing and just confuses, so the photo option is hidden unless the
# conversation is about something visible.
_VISUAL_ROOTS = [
    # Greek (accent-insensitive)
    "δερμα","εξανθημ","σπυρ","πληγ","τραυμ","κοψιμ","εκδορ","εξογκωμ","πρηξ","πρησμ",
    "πρησιμ","οιδημ","μωλωπ","ελια","σπιλ","μελανωμ","εγκαυμ","καψιμ","δαγκωμ","τσιμπ",
    "κνησμ","φαγουρ","φουσκαλ","φλυκταιν","εκζεμ","ψωριασ","ελκος","εξελκωσ","αφθ",
    "οφθαλμ","ματι","λαιμ","αμυγδαλ","φαρυγγ","γλωσσ","νυχι","ονυχ","ουλη","κονδυλωμ",
    "αλλοιωσ","κηλιδ","δοθιην","αποστημ","σπυρακ","πρηξιμ","οζο",
    # Additional: swelling / lump / bump variants (medical + casual Greek)
    "διογκωσ","καρουμπαλ","ογκο","πεταξ","βγηκ","πρισμ","φουσκωμ","φουσκωσ",
    # English
    "skin","rash","lesion","wound","laceration","abrasion","lump","bump","swelling",
    "swollen","bruise","mole","melanoma","eye","throat","tonsil","tongue","nail",
    "burn","bite","itch","blister","eczema","psoriasis","ulcer","pimple","cyst","wart",
]
def _visual_relevant():
    """Show the photo upload option when EITHER:
    (a) the user explicitly mentions a visual symptom (skin/rash/wound/lump/…), OR
    (b) Asklepios's most recent reply explicitly suggested the photo option.
    Case (b) catches descriptions in casual/regional Greek (e.g. «πετάξει κάτι σαν
    βυζί στον αγκώνα» = an elbow lump) where Claude understood and offered the
    photo, but our keyword list couldn't match the unusual phrasing. We only
    check the LAST assistant message so old differential-diagnosis mentions of
    'εξάνθημα' in unrelated chest-pain workups do NOT trigger false positives."""
    # (a) User explicitly mentions a visual symptom
    user_txt = _strip_accents(" ".join(m["content"] for m in st.session_state.triage_chat
                                       if m["role"] == "user"))
    if any(r in user_txt for r in _VISUAL_ROOTS):
        return True
    # (b) Asklepios's LAST message suggests the photo option
    last_assistant = next((m["content"] for m in reversed(st.session_state.triage_chat)
                           if m["role"] == "assistant"), "")
    a_txt = _strip_accents(last_assistant)
    photo_hints = [
        _strip_accents("αναλυση φωτογραφιας"),
        _strip_accents("ανεβασεις φωτογραφια"),
        _strip_accents("ανεβασετε φωτογραφια"),
        _strip_accents("φωτογραφια απο την επιλογη"),
        "photo analysis",
        "upload a photo",
        "upload photo",
    ]
    return any(p in a_txt for p in photo_hints)

# Quick-select symptom chips, tailored to the person (age + sex from the profile).
# These are common PRESENTING COMPLAINTS per group — not diagnoses — to speed up the
# first message. Age takes precedence over sex (a child gets paediatric chips). The
# user can always type freely or tap "Άλλο/Other".
_CHIP_SETS = {
    "female": {
        "el": (["Πονοκέφαλος/Ημικρανία","Κοιλιακός/πυελικός πόνος","Διαταραχές περιόδου",
                "Ούρα: καύσος/συχνουρία","Κόπωση","Ναυτία","Ζάλη","Πόνος στήθους",
                "Δύσπνοια","Πόνος μέσης","Εξάνθημα/δέρμα","Άλλο"], "συχνά σε γυναίκες"),
        "en": (["Headache/Migraine","Abdominal/pelvic pain","Menstrual changes",
                "Urinary burning/frequency","Fatigue","Nausea","Dizziness","Chest pain",
                "Shortness of breath","Back pain","Rash/skin","Other"], "common in women"),
    },
    "male": {
        "el": (["Πόνος στήθους","Δύσπνοια","Κοιλιακός πόνος","Πόνος μέσης",
                "Ούρα: δυσουρία/συχνουρία","Πονοκέφαλος","Ζάλη","Κόπωση","Βήχας",
                "Πόνος αρθρώσεων","Εξάνθημα/δέρμα","Άλλο"], "συχνά σε άνδρες"),
        "en": (["Chest pain","Shortness of breath","Abdominal pain","Back pain",
                "Urinary problems","Headache","Dizziness","Fatigue","Cough",
                "Joint pain","Rash/skin","Other"], "common in men"),
    },
    "infant": {
        "el": (["Πυρετός","Ανήσυχο/κλάματα","Εμετός/αναγωγές","Διάρροια","Βήχας/συνάχι",
                "Δυσκολία αναπνοής","Εξάνθημα","Δυσκολία σίτισης","Δυσκοιλιότητα",
                "Ίκτερος (κιτρίνισμα)","Άλλο"], "συχνά σε βρέφη"),
        "en": (["Fever","Irritable/crying","Vomiting/spit-up","Diarrhoea","Cough/congestion",
                "Breathing difficulty","Rash","Feeding difficulty","Constipation",
                "Jaundice","Other"], "common in infants"),
    },
    "child": {
        "el": (["Πυρετός","Βήχας","Πονόλαιμος","Πόνος αυτιού","Κοιλιακός πόνος","Εμετός",
                "Διάρροια","Εξάνθημα","Πονοκέφαλος","Δυσκολία αναπνοής","Άλλο"],
               "συχνά σε παιδιά/εφήβους"),
        "en": (["Fever","Cough","Sore throat","Ear pain","Abdominal pain","Vomiting",
                "Diarrhoea","Rash","Headache","Breathing difficulty","Other"],
               "common in children/teens"),
    },
    "adult": {
        "el": (["Πονοκέφαλος","Πυρετός","Βήχας","Δύσπνοια","Ναυτία","Πόνος στήθους",
                "Κοιλιακός πόνος","Ζάλη","Κόπωση","Πόνος πλάτης","Διάρροια","Άλλο"], ""),
        "en": (["Headache","Fever","Cough","Shortness of breath","Nausea","Chest pain",
                "Abdominal pain","Dizziness","Fatigue","Back pain","Diarrhoea","Other"], ""),
    },
}
def _symptom_chips(profile, lang):
    """Return (chips, group_label) for the person's age/sex group."""
    age = profile.get("age", 0) or 0
    sex = profile.get("sex", "")
    if age <= 16:
        g = "infant" if age < 2 else "child"
    elif sex in ("Γυναίκα", "Female"):
        g = "female"
    elif sex in ("Άνδρας", "Male"):
        g = "male"
    else:
        g = "adult"
    return _CHIP_SETS[g]["el" if lang == "el" else "en"]

def generate_html_report(profile, vitals, report_text, pubmed_refs, lang="el", recs=None, photo_findings=None, lab_findings=None):
    import re as _re, html as _html
    name=_html.escape(str(profile.get("name","—"))); age=str(profile.get("age","—"))
    sex=_html.escape(str(profile.get("sex",""))); hx=_html.escape(str(profile.get("history","") or "—"))
    allg=_html.escape(str(profile.get("allergies","") or "—")); meds=_html.escape(str(profile.get("meds_raw","") or "—"))
    ts=datetime.now().strftime("%d %B %Y  %H:%M")
    VLABELS={"hr":("Καρδιακός Ρυθμός","bpm"),"bp_sys":("ΑΠ Συστολική","mmHg"),"bp_dia":("ΑΠ Διαστολική","mmHg"),"br":("Αναπνευστικός Ρυθμός","/min"),"spo2":("SpO2","%"),"temp":("Θερμοκρασία","°C"),"weight":("Βάρος","kg"),"height":("Ύψος","cm"),"bmi":("ΔΜΣ","kg/m²"),"hrv":("HRV","ms"),"stress":("Δείκτης Στρες","/100")}
    vitals_rows="".join(f"<tr><td>{VLABELS.get(k,(k,''))[0]}</td><td><strong>{_html.escape(str(val))}</strong> {VLABELS.get(k,(k,''))[1]}</td></tr>" for k,val in (vitals or {}).items())
    vitals_sec=f"<h2>Ζωτικές Ενδείξεις</h2><table class='vitals'><thead><tr><th>Παράμετρος</th><th>Τιμή</th></tr></thead><tbody>{vitals_rows}</tbody></table>" if vitals_rows else ""
    def md2h(text):
        out=[]
        for line in text.splitlines():
            l=line.strip()
            if not l: out.append("<br>"); continue
            if l.startswith("## ") or l.startswith("# "): out.append(f"<h2>{_html.escape(l.lstrip('#').strip())}</h2>")
            elif l.startswith(("- ","* ","• ")): out.append(f"<li>{_re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',_html.escape(l[2:]))}</li>")
            else: out.append(f"<p>{_re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',_html.escape(l))}</p>")
        r="\n".join(out)
        return _re.sub(r"(<li>.*?</li>\n)+",lambda m:"<ul>"+m.group(0)+"</ul>",r,flags=_re.DOTALL)
    refs_html=""
    if pubmed_refs:
        refs_html="<h2>Βιβλιογραφία</h2><ol>"+"".join(f'<li>{_html.escape(a.get("title","—"))} — {_html.escape(a.get("authors",""))}. <em>{_html.escape(a.get("journal",""))}</em>, {_html.escape(a.get("date",""))}. <a href="{_html.escape(a.get("url",""))}">{_html.escape(a.get("url",""))}</a></li>' for a in pubmed_refs)+"</ol>"
    # PNOE-style Recommendations section (Exercise / Nutrition / Lifestyle)
    recs_html = ""
    if recs and any(recs.get(k) for k in ("exercise","nutrition","lifestyle")):
        _ex = _html.escape(recs.get("exercise","—"))
        _nu = _html.escape(recs.get("nutrition","—"))
        _li = _html.escape(recs.get("lifestyle","—"))
        _t = ("Εξατομικευμένες Συστάσεις", "Φυσική Δραστηριότητα", "Διατροφή", "Τρόπος Ζωής",
              "Οδηγίες & μετα-αναλύσεις") if lang=="el" \
             else ("Personalised Recommendations", "Exercise", "Nutrition", "Lifestyle",
                   "Guidelines & meta-analyses")
        def _refs_box(pillar):
            items = (recs.get("_refs", {}) or {}).get(pillar) or []
            if not items: return ""
            lis = "".join(
                f'<li><a href="{_html.escape(r.get("url",""))}" target="_blank" '
                f'style="color:#1E40AF;text-decoration:none">'
                f'{_html.escape((r.get("title","—") or "")[:120])}</a>'
                f'<span style="color:#9CA3AF"> · {_html.escape(r.get("journal","") or "")}'
                f'{(" " + _html.escape((r.get("date","") or "")[:4])) if r.get("date") else ""}</span></li>'
                for r in items
            )
            return (f'<div class="recs-refs"><div class="recs-refs-lbl">📚 {_t[4]}</div>'
                    f'<ul>{lis}</ul></div>')
        recs_html = (
            f'<h2>📍 {_t[0]}</h2>'
            '<div class="recs-grid">'
            f'<div class="recs-box exercise"><div class="recs-lbl">🏃 {_t[1]}</div><div>{_ex}</div>{_refs_box("exercise")}</div>'
            f'<div class="recs-box nutrition"><div class="recs-lbl">🥗 {_t[2]}</div><div>{_nu}</div>{_refs_box("nutrition")}</div>'
            f'<div class="recs-box lifestyle"><div class="recs-lbl">🌿 {_t[3]}</div><div>{_li}</div>{_refs_box("lifestyle")}</div>'
            '</div>'
        )
    # Photo findings — if visual analyses exist, add a section with each one.
    photo_html = ""
    if photo_findings and isinstance(photo_findings, list):
        _pf_title = "📷 Ευρήματα από Φωτογραφίες" if lang=="el" else "📷 Photo Findings"
        _pf_items = ""
        for i, pf in enumerate(photo_findings, 1):
            _lbl = _html.escape(pf.get("scan_label","—"))
            _an = _re.sub(r"\s+", " ", (pf.get("analysis","") or "").strip())
            _an = _html.escape(_an)
            _an = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _an)
            _pf_items += (
                f'<div class="pf-row"><div class="pf-row-head">'
                f'<span class="pf-row-num">{i}</span><span class="pf-row-lbl">{_lbl}</span>'
                f'</div><div class="pf-row-body">{_an}</div></div>'
            )
        photo_html = f'<h2>{_pf_title}</h2><div class="pf-list">{_pf_items}</div>'
    # Lab findings — same structure as photo, green accent for lab data.
    lab_html = ""
    if lab_findings and isinstance(lab_findings, list):
        _lf_title = "🧪 Ευρήματα Εργαστηριακών Εξετάσεων" if lang=="el" else "🧪 Lab Findings"
        _lf_items = ""
        for i, lf in enumerate(lab_findings, 1):
            _lbl = _html.escape(lf.get("file_name","—"))
            _an = _re.sub(r"\s+", " ", (lf.get("analysis","") or "").strip())
            _an = _html.escape(_an)
            _an = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _an)
            _lf_items += (
                f'<div class="lf-row"><div class="lf-row-head">'
                f'<span class="lf-row-num">{i}</span><span class="lf-row-lbl">📄 {_lbl}</span>'
                f'</div><div class="lf-row-body">{_an}</div></div>'
            )
        lab_html = f'<h2>{_lf_title}</h2><div class="lf-list">{_lf_items}</div>'
    html_out=f"""<!DOCTYPE html><html lang="{lang}"><head><meta charset="UTF-8"><title>Asklepios Report — {name}</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Inter',sans-serif;font-size:13px;color:#1A1A2E;max-width:820px;margin:0 auto;padding:32px 40px}}
.hdr{{display:flex;justify-content:space-between;align-items:center;border-bottom:3px solid #2D3FE7;padding-bottom:14px;margin-bottom:20px}}
.hdr-logo{{font-size:22px;font-weight:800;color:#2D3FE7}}.hdr-date{{font-size:11px;color:#6B7280;text-align:right}}
.patient{{background:linear-gradient(135deg,#2D3FE7,#7B2FE0);color:white;border-radius:12px;padding:18px 22px;margin-bottom:20px}}
.patient-name{{font-size:20px;font-weight:700;margin-bottom:4px}}.patient-meta{{font-size:12px;opacity:.8}}.patient-detail{{font-size:11px;opacity:.75;margin-top:10px;line-height:1.8}}
h2{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7B2FE0;border-bottom:1px solid #E0E5FF;padding-bottom:5px;margin:20px 0 10px}}
p{{margin:4px 0;line-height:1.65}}ul{{margin:6px 0 6px 18px}}li{{margin:3px 0;line-height:1.6}}
table.vitals{{width:100%;border-collapse:collapse;margin:10px 0;font-size:12px}}
table.vitals thead tr{{background:#2D3FE7;color:white}}table.vitals th,table.vitals td{{padding:7px 12px;text-align:left;border:1px solid #E0E5FF}}
table.vitals tbody tr:nth-child(even){{background:#F8FAFF}}
.emergency{{background:#DC2626;color:white;border-radius:8px;padding:12px 16px;font-weight:700;margin:16px 0}}
.disclaimer{{background:#FFFBEB;border:1px solid #FCD34D;border-radius:8px;padding:10px 14px;font-size:11px;color:#92400E;margin:12px 0}}
.recs-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:10px 0 16px}}
.recs-box{{border:1px solid;border-radius:10px;padding:12px 14px;font-size:12px;line-height:1.55}}
.recs-box.exercise{{background:#EFF6FF;border-color:#BFDBFE}}
.recs-box.nutrition{{background:#ECFDF5;border-color:#A7F3D0}}
.recs-box.lifestyle{{background:#FEF3F2;border-color:#FECDD3}}
.recs-lbl{{font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#1F2937;margin-bottom:6px}}
.recs-refs{{margin-top:8px;padding-top:6px;border-top:1px dashed rgba(0,0,0,0.10)}}
.recs-refs-lbl{{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6B7280;margin-bottom:4px}}
.recs-refs ul{{list-style:none;padding:0;margin:0}}.recs-refs li{{font-size:10.5px;line-height:1.4;margin-bottom:3px}}
.pf-list{{margin:8px 0 16px}}.pf-row{{padding:10px 0;border-bottom:1px solid #F3F4F6}}.pf-row:last-child{{border-bottom:none}}
.pf-row-head{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.pf-row-num{{background:#DBEAFE;color:#1E40AF;font-size:10px;font-weight:700;width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center}}
.pf-row-lbl{{font-size:12px;font-weight:700;color:#111827}}.pf-row-body{{font-size:11.5px;color:#374151;line-height:1.55}}
.lf-list{{margin:8px 0 16px}}.lf-row{{padding:10px 0;border-bottom:1px solid #F3F4F6}}.lf-row:last-child{{border-bottom:none}}
.lf-row-head{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.lf-row-num{{background:#D1FAE5;color:#065F46;font-size:10px;font-weight:700;width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center}}
.lf-row-lbl{{font-size:12px;font-weight:700;color:#111827}}.lf-row-body{{font-size:11.5px;color:#374151;line-height:1.55}}
@media print{{.recs-box{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}.recs-grid{{grid-template-columns:1fr 1fr 1fr !important}}}}
.hint{{text-align:center;margin:24px 0 0;font-size:12px;color:#94A3B8;border-top:1px dashed #E0E5FF;padding-top:14px}}
@media print{{body{{padding:16px}}.patient,.emergency{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}@page{{margin:15mm}}}}</style></head><body>
<div class="hdr"><div class="hdr-logo">🩺 Asklepios AI Nurse</div><div class="hdr-date">Κλινική Εκτίμηση<br>{ts}</div></div>
<div class="patient"><div class="patient-name">{name}</div><div class="patient-meta">{age} ετών · {sex}</div>
<div class="patient-detail"><strong>Ιστορικό:</strong> {hx}<br><strong>Αλλεργίες:</strong> {allg}<br><strong>Φάρμακα:</strong> {meds}</div></div>
{vitals_sec}<h2>Κλινική Αξιολόγηση</h2>{md2h(report_text or "")}{photo_html}{lab_html}{recs_html}{refs_html}
<div class="emergency">🚨 ΣΕ ΕΠΕΙΓΟΥΣΑ ΑΝΑΓΚΗ: ΚΑΛΕΣΤΕ 166 (ΕΚΑΒ) ή 112</div>
<div class="disclaimer">⚠️ AI-generated. Δεν αποτελεί ιατρική διάγνωση. Απαιτείται επίσκεψη σε επαγγελματία υγείας.</div>
<div class="hint">💡 Ctrl+P → Save as PDF</div></body></html>"""
    return html_out.encode("utf-8")

def _render_symptom_tracker(lang):
    """Browser-only symptom log. All data in localStorage — nothing on servers.
    User can add dated entries (symptom + severity + notes), view history,
    and export as text. Built as a self-contained HTML/JS component so it works
    regardless of login state. Privacy: we never see this data.
    """
    # Symptom tracker uses st.iframe (HTML string mode)
    _title = "📅 Ημερολόγιο Συμπτωμάτων" if lang=="el" else "📅 Symptom Log"
    _privacy = ("Αποθηκεύεται μόνο στον browser σου — δεν αποστέλλεται πουθενά."
                if lang=="el" else
                "Stored only in your browser — never sent anywhere.")
    with st.expander(f"{_title} — {_privacy}", expanded=False):
        if lang == "el":
            tx = {
                "add_title":   "Προσθήκη σημερινού συμπτώματος",
                "symptom_ph":  "π.χ. πονοκέφαλος, βήχας, κοιλιακός πόνος",
                "sev_lbl":     "Βαρύτητα (1–10)",
                "notes_ph":    "Επιπλέον παρατηρήσεις (προαιρετικό)",
                "add_btn":     "➕ Καταχώρηση",
                "history":     "Ιστορικό",
                "no_entries":  "Κανένα σύμπτωμα ακόμη.",
                "clear_btn":   "🗑️ Διαγραφή όλων",
                "export_btn":  "📋 Αντιγραφή ιστορικού",
                "exported":    "✅ Αντιγράφηκε!",
                "sev_prefix":  "Βαρύτητα",
                "confirm_clear":"Διαγραφή ΟΛΩΝ των συμπτωμάτων; Δεν αναιρείται.",
            }
        else:
            tx = {
                "add_title":   "Log today's symptom",
                "symptom_ph":  "e.g. headache, cough, stomach pain",
                "sev_lbl":     "Severity (1–10)",
                "notes_ph":    "Additional notes (optional)",
                "add_btn":     "➕ Add entry",
                "history":     "History",
                "no_entries":  "No symptoms logged yet.",
                "clear_btn":   "🗑️ Clear all",
                "export_btn":  "📋 Copy log",
                "exported":    "✅ Copied!",
                "sev_prefix":  "Severity",
                "confirm_clear":"Delete ALL symptom entries? Cannot be undone.",
            }
        st.iframe(f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,sans-serif}}
body{{background:transparent;padding:0;font-size:14px;color:#1F2937}}
.st-card{{background:white;border:1px solid #E5E7EB;border-radius:12px;padding:16px 18px;margin-bottom:12px}}
.st-card h3{{font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6B7280;margin-bottom:12px}}
.st-row{{display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap}}
input[type=text],textarea{{width:100%;border:1px solid #D1D5DB;border-radius:8px;padding:8px 10px;font-size:13px;color:#1F2937;background:white}}
input[type=text]:focus,textarea:focus{{outline:none;border-color:#2D3FE7;box-shadow:0 0 0 2px rgba(45,63,231,.10)}}
textarea{{resize:vertical;min-height:48px}}
input[type=range]{{width:100%;accent-color:#2D3FE7}}
.sev-row{{display:flex;align-items:center;gap:8px}}
.sev-label{{font-size:11px;color:#6B7280;white-space:nowrap}}
.sev-val{{font-size:18px;font-weight:700;color:#2D3FE7;min-width:24px;text-align:right}}
.btn{{padding:9px 16px;border-radius:8px;border:none;cursor:pointer;font-weight:600;font-size:13px;transition:all .15s}}
.btn-primary{{background:#2D3FE7;color:white}}.btn-primary:hover{{background:#1E30CC}}
.btn-ghost{{background:#F3F4F6;color:#374151;border:1px solid #E5E7EB}}.btn-ghost:hover{{background:#E5E7EB}}
.btn-danger{{background:#FEF2F2;color:#DC2626;border:1px solid #FCA5A5}}.btn-danger:hover{{background:#FEE2E2}}
.entry{{border-bottom:1px solid #F3F4F6;padding:10px 0;display:flex;justify-content:space-between;align-items:flex-start;gap:8px}}
.entry:last-child{{border-bottom:none}}
.entry-main{{flex:1}}
.entry-date{{font-size:11px;color:#9CA3AF;margin-bottom:2px}}
.entry-symptom{{font-size:14px;font-weight:600;color:#111827}}
.entry-sev{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;margin-left:6px}}
.entry-notes{{font-size:12px;color:#6B7280;margin-top:3px}}
.del-btn{{background:none;border:none;cursor:pointer;color:#9CA3AF;font-size:16px;padding:2px 4px;flex-shrink:0}}.del-btn:hover{{color:#DC2626}}
.empty{{text-align:center;padding:24px;color:#9CA3AF;font-size:13px}}
.tools{{display:flex;gap:8px;margin-top:8px}}
</style></head><body>

<div class="st-card">
  <h3>{tx['add_title']}</h3>
  <input type="text" id="symp" placeholder="{tx['symptom_ph']}" />
  <div style="margin-top:10px">
    <div class="sev-row">
      <span class="sev-label">{tx['sev_lbl']}</span>
      <input type="range" id="sev" min="1" max="10" value="5"
             oninput="document.getElementById('sev-val').textContent=this.value" />
      <span class="sev-val" id="sev-val">5</span>
    </div>
  </div>
  <textarea id="notes" placeholder="{tx['notes_ph']}" style="margin-top:10px"></textarea>
  <div style="margin-top:10px">
    <button class="btn btn-primary" onclick="addEntry()">{tx['add_btn']}</button>
  </div>
</div>

<div class="st-card">
  <h3>{tx['history']}</h3>
  <div id="list"></div>
  <div class="tools" id="tools" style="display:none">
    <button class="btn btn-ghost" onclick="exportLog()">{tx['export_btn']}</button>
    <button class="btn btn-danger" onclick="clearAll()">{tx['clear_btn']}</button>
  </div>
</div>

<script>
var STORE_KEY = "asklepios_symptoms_v1";

function load() {{
  try {{ return JSON.parse(localStorage.getItem(STORE_KEY) || "[]"); }}
  catch(e) {{ return []; }}
}}
function save(entries) {{
  localStorage.setItem(STORE_KEY, JSON.stringify(entries));
}}

function sevColor(s) {{
  if(s<=3) return "#ECFDF5;color:#065F46";
  if(s<=6) return "#FFFBEB;color:#92400E";
  return "#FEF2F2;color:#991B1B";
}}

function renderList() {{
  var entries = load();
  var el = document.getElementById("list");
  var tools = document.getElementById("tools");
  if(!entries.length) {{
    el.innerHTML = '<div class="empty">{tx['no_entries']}</div>';
    tools.style.display = "none";
    return;
  }}
  tools.style.display = "flex";
  // newest first
  var html = "";
  for(var i=entries.length-1; i>=0; i--) {{
    var e = entries[i];
    var sc = sevColor(e.sev);
    var sc_parts = sc.split(";color:");
    var bg = sc_parts[0];
    var fg = sc_parts[1] || "#111";
    html += '<div class="entry">';
    html += '<div class="entry-main">';
    html += '<div class="entry-date">'+e.date+'</div>';
    html += '<div class="entry-symptom">'+escape_html(e.symptom);
    html += ' <span class="entry-sev" style="background:'+bg+';color:'+fg+'">'+e.sev+'/10</span></div>';
    if(e.notes) html += '<div class="entry-notes">'+escape_html(e.notes)+'</div>';
    html += '</div>';
    html += '<button class="del-btn" onclick="deleteEntry('+i+')" title="Delete">✕</button>';
    html += '</div>';
  }}
  el.innerHTML = html;
}}

function escape_html(s) {{
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

function addEntry() {{
  var symp = document.getElementById("symp").value.trim();
  if(!symp) {{ document.getElementById("symp").focus(); return; }}
  var sev  = parseInt(document.getElementById("sev").value);
  var notes= document.getElementById("notes").value.trim();
  var now  = new Date();
  var date = now.toLocaleDateString("{("el-GR" if lang=="el" else "en-GB")}",
    {{day:"2-digit",month:"short",year:"numeric",hour:"2-digit",minute:"2-digit"}});
  var entries = load();
  entries.push({{date:date, symptom:symp, sev:sev, notes:notes}});
  save(entries);
  document.getElementById("symp").value="";
  document.getElementById("notes").value="";
  document.getElementById("sev").value=5;
  document.getElementById("sev-val").textContent="5";
  renderList();
}}

function deleteEntry(idx) {{
  var entries = load();
  entries.splice(idx,1);
  save(entries);
  renderList();
}}

function clearAll() {{
  if(confirm("{tx['confirm_clear']}")) {{
    localStorage.removeItem(STORE_KEY);
    renderList();
  }}
}}

function exportLog() {{
  var entries = load();
  if(!entries.length) return;
  var txt = entries.map(function(e){{
    var line = e.date+" | "+e.symptom+" | {tx['sev_prefix']}: "+e.sev+"/10";
    if(e.notes) line += " | "+e.notes;
    return line;
  }}).join("\\n");
  navigator.clipboard.writeText(txt).then(function(){{
    var b = document.querySelector(".btn-ghost");
    var orig = b.textContent;
    b.textContent="{tx['exported']}";
    setTimeout(function(){{b.textContent=orig;}},2000);
  }});
}}

renderList();
</script>
</body></html>""", height=520)


def render_home():
    lang = st.session_state.lang
    p = st.session_state.profile
    name = p.get("name", "")
    today_str = datetime.now().strftime("%d.%m.%Y")
    # Editorial banner — now visible on home too (no CTA, Start button is right below).
    render_ad_banner(lang)
    # Doc-template "assessment sheet" card — Notion-clean white surface with
    # doctor's-report aesthetic: title block, patient fields, checklist, blue emergency box.
    if lang == "el":
        txt = {
            "org": "ASKLEPIOS · AI ΝΟΣΗΛΕΥΤΗΣ",
            "title": "Φύλλο Εκτίμησης Υγείας",
            "patient_lbl": "Όνομα:",
            "date_lbl": "Ημερομηνία:",
            "includes": "Τι περιλαμβάνει η εκτίμηση:",
            "check1": "Κλινική εκτίμηση συμπτωμάτων",
            "check2": "Αναφορές από PubMed (επιστημονική βιβλιογραφία)",
            "check3": "Δεύτερη γνώμη GPT-4o",
            "check4": "Έλεγχος αλληλεπιδράσεων φαρμάκων",
            "check5": "PDF αναφορά για τον γιατρό σου",
            "start": "📋 Ξεκίνα την εκτίμηση →",
            "emergency_lbl": "Επείγον:",
            "emergency_text": "Για πόνο στο στήθος, δυσκολία αναπνοής, σοβαρή αιμορραγία, απώλεια συνείδησης ή συμπτώματα εγκεφαλικού, καλέστε αμέσως 166 (ΕΚΑΒ) ή 112.",
            "blank": "_____________________",
        }
    else:
        txt = {
            "org": "ASKLEPIOS · AI NURSE",
            "title": "Health Assessment Sheet",
            "patient_lbl": "Name:",
            "date_lbl": "Date:",
            "includes": "What this assessment includes:",
            "check1": "Clinical symptom evaluation",
            "check2": "PubMed references (scientific literature)",
            "check3": "GPT-4o second opinion",
            "check4": "Drug interaction check",
            "check5": "PDF report for your doctor",
            "start": "📋 Start assessment →",
            "emergency_lbl": "Emergency:",
            "emergency_text": "For chest pain, difficulty breathing, severe bleeding, loss of consciousness, or stroke symptoms, call 166 (EKAB) or 112 immediately.",
            "blank": "_____________________",
        }
    name_display = name if name else txt["blank"]
    st.markdown(f"""
<style>
.doc-card {{
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 16px;
  padding: 36px 36px 32px;
  margin: 8px 0 24px;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.doc-head {{
  display: flex; align-items: center; gap: 18px;
  border-bottom: 1px solid #F3F4F6;
  padding-bottom: 22px; margin-bottom: 26px;
}}
.doc-logo {{
  width: 60px; height: 60px; border-radius: 50%;
  background: #DBEAFE;
  display: flex; align-items: center; justify-content: center;
  font-size: 26px; flex-shrink: 0;
}}
.doc-head-text .org {{
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.14em;
  color: #6B7280; text-transform: uppercase; margin-bottom: 4px;
}}
.doc-head-text .title {{
  font-size: 26px; font-weight: 700; color: #111827;
  letter-spacing: -0.02em; line-height: 1.15;
}}
.doc-fields {{
  display: flex; gap: 36px; margin-bottom: 26px; flex-wrap: wrap;
}}
.doc-field {{ flex: 1; min-width: 200px; }}
.doc-field label {{
  display: block;
  font-size: 12.5px; font-weight: 600; color: #374151;
  margin-bottom: 6px;
}}
.doc-field .underline {{
  font-size: 15px; color: #111827; font-weight: 500;
  border-bottom: 1.5px solid #1F2937;
  padding-bottom: 5px;
}}
.doc-section {{
  font-size: 14px; font-weight: 700; color: #111827;
  margin: 4px 0 14px;
}}
.doc-checklist {{
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 11px 28px; margin-bottom: 26px;
}}
.doc-check-item {{
  display: flex; align-items: flex-start; gap: 11px;
  font-size: 13.5px; color: #374151; line-height: 1.5;
}}
.doc-check-box {{
  width: 18px; height: 18px; border: 1.5px solid #2563EB;
  border-radius: 4px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  color: #2563EB; font-size: 11px; font-weight: 800;
  background: #EFF6FF; margin-top: 1px;
}}
.doc-emergency {{
  background: #DBEAFE;
  border: 1px solid #93C5FD;
  border-radius: 12px;
  padding: 16px 18px;
  font-size: 13px; color: #1E3A8A;
  line-height: 1.55;
}}
.doc-emergency strong {{
  color: #1E40AF; font-weight: 700;
}}
@media (max-width: 640px) {{
  .doc-card {{ padding: 24px 20px 22px; }}
  .doc-head {{ flex-direction: column; align-items: flex-start; gap: 12px; padding-bottom: 18px; margin-bottom: 20px; }}
  .doc-logo {{ width: 50px; height: 50px; font-size: 22px; }}
  .doc-head-text .title {{ font-size: 21px; }}
  .doc-checklist {{ grid-template-columns: 1fr; gap: 10px; margin-bottom: 22px; }}
  .doc-fields {{ gap: 18px; margin-bottom: 22px; }}
}}
</style>
<div class="doc-card">
  <div class="doc-head">
    <div class="doc-logo">📋</div>
    <div class="doc-head-text">
      <div class="org">{txt['org']}</div>
      <div class="title">{txt['title']}</div>
    </div>
  </div>
  <div class="doc-fields">
    <div class="doc-field">
      <label>{txt['patient_lbl']}</label>
      <div class="underline">{name_display}</div>
    </div>
    <div class="doc-field">
      <label>{txt['date_lbl']}</label>
      <div class="underline">{today_str}</div>
    </div>
  </div>
  <div class="doc-section">{txt['includes']}</div>
  <div class="doc-checklist">
    <div class="doc-check-item"><span class="doc-check-box">✓</span><span>{txt['check1']}</span></div>
    <div class="doc-check-item"><span class="doc-check-box">✓</span><span>{txt['check2']}</span></div>
    <div class="doc-check-item"><span class="doc-check-box">✓</span><span>{txt['check3']}</span></div>
    <div class="doc-check-item"><span class="doc-check-box">✓</span><span>{txt['check4']}</span></div>
    <div class="doc-check-item"><span class="doc-check-box">✓</span><span>{txt['check5']}</span></div>
  </div>
  <div class="doc-emergency">
    <strong>🚨 {txt['emergency_lbl']}</strong> {txt['emergency_text']}
  </div>
</div>
""", unsafe_allow_html=True)
    # Start button — primary CTA, sits right below the doc card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(txt["start"], type="primary", use_container_width=True, key="home_start"):
            st.session_state.screen = "intake"; st.rerun()
    # ── Symptom Tracker ────────────────────────────────────────────────────────
    # localStorage-only: nothing goes to our servers. The tracker lives entirely
    # in the browser. Privacy note shown inline.
    _render_symptom_tracker(lang)


    render_stepper("intake")
    lang = st.session_state.lang
    render_doc_header(
        "Πες μας λίγα για σένα", "Tell us about yourself",
        icon="👤",
        sub_el="Όνομα, ηλικία, ιατρικό ιστορικό",
        sub_en="Name, age, medical history",
    )
    # ── Caregiver toggle ───────────────────────────────────────────────────
    # First question: is this assessment for the user themselves or someone
    # they care for (γιαγιά, παιδί, κλπ). Affects copy + Claude system prompt.
    _caregiver_q = ("Για ποιον είναι αυτή η αξιολόγηση;" if lang=="el"
                    else "Who is this assessment for?")
    _opt_self = "Για μένα" if lang=="el" else "For me"
    _opt_other = "Για άλλο άτομο που φροντίζω" if lang=="el" else "For someone I care for"
    _current = st.session_state.profile.get("for_whom", "self")
    _choice = st.radio(
        _caregiver_q,
        [_opt_self, _opt_other],
        index=(0 if _current == "self" else 1),
        horizontal=True,
        key="intake_for_whom",
    )
    is_caregiver = (_choice == _opt_other)
    if is_caregiver:
        st.info("💡 " + ("Συμπλήρωσε τα στοιχεία του ατόμου που φροντίζεις (όχι τα δικά σου)."
                         if lang=="el" else
                         "Fill in details of the person you care for (not your own)."))
        _name_lbl = "Όνομα του ασθενούς" if lang=="el" else "Patient's name"
        _name_ph  = "π.χ. Γιαγιά Ελένη" if lang=="el" else "e.g. Grandma Helen"
    else:
        _name_lbl = t("name")
        _name_ph  = "Χριστόφορος"
    c1,c2,c3=st.columns([2,1,1])
    with c1: name=st.text_input(_name_lbl,value=st.session_state.profile.get("name",""),placeholder=_name_ph)
    with c2: age=st.number_input(t("age"),min_value=0,max_value=120,value=st.session_state.profile.get("age",40))
    with c3: sex=st.selectbox(t("sex"),[t("male"),t("female"),t("other")])
    # ── Pregnancy checkbox ──────────────────────────────────────────────────
    # Only shown for female + age 12-55 (reproductive age). Affects drug
    # contraindications + Claude system prompt + recs.
    pregnancy = False
    _is_female = sex in ("Γυναίκα", "Female")
    if _is_female and 12 <= age <= 55:
        _preg_lbl = "🤰 Είναι έγκυος;" if lang=="el" else "🤰 Is she pregnant?"
        pregnancy = st.checkbox(_preg_lbl, value=st.session_state.profile.get("pregnancy", False))
        if pregnancy:
            st.info("💡 " + ("Σημειώνεται για έλεγχο αντενδείξεων φαρμάκων και συστάσεων."
                             if lang=="el" else
                             "Noted — used to flag drug contraindications and adjusted recommendations."))
    history=st.text_area(t("history"),value=st.session_state.profile.get("history",""),height=90,placeholder="Π.χ. Υπέρταση, Τ2 Διαβήτης")
    allergies=st.text_input(t("allergies"),value=st.session_state.profile.get("allergies",""),placeholder="Π.χ. Πενικιλλίνη")
    st.markdown("**"+t("meds")+"**")
    if not st.session_state.med_inputs:
        prev=st.session_state.profile.get("meds_raw","")
        st.session_state.med_inputs=[m.strip() for m in prev.split(",") if m.strip()] or [""]
    for mi,med_val in enumerate(st.session_state.med_inputs):
        mc1,mc2=st.columns([5,1])
        with mc1: st.session_state.med_inputs[mi]=st.text_input(f"Φάρμακο {mi+1}",value=med_val,key=f"med_field_{mi}",label_visibility="collapsed",placeholder="Π.χ. Metformin 500mg" if mi==0 else "")
        with mc2:
            if st.button("✕",key=f"del_med_{mi}"): st.session_state.med_inputs.pop(mi); st.rerun()
    if st.button("＋ "+("Προσθήκη" if st.session_state.lang=="el" else "Add med")): st.session_state.med_inputs.append(""); st.rerun()
    meds_raw=", ".join(m for m in st.session_state.med_inputs if m.strip())
    col_b,col_n=st.columns([1,3])
    with col_b:
        if st.button(t("back")): st.session_state.screen="home"; st.rerun()
    with col_n:
        if st.button(t("next"),type="primary",use_container_width=True):
            if name:
                st.session_state.profile={
                    "name":name, "age":age, "sex":sex,
                    "history":history, "allergies":allergies, "meds_raw":meds_raw,
                    "for_whom": "other" if is_caregiver else "self",
                    "pregnancy": bool(pregnancy),
                }
                st.session_state.medications=[{"name":m.strip(),"freq":"","notes":""} for m in meds_raw.split(",") if m.strip()] if meds_raw else []
                if st.session_state.get("_from_facescan") and st.session_state.vitals:
                    st.session_state.screen="triage"
                else:
                    st.session_state.screen="vitals"
                st.rerun()
            else:
                st.warning("Παρακαλώ εισάγετε το όνομά σας." if st.session_state.lang=="el" else "Please enter your name.")

def render_vitals():
    render_stepper("vitals")
    p=st.session_state.profile
    lang=st.session_state.lang
    nm = p.get("name","")
    render_doc_header(
        "Πώς είναι τα ζωτικά σου;", "How are your vitals?",
        icon="❤️",
        sub_el=(f"για τον/την {nm}" if nm else "Χειροκίνητα, με συσκευή ή σάρωση προσώπου"),
        sub_en=(f"for {nm}" if nm else "Manual, device, or face scan"),
    )

    # ── Tab layout: Manual (default) | Device Import | Face Scan (experimental) ──
    tab_manual, tab_device, tab_scan = st.tabs([
        "✏️ " + ("Χειροκίνητη Εισαγωγή" if lang=="el" else "Manual Entry"),
        "⌚ " + ("Εισαγωγή από Συσκευή" if lang=="el" else "Import from Device"),
        "📷 " + ("Σάρωση (πειραματικό)" if lang=="el" else "Face Scan (experimental)"),
    ])

    with tab_scan:
        st.caption(("⚠️ Πειραματικό. Η σάρωση με κάμερα δίνει μόνο ενδεικτικό καρδιακό ρυθμό — για αξιόπιστες τιμές χρησιμοποίησε «Χειροκίνητη Εισαγωγή» ή «Συσκευή»."
                    if lang=="el" else
                    "⚠️ Experimental. The camera scan gives only an indicative heart rate — for reliable values use 'Manual Entry' or 'Device'."))
        facescan_url=_secret("FACESCAN_URL","https://asklepiosnurse.netlify.app")
        kira_url=_secret("ASKLEPIOS_URL","https://asklepiosainurse.up.railway.app")
        scan_link=f"{facescan_url}?kira_url={urllib.parse.quote(kira_url)}"
        _save_session_for_external_nav()
        st.markdown(f'''<div style="background:linear-gradient(135deg,#2D3FE7,#7B2FE0);border-radius:16px;padding:28px;text-align:center;color:white;margin:8px 0">
            <div style="font-size:40px;margin-bottom:8px">📷</div>
            <div style="font-size:18px;font-weight:700;margin-bottom:8px">{"Σάρωση Προσώπου rPPG" if lang=="el" else "rPPG Face Scan"}</div>
            <div style="font-size:13px;opacity:0.8;margin-bottom:16px">{"Μέτρηση καρδιακού ρυθμού & αναπνοής σε 30 δευτερόλεπτα μέσω κάμερας" if lang=="el" else "Measure heart rate & breathing in 60 seconds via camera"}</div>
            <a href="{scan_link}" target="_blank" style="background:white;color:#2D3FE7;padding:12px 28px;border-radius:8px;font-weight:700;text-decoration:none;font-size:14px">
                {"Έναρξη Σάρωσης →" if lang=="el" else "Start Scan →"}
            </a>
        </div>''', unsafe_allow_html=True)
        st.caption("✅ Μετράει: Καρδιακός ρυθμός, αναπνοή  |  ⚠️ Εκτίμηση: HRV, stress  |  ❌ Δεν μετράει: Αρτηριακή πίεση" if lang=="el"
                   else "✅ Measures: Heart rate, breathing  |  ⚠️ Estimate: HRV, stress  |  ❌ Does not measure: Blood pressure")

    with tab_device:
        st.markdown(f"### {'Εισαγωγή από Smartwatch / Οξύμετρο' if lang=='el' else 'Import from Smartwatch / Oximeter'}")
        st.caption("Apple Watch · Fitbit · Garmin · Polar · Finger oximeter" if lang=="el" else "Apple Watch · Fitbit · Garmin · Polar · Finger oximeter")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**{'Apple Watch / Smartwatch' if lang=='el' else 'Apple Watch / Smartwatch'}**")
            dev_hr   = st.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=None, placeholder="76", key="dev_hr")
            dev_hrv  = st.number_input("HRV (ms)", min_value=0, max_value=300, value=None, placeholder="45", key="dev_hrv")
            dev_spo2 = st.number_input("SpO2 (%)", min_value=0, max_value=100, value=None, placeholder="98", key="dev_spo2")
            dev_br   = st.number_input("Breathing Rate (/min)", min_value=0, max_value=60, value=None, placeholder="15", key="dev_br")
        with d2:
            st.markdown(f"**{'Πιεσόμετρο / Άλλη Συσκευή' if lang=='el' else 'Blood Pressure Monitor / Other'}**")
            dev_bps  = st.number_input("BP Systolic (mmHg)", min_value=0, max_value=300, value=None, placeholder="120", key="dev_bps")
            dev_bpd  = st.number_input("BP Diastolic (mmHg)", min_value=0, max_value=200, value=None, placeholder="80", key="dev_bpd")
            dev_temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=45.0, value=None, placeholder="36.6", key="dev_temp", format="%.1f")
            dev_wt   = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=None, placeholder="75", key="dev_wt", format="%.1f")

        st.markdown(f"**{'Ύψος (για ΔΜΣ)' if lang=='el' else 'Height (for BMI)'}**")
        dev_ht = st.number_input("Height (cm)", min_value=0, max_value=250, value=None, placeholder="175", key="dev_ht")

        if st.button(f"{'Φόρτωση δεδομένων συσκευής' if lang=='el' else 'Load device data'}", type="primary", key="load_device", use_container_width=True):
            vd = {}
            if dev_hr:   vd["hr"]     = int(dev_hr)
            if dev_hrv:  vd["hrv"]    = int(dev_hrv)
            if dev_spo2: vd["spo2"]   = int(dev_spo2)
            if dev_br:   vd["br"]     = int(dev_br)
            if dev_bps:  vd["bp_sys"] = int(dev_bps)
            if dev_bpd:  vd["bp_dia"] = int(dev_bpd)
            if dev_temp: vd["temp"]   = float(dev_temp)
            if dev_wt:   vd["weight"] = float(dev_wt)
            if dev_ht:   vd["height"] = int(dev_ht)
            if vd:
                classify_vitals(vd, age=p.get("age"))
                st.session_state.vitals = vd
                st.session_state["_device_loaded"] = True
            else:
                st.warning("Εισάγετε τουλάχιστον έναν δείκτη." if lang=="el" else "Enter at least one metric.")

        # Show confirmation + vitals + proceed button (no rerun needed)
        if st.session_state.get("_device_loaded") and st.session_state.vitals:
            v_loaded = st.session_state.vitals
            st.success(f"{'✅ Δεδομένα φορτώθηκαν:' if lang=='el' else '✅ Data loaded:'} " +
                       " | ".join(f"{k}={v}" for k,v in v_loaded.items()))
            if st.button(f"{'Συνέχεια στην Εκτίμηση →' if lang=='el' else 'Continue to Assessment →'}",
                         type="primary", key="dev_continue", use_container_width=True):
                st.session_state["_device_loaded"] = False
                with st.spinner("Ανάλυση..."):
                    vtext = "\n".join(f"- {k}: {val}" for k,val in v_loaded.items())
                    pp = p.get
                    st.session_state.vitals_analysis = claude(
                        [{"role":"user","content":f"Patient: {pp('name')}, {pp('age')}yo {pp('sex')}, Hx: {pp('history','none')}, Meds: {pp('meds_raw','none')}\n\nVitals:\n{vtext}\n\nInterpret. Categorise each. Flag urgent findings. Be direct."}],
                        system=kira_system(), max_tokens=1200
                    )
                st.session_state.screen = "triage"
                st.rerun()

        # How-to guide
        with st.expander(f"{'Πώς να εξαγάγετε δεδομένα από τη συσκευή σας' if lang=='el' else 'How to export data from your device'}"):
            st.markdown("""
**Apple Watch / iPhone:**
Health app → Browse → Heart → Heart Rate → export or note the value

**Fitbit:**
Fitbit app → Today → Heart Rate tile

**Garmin / Polar:**
Garmin Connect / Polar Flow app → Dashboard → Heart Rate

**Finger oximeter:**
Read SpO2 and HR directly from the device display

**Blood pressure monitor:**
Use a certified upper-arm cuff device, note systolic/diastolic values
            """)
    with tab_manual:
        v=st.session_state.vitals
        c1,c2,c3=st.columns(3)
        with c1:
            hr=st.number_input(t("hr"),min_value=0,max_value=300,value=int(v.get("hr",0)) or None,placeholder="76")
            spo2=st.number_input(t("spo2"),min_value=0,max_value=100,value=int(v.get("spo2",0)) or None,placeholder="98")
            temp=st.number_input(t("temp"),min_value=0.0,max_value=45.0,value=float(v.get("temp",0.0)) or None,placeholder="36.6",format="%.1f")
        with c2:
            bp_s=st.number_input(t("bp_sys"),min_value=0,max_value=300,value=int(v.get("bp_sys",0)) or None,placeholder="120")
            bp_d=st.number_input(t("bp_dia"),min_value=0,max_value=200,value=int(v.get("bp_dia",0)) or None,placeholder="80")
            br=st.number_input(t("br"),min_value=0,max_value=60,value=int(v.get("br",0)) or None,placeholder="15")
        with c3:
            weight=st.number_input(t("weight"),min_value=0.0,max_value=300.0,value=float(v.get("weight",0.0)) or None,placeholder="75",format="%.1f")
            height=st.number_input(t("height"),min_value=0,max_value=250,value=int(v.get("height",0)) or None,placeholder="175")

        if st.button(t("analyse_vitals"),type="primary",use_container_width=True,key="analyse_manual"):
            vd={}
            if hr: vd["hr"]=hr
            if bp_s: vd["bp_sys"]=bp_s
            if bp_d: vd["bp_dia"]=bp_d
            if br: vd["br"]=br
            if spo2: vd["spo2"]=spo2
            if temp: vd["temp"]=temp
            if weight: vd["weight"]=weight
            if height: vd["height"]=height
            for extra in ["hrv","stress","cardio"]:
                if extra in st.session_state.vitals: vd[extra]=st.session_state.vitals[extra]
            st.session_state.vitals=vd; classify_vitals(vd, age=p.get("age"))
            if vd:
                with st.spinner("Ανάλυση..."):
                    vtext="\n".join(f"- {k}: {val}" for k,val in vd.items())
                    pp=p.get
                    st.session_state.vitals_analysis=claude([{"role":"user","content":f"Patient: {pp('name')}, {pp('age')}yo {pp('sex')}, Hx: {pp('history','none')}, Meds: {pp('meds_raw','none')}\n\nVitals:\n{vtext}\n\nInterpret. Categorise each. Flag urgent findings. Be direct."}],system=kira_system(),max_tokens=1200)
            st.session_state.screen="triage"; st.rerun()

    # ── BP Estimation — Railway GPR API + Demographic fallback ───────────────
    st.divider()
    pr = st.session_state.profile
    age_val  = pr.get("age", 0)
    v_now    = st.session_state.vitals
    hr_val   = v_now.get("hr")
    wt_val   = v_now.get("weight") or pr.get("weight")
    ht_val   = v_now.get("height") or pr.get("height")
    bmi_val  = v_now.get("bmi")
    if not bmi_val and wt_val and ht_val:
        bmi_val = round(wt_val / ((ht_val/100)**2), 1)
    sex_val  = pr.get("sex","")
    gender_n = 1 if sex_val in ["Άνδρας","Male"] else 0

    bp_api_url = _secret("BP_API_URL","")
    api_result = None

    # Try Railway GPR model first (real ML prediction)
    if bp_api_url and age_val >= 18 and wt_val and ht_val and hr_val:
        try:
            payload = json.dumps({
                "age": int(age_val), "height": float(ht_val),
                "weight": float(wt_val), "hr": int(hr_val),
                "gender": gender_n
            }).encode()
            req = urllib.request.Request(
                f"{bp_api_url.rstrip('/')}/predict",
                data=payload,
                headers={"Content-Type":"application/json"},
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                api_result = json.loads(r.read())
        except Exception:
            api_result = None

    if age_val >= 18:
        risk = demographic_bp_risk(age_val, bmi_val, hr_val, wt_val, ht_val)
        label = risk["label_el"] if lang=="el" else risk["label_en"]
        note  = risk["note_el"]  if lang=="el" else risk["note_en"]
        color = risk["color"]

        if api_result:
            # ── ML model result (precise estimate with confidence interval) ──
            sbp     = api_result.get("sbp", "—")
            dbp     = api_result.get("dbp", "—")
            sbp_ci  = api_result.get("sbp_ci95", "")
            dbp_ci  = api_result.get("dbp_ci95", "")
            bmi_api = api_result.get("bmi", bmi_val or "—")
            title   = "Εκτίμηση Αρτηριακής Πίεσης — GPR Model" if lang=="el" else "Blood Pressure Estimate — GPR Model"
            subtitle= "Gaussian Process Regression · Chowdhury et al. (2020) · Railway API" if lang=="el" else "Gaussian Process Regression · Chowdhury et al. (2020) · Railway API"
            sbp_disp= f"{sbp} <span style='font-size:11px;color:#6B7280'>± {sbp_ci}</span>"
            dbp_disp= f"{dbp} <span style='font-size:11px;color:#6B7280'>± {dbp_ci}</span>"
            unit    = "mmHg"
            badge   = f"<div style='background:{color};color:white;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:700'>{label}</div><div style='font-size:9px;color:#6B7280;text-align:right;margin-top:4px'>GPR Model ✓</div>"
        else:
            # ── Demographic fallback (range estimate) ──
            sbp_disp= risk["sbp"]
            dbp_disp= risk["dbp"]
            unit    = "mmHg"
            title   = "Εκτίμηση Κινδύνου Αρτηριακής Πίεσης" if lang=="el" else "Blood Pressure Risk Estimate"
            subtitle= "Βάσει: ηλικία, ΔΜΣ, HR — Chowdhury et al. (2020)" if lang=="el" else "Based on: age, BMI, HR — Chowdhury et al. (2020)"
            badge   = f"<div style='background:{color};color:white;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:700'>{label}</div>"

        st.markdown(f"""
<div style="background:rgba(45,63,231,0.06);border:1px solid rgba(45,63,231,0.15);border-radius:14px;padding:18px 20px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
    <div>
      <div style="font-size:13px;font-weight:700;color:#1A1A2E">🩺 {title}</div>
      <div style="font-size:11px;color:#6B7280;margin-top:2px">{subtitle}</div>
    </div>
    {badge}
  </div>
  <div style="display:flex;gap:16px;margin-bottom:10px">
    <div style="background:white;border:1px solid #E0E5FF;border-radius:10px;padding:10px 16px;flex:1;text-align:center">
      <div style="font-size:11px;color:#6B7280">{"Εκτιμ. Συστολική" if lang=="el" else "Est. Systolic"}</div>
      <div style="font-size:20px;font-weight:700;color:{color}">{sbp_disp} <span style="font-size:12px;font-weight:400">{unit}</span></div>
    </div>
    <div style="background:white;border:1px solid #E0E5FF;border-radius:10px;padding:10px 16px;flex:1;text-align:center">
      <div style="font-size:11px;color:#6B7280">{"Εκτιμ. Διαστολική" if lang=="el" else "Est. Diastolic"}</div>
      <div style="font-size:20px;font-weight:700;color:{color}">{dbp_disp} <span style="font-size:12px;font-weight:400">{unit}</span></div>
    </div>
  </div>
  <div style="font-size:12px;color:#374151">{note}</div>
  <div style="font-size:10px;color:#9CA3AF;margin-top:6px">⚠️ {"Εκτίμηση μόνο — όχι αντικατάσταση πιεσομέτρου. Χρησιμοποιείστε πιστοποιημένο πιεσόμετρο για ακριβή μέτρηση." if lang=="el" else "Estimate only — not a substitute for a blood pressure monitor. Use a certified BP cuff for accurate measurement."}</div>
</div>
        """, unsafe_allow_html=True)

    # Navigation buttons
    col_b, col_s = st.columns([1, 3])
    with col_b:
        if st.button(t("back")): st.session_state.screen="intake"; st.rerun()
    with col_s:
        if st.button(("Δεν χρειάζομαι ζωτικά — Συνέχεια στα συμπτώματα →" if lang=="el"
                      else "I don't need vitals — Continue to symptoms →"), use_container_width=True):
            st.session_state.vitals={}; st.session_state.screen="triage"; st.rerun()

def render_vitals_summary():
    v=st.session_state.vitals
    if not v: return
    status=classify_vitals(v, age=st.session_state.profile.get("age"))
    LABELS={"hr":("❤️","Heart Rate","bpm"),"bp":("🩸","Blood Pressure","mmHg"),"br":("🌬️","Breathing","/min"),"spo2":("💧","SpO2","%"),"temp":("🌡️","Temp","°C"),"bmi":("⚖️","BMI","kg/m²")}
    badges=[]
    if "hr" in v: badges.append(("hr",v["hr"],"bpm",status.get("hr","green")))
    if "bp_sys" in v and "bp_dia" in v: badges.append(("bp",f"{v['bp_sys']}/{v['bp_dia']}","mmHg",status.get("bp","green")))
    if "br" in v: badges.append(("br",v["br"],"/min",status.get("br","green")))
    if "spo2" in v: badges.append(("spo2",v["spo2"],"%",status.get("spo2","green")))
    if "temp" in v: badges.append(("temp",v["temp"],"°C",status.get("temp","green")))
    if "bmi" in v: badges.append(("bmi",v["bmi"],"kg/m²",status.get("bmi","green")))
    if not badges: return
    cols=st.columns(len(badges))
    for i,(key,val,unit,col) in enumerate(badges):
        icon,label,_=LABELS.get(key,("","",""))
        with cols[i]:
            bg={"green":"#EDFBF0","yellow":"#FFFBEB","red":"#FEF2F2"}.get(col,"#F4F6FF")
            brd={"green":"#A3E6B5","yellow":"#FCD34D","red":"#FCA5A5"}.get(col,"#E0E5FF")
            st.markdown(f'<div style="background:{bg};border:1px solid {brd};border-radius:12px;padding:12px;text-align:center"><div style="font-size:18px">{icon}</div><div style="font-size:20px;font-weight:700">{val}</div><div style="font-size:10px;color:#6B7280">{unit}</div><div style="font-size:11px;color:#374151">{label}</div></div>',unsafe_allow_html=True)
    if st.session_state.vitals_analysis:
        with st.expander("📋 Ανάλυση ζωτικών" if st.session_state.lang=="el" else "📋 Vitals analysis"):
            st.markdown(st.session_state.vitals_analysis)

def render_photo_scan():
    """Photo health analysis (Florence-2 + Claude Vision). Lives inside the assessment."""
    p = st.session_state.profile
    lang = st.session_state.lang
    rf_key = _secret("ROBOFLOW_API_KEY","")
    st.caption(("Ανέβασε φωτογραφία για κλινική εκτίμηση" if lang=="el"
                else "Upload photo for clinical assessment"))

    SCAN_OPTS = {
        "el":[("eye","👁️ Μάτι"),("skin","🔬 Δέρμα/Εξάνθημα"),
              ("wound","🤕 Τραύμα/Πληγή"),("throat","🦷 Στόμα/Λαιμός"),
              ("nails","💅 Νύχια"),("body","🩹 Γενική Εμφάνιση")],
        "en":[("eye","👁️ Eye"),("skin","🔬 Skin/Rash"),
              ("wound","🤕 Wound"),("throat","🦷 Mouth/Throat"),
              ("nails","💅 Nails"),("body","🩹 Body/Lesion")],
    }
    opts   = SCAN_OPTS[lang]
    labels = [o[1] for o in opts]
    keys_  = [o[0] for o in opts]
    sel    = st.radio(("Τύπος σάρωσης" if lang=="el" else "Scan type"),
                      labels, horizontal=True, key="h_scan_type",
                      label_visibility="collapsed")
    scan_k = keys_[labels.index(sel)] if sel in labels else "skin"

    tips = {
        "eye":   {"el":"📸 Κοντά (10-15cm), καλό φωτισμό, ανοιχτό μάτι","en":"📸 Close-up (10-15cm), good light, eye open"},
        "skin":  {"el":"📸 Καθαρή εικόνα αλλοίωσης, φυσικό φωτισμό","en":"📸 Clear image of lesion, natural lighting"},
        "wound": {"el":"📸 Καλός φωτισμός, χωρίς αίμα να καλύπτει την πληγή","en":"📸 Good lighting, wound visible and clean"},
        "throat":{"el":"📸 Ανοιχτό στόμα, λαμπάκι αν υπάρχει","en":"📸 Open mouth, torch if available"},
        "nails": {"el":"📸 Κοντινή λήψη νυχιών σε λευκό φόντο","en":"📸 Close-up of nails on white background"},
        "body":  {"el":"📸 Ολόκληρη η πάσχουσα περιοχή στο κάδρο","en":"📸 Full affected area in frame"},
    }
    st.caption(tips.get(scan_k, tips["skin"])[lang])
    st.markdown(f'<div class="disclaimer">{"⚠️ Εργαλείο AI screening. Δεν αντικαθιστά κλινική εξέταση." if lang=="el" else "⚠️ AI screening tool. Does not replace clinical examination."}</div>', unsafe_allow_html=True)

    uploaded_photo = st.file_uploader(
        ("Φωτογραφία" if lang=="el" else "Upload photo"),
        type=["jpg","jpeg","png","webp","heic","heif"],
        key="human_photo_upload"
    )

    # Identity of the currently uploaded file — used to detect when the user
    # swaps to a different photo and we need to discard a stale preview.
    _current_file_id = (f"{uploaded_photo.name}|{uploaded_photo.size}|{scan_k}"
                        if uploaded_photo else None)

    # ── STAGE 1: Analyse button. Runs the vision pipeline and STORES the
    # result in session_state so it survives the rerun. Critically, the
    # 'Πρόσθεση στην εκτίμηση' button is NOT nested inside this if-block —
    # nested Streamlit buttons silently fail because the outer condition
    # becomes False on the next interaction.
    if uploaded_photo:
        c_img, c_info = st.columns([1,1])
        with c_img: st.image(uploaded_photo, use_container_width=True)
        with c_info:
            st.markdown(f"**{p.get('name','')}** · {sel}")
            if st.button("🔬 " + ("Ανάλυση" if lang=="el" else "Analyse"),
                         type="primary", use_container_width=True, key="analyse_human"):
                img_bytes = uploaded_photo.read()
                fname = uploaded_photo.name.lower()
                if fname.endswith((".heic",".heif")):
                    if HEIC_OK:
                        try: img_bytes, img_type = convert_heic_human(img_bytes)
                        except Exception as e: st.error(f"HEIC: {e}"); st.stop()
                    else:
                        st.error("Add pillow-heif to requirements.txt"); st.stop()
                else:
                    img_type = "image/jpeg"
                    if fname.endswith(".png"):  img_type = "image/png"
                    if fname.endswith(".webp"): img_type = "image/webp"

                img_b64 = _b64.b64encode(img_bytes).decode()

                with st.spinner("Ο Asklepios αναλύει τη φωτογραφία..." if lang=="el" else "Asklepios is analysing the photo..."):
                    f2_desc = ""
                    if rf_key:
                        f2 = florence2_human(img_b64, scan_k, rf_key)
                        if f2.get("ok"): f2_desc = f2.get("description","")

                    # Clinical context from the ongoing assessment so the photo is read
                    # WITHIN the reported complaint — not as an isolated, context-free image.
                    conv = st.session_state.triage_chat
                    convo_txt = "\n".join(
                        f"{'Ασθενής' if m['role']=='user' else 'Asklepios'}: {m['content']}"
                        for m in conv[-6:]
                    ) if conv else ("Δεν έχει καταγραφεί συνομιλία ακόμη." if lang=="el" else "No conversation yet.")
                    ctx_el = (f"ΚΛΙΝΙΚΟ ΠΛΑΙΣΙΟ (ο ασθενής έχει ΗΔΗ περιγράψει το πρόβλημα):\n"
                              f"Ασθενής: {p.get('age','?')} ετών, {p.get('sex','')}. Ιστορικό: {p.get('history','') or '—'}.\n"
                              f"Συζήτηση μέχρι τώρα:\n{convo_txt}\n\n"
                              f"Η φωτογραφία αφορά ΑΥΤΟ το παράπονο. Ερμήνευσέ την ΜΕΣΑ σε αυτό το πλαίσιο. "
                              f"ΜΗΝ αλλάζεις την ανατομική περιοχή ή το πρόβλημα που έχει ήδη περιγραφεί (π.χ. αν ο ασθενής λέει αγκώνας, μην το μετατρέπεις σε μασχάλη/θώρακα). "
                              f"ΜΗΝ εφευρίσκεις νέα διάγνωση ή νέο επίπεδο επείγοντος που έρχεται σε αντίθεση με την τρέχουσα εκτίμηση. "
                              f"Αν η εικόνα είναι ασαφής ή δεν προσθέτει κάτι, πες το ειλικρινά.")
                    ctx_en = (f"CLINICAL CONTEXT (the patient has ALREADY described the problem):\n"
                              f"Patient: {p.get('age','?')}yo {p.get('sex','')}. History: {p.get('history','') or '—'}.\n"
                              f"Conversation so far:\n{convo_txt}\n\n"
                              f"The photo relates to THIS complaint. Interpret it WITHIN this context. "
                              f"Do NOT change the anatomical region or the problem already described (e.g. if the patient says elbow, do not turn it into armpit/chest). "
                              f"Do NOT invent a new diagnosis or a new urgency level that contradicts the ongoing assessment. "
                              f"If the image is unclear or adds nothing, say so honestly.")
                    clin_ctx = (ctx_el if lang=="el" else ctx_en)

                    base_prompt = HUMAN_SCAN_PROMPTS.get(scan_k, HUMAN_SCAN_PROMPTS["skin"])
                    rf_context  = f"\n\nFLORENCE-2 DESCRIPTION: {f2_desc}" if f2_desc else ""
                    suffix_el   = "\n\nΔώσε ΣΥΜΠΛΗΡΩΜΑΤΙΚΑ ΟΠΤΙΚΑ ΕΥΡΗΜΑΤΑ (όχι ξεχωριστή διάγνωση): **ΟΡΑΤΑ ΕΥΡΗΜΑΤΑ** (μόνο ό,τι φαίνεται) | **ΣΥΜΒΑΤΟΤΗΤΑ με το παράπονο** (στηρίζει/δεν στηρίζει την τρέχουσα εκτίμηση) | **ΣΗΜΕΙΑ ΠΡΟΣΟΧΗΣ** (μόνο αν φαίνονται καθαρά στην εικόνα). Σύντομα και συνεπή με την τρέχουσα εκτίμηση."
                    suffix_en   = "\n\nGive SUPPLEMENTARY VISUAL FINDINGS (not a separate diagnosis): **VISIBLE FINDINGS** (only what is visible) | **CONSISTENCY with the complaint** (supports/does not support the current assessment) | **WARNING SIGNS** (only if clearly visible in the image). Brief and consistent with the current assessment."
                    full_prompt = clin_ctx + "\n\n" + base_prompt + rf_context + (suffix_el if lang=="el" else suffix_en)
                    sys_prompt  = ("Είσαι ο βοηθός οπτικής εξέτασης του Asklepios AI. Συμπληρώνεις μια εκτίμηση που ήδη εξελίσσεται — ΔΕΝ ξεκινάς νέα. Μένεις πιστός στο παράπονο και στην ανατομική περιοχή που έχει δηλωθεί, είσαι ακριβής, προσεκτικός και δεν δραματοποιείς." if lang=="el"
                                   else "You are Asklepios AI's visual-exam assistant. You SUPPLEMENT an assessment already in progress — you do NOT start a new one. Stay faithful to the stated complaint and anatomical region, be accurate, cautious, and do not dramatise.")
                    analysis = claude_vision_human(img_b64, img_type, full_prompt, sys_prompt)

                # Persist the preview so the next rerun renders Stage 2 at top
                # level — NOT nested inside this button block (which would die
                # on the next interaction).
                st.session_state["_photo_preview"] = {
                    "file_id":     _current_file_id,
                    "scan_type":   scan_k,
                    "scan_label":  sel,
                    "florence_desc": f2_desc,
                    "analysis":    analysis,
                }
                st.rerun()

    # ── STAGE 2: render the preview + 'Add to assessment' button at TOP LEVEL
    # (not nested), so the button actually fires on click.
    preview = st.session_state.get("_photo_preview")
    if preview:
        # If the user uploaded a different file or changed scan type, the old
        # preview is stale — discard it so they can re-analyse the new one.
        if uploaded_photo and preview.get("file_id") and preview["file_id"] != _current_file_id:
            st.session_state.pop("_photo_preview", None)
            preview = None
    if preview:
        analysis = preview["analysis"]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(analysis)
        st.markdown('</div>', unsafe_allow_html=True)

        urgent_kw = ["urgent","immediate","επείγον","άμεσα","ιατρό αμέσως","emergency","melanoma","cancer","carcinoma","καρκίν"]
        if any(k.lower() in analysis.lower() for k in urgent_kw):
            st.error("🚨 " + ("Επείγοντα ευρήματα — επικοινωνήστε με ιατρό ΑΜΕΣΑ" if lang=="el"
                              else "Urgent findings — contact a doctor IMMEDIATELY"))

        if st.button("➤ " + ("Πρόσθεση στην εκτίμηση" if lang=="el" else "Add to assessment"),
                     type="primary", use_container_width=True, key="photo_to_triage_h"):
            _lbl = preview["scan_label"]
            msg = (f"Αποτέλεσμα φωτογραφικής ανάλυσης ({_lbl}):\n\n{analysis}"
                   if lang=="el" else
                   f"Photo analysis result ({_lbl}):\n\n{analysis}")
            st.session_state.triage_chat.append({"role":"user","content":msg})
            # Append to the photo findings LIST so multiple uploads accumulate
            # and all become visible in the final report.
            _pf = st.session_state.get("photo_findings")
            if not isinstance(_pf, list):
                _pf = []
            _pf.append({
                "scan_type":     preview["scan_type"],
                "scan_label":    preview["scan_label"],
                "florence_desc": preview.get("florence_desc",""),
                "analysis":      analysis,
            })
            st.session_state["photo_findings"] = _pf
            st.session_state["photo_added"]    = True
            st.session_state.pop("_photo_preview", None)
            st.rerun()
    elif not uploaded_photo:
        st.info("👆 " + ("Ανεβάστε φωτογραφία για να ξεκινήσει η ανάλυση" if lang=="el"
                        else "Upload a photo to begin analysis"))


def render_lab_analysis():
    """Lab PDF/image upload + Claude interpretation. 2-stage flow at top level
    (no nested buttons — same fix as photo scan).
    
    Privacy: file is sent to Claude API for analysis and discarded immediately.
    Nothing about the lab values is stored on our servers.
    """
    p = st.session_state.profile
    lang = st.session_state.lang
    st.caption(("Ανέβασε PDF ή φωτογραφία αιματολογικών, ορμονολογικών, βιοχημικών ή ουρολογικών εξετάσεων."
                if lang=="el" else
                "Upload PDF or photo of blood, hormonal, biochemistry, or urinalysis results."))
    st.markdown(f'<div class="disclaimer">{"⚠️ Εκπαιδευτικό εργαλείο, δεν αντικαθιστά ιατρό. Το αρχείο δεν αποθηκεύεται στους server μας." if lang=="el" else "⚠️ Educational tool, does not replace a doctor. The file is not stored on our servers."}</div>', unsafe_allow_html=True)
    
    uploaded_lab = st.file_uploader(
        ("Εξετάσεις (PDF, JPG, PNG)" if lang=="el" else "Lab tests (PDF, JPG, PNG)"),
        type=["pdf","jpg","jpeg","png","webp"],
        key="lab_upload"
    )
    
    _current_file_id = (f"{uploaded_lab.name}|{uploaded_lab.size}"
                        if uploaded_lab else None)
    
    # ── STAGE 1: trigger analysis ──
    if uploaded_lab:
        c_info, c_btn = st.columns([2,1])
        with c_info:
            st.markdown(f"**📄 {uploaded_lab.name}** · {round(uploaded_lab.size/1024)} KB")
        with c_btn:
            if st.button("🔬 " + ("Ανάλυση" if lang=="el" else "Analyse"),
                         type="primary", use_container_width=True, key="analyse_lab"):
                file_bytes = uploaded_lab.read()
                fname = uploaded_lab.name.lower()
                if fname.endswith(".pdf"):
                    mime = "application/pdf"
                elif fname.endswith(".png"):
                    mime = "image/png"
                elif fname.endswith(".webp"):
                    mime = "image/webp"
                else:
                    mime = "image/jpeg"
                
                with st.spinner(("Ο Asklepios ερμηνεύει τα αποτελέσματα..." if lang=="el"
                                else "Asklepios is interpreting the results...")):
                    analysis = claude_analyze_lab(
                        file_bytes, mime,
                        p, st.session_state.triage_chat, lang,
                        file_name=uploaded_lab.name,
                    )
                
                # Persist preview to state — Stage 2 renders OUTSIDE this button
                # block so the "Add to assessment" button actually fires on click.
                st.session_state["_lab_preview"] = {
                    "file_id":   _current_file_id,
                    "file_name": uploaded_lab.name,
                    "mime":      mime,
                    "analysis":  analysis,
                }
                st.rerun()
    
    # ── STAGE 2: preview + Add-to-assessment (TOP LEVEL — not nested) ──
    preview = st.session_state.get("_lab_preview")
    if preview:
        # Stale preview detection: user uploaded a different file
        if uploaded_lab and preview.get("file_id") and preview["file_id"] != _current_file_id:
            st.session_state.pop("_lab_preview", None)
            preview = None
    if preview:
        analysis = preview["analysis"]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(analysis)
        st.markdown('</div>', unsafe_allow_html=True)
        
        urgent_kw = ["επείγον","άμεσα","emergency","critical","κρίσιμ","ιατρό αμέσως","immediately"]
        if any(k.lower() in analysis.lower() for k in urgent_kw):
            st.error("🚨 " + ("Ευρήματα που χρήζουν άμεσης ιατρικής αξιολόγησης"
                              if lang=="el" else
                              "Findings requiring immediate medical evaluation"))
        
        if st.button("➤ " + ("Πρόσθεση στην εκτίμηση" if lang=="el" else "Add to assessment"),
                     type="primary", use_container_width=True, key="lab_to_triage"):
            _name = preview["file_name"]
            msg = (f"Αποτέλεσμα ανάλυσης εξετάσεων ({_name}):\n\n{analysis}"
                   if lang=="el" else
                   f"Lab analysis result ({_name}):\n\n{analysis}")
            st.session_state.triage_chat.append({"role":"user","content":msg})
            _lf = st.session_state.get("lab_findings")
            if not isinstance(_lf, list):
                _lf = []
            _lf.append({
                "file_name": _name,
                "analysis":  analysis,
            })
            st.session_state["lab_findings"] = _lf
            st.session_state["lab_added"]    = True
            st.session_state.pop("_lab_preview", None)
            st.rerun()
    elif not uploaded_lab:
        st.info("👆 " + ("Ανεβάστε PDF ή φωτογραφία για να ξεκινήσει η ανάλυση"
                        if lang=="el" else
                        "Upload a PDF or photo to begin analysis"))


def render_triage():
    render_stepper("triage")
    p=st.session_state.profile
    nm = p.get("name","")
    render_doc_header(
        "Ας μιλήσουμε για τα συμπτώματα", "Let's talk about your symptoms",
        icon="💬",
        sub_el=(f"συνομιλία με {nm}" if nm else "Πες τι σε απασχολεί — μία ερώτηση κάθε φορά"),
        sub_en=(f"chat with {nm}" if nm else "Tell me what's bothering you — one question at a time"),
    )
    render_vitals_summary()
    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>',unsafe_allow_html=True)
    # Symptom quick-select: only BEFORE the conversation starts, so once chatting
    # the previous Q&A stays visible instead of being buried under the buttons.
    if not st.session_state.triage_chat:
        st.info(("👇 Βήμα 3 — Περίγραψε εδώ τι σε απασχολεί (π.χ. «πόνος στο μάτι 2 μέρες»). "
                 "Ο Asklepios θα σου κάνει ερωτήσεις και στο τέλος θα δημιουργήσει αναφορά."
                 if st.session_state.lang=="el" else
                 "👇 Step 3 — Describe what's bothering you (e.g. 'eye pain for 2 days'). "
                 "Asklepios will ask follow-up questions and then generate a report."))
        chips, _chips_label = _symptom_chips(st.session_state.profile, st.session_state.lang)
        _cap = ("Γρήγορη επιλογή" if st.session_state.lang=="el" else "Quick select")
        if _chips_label:
            _cap += f" ({_chips_label})"
        st.caption(_cap + ":")
        _PER_ROW = 3
        for _rs in range(0, len(chips), _PER_ROW):
            _row = chips[_rs:_rs+_PER_ROW]
            _cc = st.columns(len(_row))
            for _j, chip in enumerate(_row):
                _i = _rs + _j
                with _cc[_j]:
                    sel = chip in st.session_state.symptom_chips
                    if st.button(("✓ " if sel else "")+chip, key=f"chip_{_i}", use_container_width=True):
                        if chip in st.session_state.symptom_chips: st.session_state.symptom_chips.remove(chip)
                        else: st.session_state.symptom_chips.append(chip)
                        st.rerun()
        if st.session_state.symptom_chips:
            if st.button("➤ "+("Αποστολή επιλεγμένων" if st.session_state.lang=="el" else "Send selected"),type="primary"):
                msg=("Κύρια συμπτώματα: " if st.session_state.lang=="el" else "Main symptoms: ")+", ".join(st.session_state.symptom_chips)
                st.session_state.triage_chat.append({"role":"user","content":msg}); st.session_state.symptom_chips=[]; st.rerun()
    st.divider()
    for msg in st.session_state.triage_chat:
        with st.chat_message(msg["role"], avatar="🩺" if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])
    # Context-aware vitals: suggest the SPECIFIC measurement that fits the symptoms.
    # Scan button appears only for the cardiac category (camera → heart rate only).
    _lang = st.session_state.lang
    _relv = _relevant_vitals()
    if (any(m["role"]=="assistant" for m in st.session_state.triage_chat)
            and not st.session_state.vitals
            and _relv
            and not st.session_state.get("_vitals_nudge_off")):
        _names = ", ".join(dict.fromkeys(c["el" if _lang=="el" else "en"] for c in _relv))
        _show_scan = any(c["scan"] for c in _relv)
        st.warning("🩺 " + (f"Με βάση όσα περιγράφεις, θα βοηθούσε να μετρηθεί: {_names}. Θες να το κάνεις τώρα;"
                            if _lang=="el" else
                            f"Based on what you describe, it would help to measure: {_names}. Want to do it now?"))
        _cols = st.columns(3 if _show_scan else 2)
        with _cols[0]:
            if st.button(("✏️ Καταχώρηση" if _lang=="el" else "✏️ Enter values"), key="nudge_manual", use_container_width=True):
                st.session_state.screen = "vitals"; st.rerun()
        _ci = 1
        if _show_scan:
            with _cols[_ci]:
                _fs = _secret("FACESCAN_URL","https://asklepiosnurse.netlify.app")
                _ku = _secret("ASKLEPIOS_URL","https://asklepiosainurse.up.railway.app")
                _link = f"{_fs}?kira_url={urllib.parse.quote(_ku)}"
                _save_session_for_external_nav()
                st.markdown(f'<a href="{_link}" target="_blank" style="display:block;text-align:center;padding:8px;border-radius:8px;background:#2D3FE7;color:white;text-decoration:none;font-weight:600;font-size:13px">📷 {"Σάρωση" if _lang=="el" else "Scan"}</a>', unsafe_allow_html=True)
            _ci += 1
        with _cols[_ci]:
            if st.button(("Όχι τώρα" if _lang=="el" else "Not now"), key="nudge_off", use_container_width=True):
                st.session_state["_vitals_nudge_off"] = True; st.rerun()
    # Photo analysis appears only after an initial assessment AND only when the
    # complaint is something visible (skin, eye, wound, throat, nails...). For
    # non-visual issues (e.g. chest pain) a photo adds nothing, so it stays hidden.
    if any(m["role"]=="assistant" for m in st.session_state.triage_chat) and _visual_relevant():
        _pf_list = st.session_state.get("photo_findings") or []
        _has_photo = isinstance(_pf_list, list) and len(_pf_list) > 0
        # Label adapts so the user knows multiple uploads are allowed.
        # Collapsed by default once a photo has been added — keeps the chat
        # uncluttered but the option remains one click away.
        _exp_label = (("📷 Ανέβασε άλλη φωτογραφία (αν χρειαστεί)"
                       if _has_photo else
                       "📷 Ανάλυση φωτογραφίας (προαιρετικό)")
                      if st.session_state.lang=="el" else
                      ("📷 Upload another photo (if needed)"
                       if _has_photo else
                       "📷 Photo analysis (optional)"))
        with st.expander(_exp_label, expanded=not _has_photo):
            if _has_photo:
                st.caption("💡 " + (f"Έχουν προστεθεί {len(_pf_list)} φωτογραφία/ες. "
                                    "Ανέβασε νέα μόνο αν ο Asklepios το ζητήσει "
                                    "ή αν θέλεις άλλη πλευρά / άλλο σημείο."
                                    if st.session_state.lang=="el" else
                                    f"{len(_pf_list)} photo(s) already added. "
                                    "Upload a new one only if Asklepios asks "
                                    "or you want a different angle/area."))
            else:
                st.caption("💡 " + ("Προαιρετικό. Αν ο Asklepios χρειαστεί φωτογραφία για ορατό σύμπτωμα, "
                                    "θα στο αναφέρει — αλλά μπορείς να ανεβάσεις και προληπτικά."
                                    if st.session_state.lang=="el" else
                                    "Optional. If Asklepios needs a photo for a visible symptom, "
                                    "it will say so — but you can also upload proactively."))
            render_photo_scan()
    # Lab analysis — always available once Asklepios has started talking, since
    # blood/hormonal/urinalysis results help for ANY complaint, not just visual.
    if any(m["role"]=="assistant" for m in st.session_state.triage_chat):
        _lf_list = st.session_state.get("lab_findings") or []
        _has_lab = isinstance(_lf_list, list) and len(_lf_list) > 0
        _lab_label = (("🧪 Ανέβασε άλλες εξετάσεις (αν χρειάζεται)"
                       if _has_lab else
                       "🧪 Ανάλυση εξετάσεων (αιματολογικά, ορμονολογικά, ούρα) — προαιρετικό")
                      if st.session_state.lang=="el" else
                      ("🧪 Upload more lab tests (if needed)"
                       if _has_lab else
                       "🧪 Lab analysis (blood, hormonal, urinalysis) — optional"))
        with st.expander(_lab_label, expanded=False):
            if _has_lab:
                st.caption("💡 " + (f"{len(_lf_list)} αρχείο/α εξετάσεων έχουν προστεθεί. "
                                    "Ανέβασε άλλο αν έχεις περισσότερες εξετάσεις."
                                    if st.session_state.lang=="el" else
                                    f"{len(_lf_list)} lab file(s) added. "
                                    "Upload another if you have more tests."))
            else:
                st.caption("💡 " + ("Ανέβασε εργαστηριακές εξετάσεις (PDF ή φωτογραφία) "
                                    "και ο Asklepios θα τις ερμηνεύσει ΜΕΣΑ στο πλαίσιο των συμπτωμάτων σου."
                                    if st.session_state.lang=="el" else
                                    "Upload lab tests (PDF or photo) and Asklepios will "
                                    "interpret them WITHIN the context of your symptoms."))
            render_lab_analysis()
    # Confirmation after a photo was added — guide the user to keep answering
    if st.session_state.get("photo_added"):
        last_q = next((m["content"] for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"), "")
        if st.session_state.lang=="el":
            st.success("✅ Η ανάλυση της εικόνας προστέθηκε στην εκτίμηση. Συνέχισε απαντώντας στην τελευταία ερώτηση του Asklepios παρακάτω.")
        else:
            st.success("✅ The image analysis was added to the assessment. Continue by answering Asklepios's last question below.")
        if last_q:
            st.info(("🩺 Τελευταία ερώτηση: " if st.session_state.lang=="el" else "🩺 Last question: ") + last_q)
    # Same confirmation pattern for lab results — keeps the user on track
    if st.session_state.get("lab_added"):
        last_q = next((m["content"] for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"), "")
        if st.session_state.lang=="el":
            st.success("✅ Η ανάλυση των εξετάσεων προστέθηκε στην εκτίμηση. Συνέχισε απαντώντας στον Asklepios.")
        else:
            st.success("✅ The lab analysis was added to the assessment. Continue chatting with Asklepios.")
    ready_phrases=["έχω αρκετά στοιχεία","μπορούμε να δημιουργήσουμε","i have enough information","we can generate","full clinical report","πλήρη αναφορά"]
    last_kira=next((m["content"].lower() for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"),"")
    triage_ready=any(ph in last_kira for ph in ready_phrases)
    # ── Voice input ───────────────────────────────────────────────────────────
    # Always shown — Tab 1 (Web Speech API) needs no API key at all.
    # Tab 2 (Whisper) needs Groq or OpenAI key.
    # Critical for 60+ demographic: IOBE data shows this group has the highest
    # unmet healthcare needs and lowest digital comfort.
    _voice_lbl = ("🎤 Φωνητική εισαγωγή (μίλα αντί να γράφεις)"
                  if st.session_state.lang=="el" else
                  "🎤 Voice input (speak instead of typing)")
    with st.expander(_voice_lbl, expanded=False):
        # Web Speech API — uses st.iframe (HTML string mode)
        _has_stt = bool(get_groq_key() or get_openai_key())
        _whisper_tab_lbl = ("🎙️ Whisper AI (Ελληνικά ✓)"
                            if _has_stt else
                            "🎙️ Whisper AI (απαιτεί OPENAI_API_KEY)")
        _wsapi_tab_lbl = ("🌐 Browser (δωρεάν, Chrome/Safari)"
                          if st.session_state.lang=="el" else
                          "🌐 Browser (free, Chrome/Safari)")
        _v_tab1, _v_tab2 = st.tabs([_whisper_tab_lbl, _wsapi_tab_lbl])

        # ── Tab 1: st.audio_input + Whisper ──────────────────────────────────
        with _v_tab1:
            if not _has_stt:
                st.info("💡 " + ("Πρόσθεσε `OPENAI_API_KEY` ή `GROQ_API_KEY` στα Railway env vars για να ενεργοποιήσεις το Whisper."
                                 if st.session_state.lang=="el" else
                                 "Add `OPENAI_API_KEY` or `GROQ_API_KEY` to Railway env vars to enable Whisper."))
            else:
                st.caption("💡 " + ("Πάτησε το μικρόφωνο, μίλα φυσικά, σταμάτα. Η ηχογράφηση δεν αποθηκεύεται."
                                    if st.session_state.lang=="el" else
                                    "Press the microphone, speak naturally, stop. Audio is not stored."))
                _audio = st.audio_input(
                    ("Πες τι νιώθεις" if st.session_state.lang=="el" else "Say what you feel"),
                    key="voice_input_widget",
                    label_visibility="collapsed",
                )
                # Track by hash so the same audio isn't transcribed twice across reruns
                if _audio is not None:
                    _audio_bytes = _audio.getvalue()
                    _audio_hash = hashlib.sha256(_audio_bytes).hexdigest()[:16]
                    if st.session_state.get("_voice_last_hash") != _audio_hash:
                        with st.spinner("🎙️ " + ("Μεταγραφή με Whisper..." if st.session_state.lang=="el"
                                                  else "Transcribing with Whisper...")):
                            text, _ = transcribe_audio(
                                _audio_bytes, lang=st.session_state.lang,
                                mime="audio/webm", filename="voice.webm",
                            )
                        st.session_state["_voice_last_hash"] = _audio_hash
                        if text and not text.startswith("⚠️"):
                            st.session_state["_voice_transcript"] = text
                            st.rerun()
                        else:
                            st.error(text or "—")
                # Transcript review + confirm (never auto-submit — Whisper can mishear)
                _pending = st.session_state.get("_voice_transcript")
                if _pending:
                    st.success("📝 " + ("Μεταγραφή — διόρθωσε αν χρειαστεί:" if st.session_state.lang=="el"
                                        else "Transcription — edit if needed:"))
                    _edited = st.text_area("transcript_edit", value=_pending,
                                           label_visibility="collapsed", height=80,
                                           key="voice_edit_area")
                    _vc1, _vc2 = st.columns([3, 1])
                    with _vc1:
                        if st.button(("✓ Αποστολή στον Asklepios" if st.session_state.lang=="el"
                                      else "✓ Send to Asklepios"),
                                     type="primary", use_container_width=True, key="voice_send"):
                            _msg = _edited.strip() or _pending
                            st.session_state.triage_chat.append({"role":"user","content":_msg})
                            st.session_state.pop("photo_added", None)
                            st.session_state.pop("lab_added", None)
                            st.session_state.pop("_voice_transcript", None)
                            st.session_state.pop("_voice_last_hash", None)
                            st.session_state["_voice_send_pending"] = True
                            st.rerun()
                    with _vc2:
                        if st.button(("🗑️ Ακύρωση" if st.session_state.lang=="el" else "🗑️ Cancel"),
                                     use_container_width=True, key="voice_cancel"):
                            st.session_state.pop("_voice_transcript", None)
                            st.session_state.pop("_voice_last_hash", None)
                            st.rerun()

        # ── Tab 2: Web Speech API — browser-native, no API key, Greek support ─
        # Same pattern as HAL project. Works on Chrome/Safari.
        # Result shown below the widget for the user to copy → paste into chat.
        with _v_tab2:
            _ws_lang = "el-GR" if st.session_state.lang=="el" else "en-US"
            _ws_hint = ("Μίλα φυσικά — το κείμενο εμφανίζεται αυτόματα."
                        if st.session_state.lang=="el" else
                        "Speak naturally — text appears automatically.")
            _ws_copy_lbl = "📋 Αντιγραφή" if st.session_state.lang=="el" else "📋 Copy"
            _ws_not_sup = ("Δεν υποστηρίζεται — χρησιμοποίησε Chrome ή Safari"
                           if st.session_state.lang=="el" else
                           "Not supported — use Chrome or Safari")
            _ws_listening = "🔴 Ακούω..." if st.session_state.lang=="el" else "🔴 Listening..."
            _ws_idle = ("Πάτησε 🎙️ για ηχογράφηση" if st.session_state.lang=="el"
                        else "Press 🎙️ to record")
            _ws_done = ("✅ Αντίγραψε και επικόλλησε στο chat ↓"
                        if st.session_state.lang=="el" else
                        "✅ Copy and paste into chat ↓")
            st.iframe(f"""<!DOCTYPE html><html><head><style>
body{{margin:0;padding:0;font-family:system-ui,sans-serif;background:transparent}}
#wrap{{display:flex;align-items:flex-start;gap:10px;background:#F0F4FF;border:1px solid #C7D2FE;border-radius:10px;padding:10px 14px;flex-wrap:wrap}}
#mic{{background:none;border:2px solid #2D3FE7;border-radius:50%;width:38px;height:38px;font-size:18px;cursor:pointer;color:#2D3FE7;flex-shrink:0;transition:all .2s}}
#mic.active{{background:#2D3FE7;color:white;box-shadow:0 0 0 4px rgba(45,63,231,.15)}}
#status{{font-size:12px;color:#6B7280;flex:1;padding-top:10px}}
#result{{display:none;width:100%;background:white;border:1px solid #C7D2FE;border-radius:8px;padding:8px 12px;font-size:14px;color:#1F2937;line-height:1.5;margin-top:6px;word-break:break-word}}
#copy{{display:none;background:#2D3FE7;color:white;border:none;border-radius:8px;padding:8px 18px;font-weight:700;cursor:pointer;font-size:13px;margin-top:6px}}
#copy:hover{{background:#1E30CC}}
</style></head><body>
<div id="wrap">
  <button id="mic" onclick="toggleVoice()">🎙️</button>
  <div id="status">{_ws_idle}</div>
  <div id="result"></div>
  <button id="copy" onclick="copyText()">{_ws_copy_lbl}</button>
</div>
<script>
var recognition,listening=false,transcript="";
function toggleVoice(){{
  if(!("webkitSpeechRecognition"in window||"SpeechRecognition"in window)){{
    document.getElementById("status").textContent="{_ws_not_sup}";return;
  }}
  if(listening){{recognition.stop();return;}}
  recognition=new(window.SpeechRecognition||window.webkitSpeechRecognition)();
  recognition.lang="{_ws_lang}";recognition.interimResults=true;recognition.continuous=false;
  recognition.onstart=function(){{
    listening=true;
    document.getElementById("mic").classList.add("active");
    document.getElementById("status").textContent="{_ws_listening}";
    document.getElementById("result").style.display="none";
    document.getElementById("copy").style.display="none";
  }};
  recognition.onresult=function(e){{
    transcript=Array.from(e.results).map(r=>r[0].transcript).join("");
    document.getElementById("result").textContent=transcript;
    document.getElementById("result").style.display="block";
  }};
  recognition.onend=function(){{
    listening=false;
    document.getElementById("mic").classList.remove("active");
    if(transcript){{
      document.getElementById("status").textContent="{_ws_done}";
      document.getElementById("copy").style.display="inline-block";
    }}else{{
      document.getElementById("status").textContent="{_ws_idle}";
    }}
  }};
  recognition.onerror=function(e){{
    listening=false;
    document.getElementById("mic").classList.remove("active");
    document.getElementById("status").textContent="Error: "+e.error;
  }};
  recognition.start();
}}
function copyText(){{
  if(!transcript)return;
  navigator.clipboard.writeText(transcript).then(function(){{
    var b=document.getElementById("copy");
    b.textContent="✅ OK!";
    setTimeout(function(){{b.textContent="{_ws_copy_lbl}";}},2000);
  }});
}}
</script></body></html>""", height=100)
            st.caption("↑ " + ("Αντίγραψε το κείμενο και επικόλλησέ το στο chat παρακάτω."
                                if st.session_state.lang=="el" else
                                "Copy the text and paste it into the chat below."))

    user_input=st.chat_input(t("triage_placeholder"),key="triage_input")
    _auto_reply = st.session_state.pop("_scan_reply_pending", False)
    _voice_reply = st.session_state.pop("_voice_send_pending", False)
    if user_input or _auto_reply or _voice_reply:
        if user_input:
            st.session_state.pop("photo_added", None)
            st.session_state.pop("lab_added", None)
            st.session_state.triage_chat.append({"role":"user","content":user_input})
        with st.spinner("Asklepios..."):
            pp=p.get
            _flags = []
            if pp("pregnancy"):
                _flags.append("ΕΓΚΥΟΣ — πρόσεξε αντενδείξεις φαρμάκων/εξετάσεων κατηγορίας D/X" if st.session_state.lang=="el"
                              else "PREGNANT — flag drug/test contraindications (Category D/X)")
            if pp("for_whom") == "other":
                _flags.append("Αξιολόγηση από φροντιστή για άλλο άτομο" if st.session_state.lang=="el"
                              else "Caregiver-mode: user is asking on behalf of another person")
            _age_v = pp("age", 0) or 0
            if _age_v < 18:
                _flags.append(f"ΠΑΙΔΙΑΤΡΙΚΟΣ ΑΣΘΕΝΗΣ (ηλικία {_age_v}) — χρησιμοποίησε παιδιατρικές δόσεις/όρια"
                              if st.session_state.lang=="el" else
                              f"PEDIATRIC PATIENT (age {_age_v}) — use pediatric dosing/ranges")
            _flags_str = (" | ".join(_flags) + " | ") if _flags else ""
            profile_ctx=f"Patient: {pp('name')}, {pp('age')}yo {pp('sex')}, {_flags_str}Hx: {pp('history','none')}, Allergies: {pp('allergies','none')}, Meds: {pp('meds_raw','none')}"
            vitals_ctx="Vitals: "+", ".join(f"{k}={val}" for k,val in st.session_state.vitals.items()) if st.session_state.vitals else "Vitals: not provided"
            system_ctx=kira_system()+f"\n\n{profile_ctx}\n{vitals_ctx}"
            reply=claude([{"role":m["role"],"content":m["content"]} for m in st.session_state.triage_chat],system=system_ctx,max_tokens=1500)
            if reply and reply.strip() and reply.strip()[-1] not in ".!?»)": reply=reply.rstrip()+" ..."
        st.session_state.triage_chat.append({"role":"assistant","content":reply}); st.rerun()
    col_b,col_r=st.columns([1,2])
    with col_b:
        if st.button(t("back")): st.session_state.screen="vitals"; st.rerun()
    with col_r:
        enabled=triage_ready or len(st.session_state.triage_chat)>=6
        if st.button(t("generate_report"),type="primary",use_container_width=True,disabled=not enabled):
            st.session_state.screen="report"; st.rerun()
    if not enabled:
        st.caption("Συνεχίστε — ο Asklepios θα σας ειδοποιήσει όταν έχει αρκετά." if st.session_state.lang=="el" else "Continue — Asklepios will let you know when it has enough.")

# ── PNOE-inspired report helpers ──────────────────────────────────────────────
# Inspired by the PNOE Metabolic Blueprint report (Frank Shallenberger), which
# packages each recommendation block as three categories (EXERCISE / NUTRITION /
# LIFESTYLE) and uses a 5-level scale. We adapt both ideas:
#   1. Claude is asked to emit a delimited RECS block at the end of the report
#      with three personalised buckets. We parse it out and render as a styled
#      3-column card (PDF/TXT/WhatsApp also get the clean text).
#   2. The existing Wellness Score is augmented with a 5-segment scale bar
#      (Severe Limit. → Limit. → Neutral → Good → Excellent) matching PNOE's
#      visual language.

def _extract_recs(report_text):
    """Pull <<<RECS ... RECS>>> block out of the Claude report.
    Returns (cleaned_text, recs_dict_or_None). Graceful: if no block found,
    returns the original text unchanged and None."""
    import re as _re_r
    if not report_text:
        return report_text, None
    m = _re_r.search(r"<<<RECS\s*(.*?)\s*RECS>>>", report_text, _re_r.DOTALL)
    if not m:
        return report_text, None
    block = m.group(1)
    cleaned = (report_text[:m.start()].rstrip() + "\n\n" + report_text[m.end():].lstrip()).strip()
    recs = {}
    # Multi-line tolerant: accumulate until next label or end
    current = None
    for line in block.splitlines():
        s = line.strip()
        if not s: continue
        upper = s.upper()
        for tag, key in (("CONDITION:", "condition"),
                         ("EXERCISE:", "exercise"),
                         ("NUTRITION:", "nutrition"),
                         ("LIFESTYLE:", "lifestyle")):
            if upper.startswith(tag):
                current = key
                recs[key] = s[len(tag):].strip()
                break
        else:
            # Continuation line
            if current:
                recs[current] = (recs.get(current, "") + " " + s).strip()
    return cleaned, (recs if recs else None)


def _render_recs_card(recs, lang, refs=None):
    """3-column Exercise/Nutrition/Lifestyle card (PNOE-style), with per-pillar
    PubMed references rendered as small links under each column when available."""
    if not recs:
        return
    if lang == "el":
        tx = {
            "title":   "📍 ΕΞΑΤΟΜΙΚΕΥΜΕΝΕΣ ΣΥΣΤΑΣΕΙΣ",
            "ex_lbl":  "ΦΥΣΙΚΗ ΔΡΑΣΤΗΡΙΟΤΗΤΑ",
            "nu_lbl":  "ΔΙΑΤΡΟΦΗ",
            "li_lbl":  "ΤΡΟΠΟΣ ΖΩΗΣ",
            "refs":    "Οδηγίες & μετα-αναλύσεις",
        }
    else:
        tx = {
            "title":   "📍 PERSONALISED RECOMMENDATIONS",
            "ex_lbl":  "EXERCISE",
            "nu_lbl":  "NUTRITION",
            "li_lbl":  "LIFESTYLE",
            "refs":    "Guidelines & meta-analyses",
        }
    import html as _html_r, re as _re_rec
    # Collapse internal whitespace BEFORE escaping. Newlines in recs content
    # break Streamlit's markdown HTML mode → raw </div> tags leak as text
    # (the bug visible in the user's screenshot). Recs are short prose, so a
    # single-line collapse is safe and preserves readability.
    def _flat(t): return _re_rec.sub(r"\s+", " ", (t or "—").strip()) or "—"
    ex = _html_r.escape(_flat(recs.get("exercise")))
    nu = _html_r.escape(_flat(recs.get("nutrition")))
    li = _html_r.escape(_flat(recs.get("lifestyle")))

    def _refs_html(pillar_key):
        items = (refs or {}).get(pillar_key) or []
        if not items:
            return ""
        lis = "".join(
            f'<li><a href="{_html_r.escape(r.get("url",""))}" target="_blank" '
            f'style="color:#1E40AF;text-decoration:none">'
            f'{_html_r.escape((r.get("title","—") or "")[:120])}'
            f'</a><span style="color:#9CA3AF"> · {_html_r.escape(r.get("journal","") or "")}'
            f'{(" " + _html_r.escape(r.get("date","")[:4])) if r.get("date") else ""}</span></li>'
            for r in items
        )
        return (
            f'<div class="pnoe-refs">'
            f'<div class="pnoe-refs-lbl">📚 {tx["refs"]}</div>'
            f'<ul>{lis}</ul>'
            f'</div>'
        )

    st.markdown(f"""
<style>
.pnoe-recs {{
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  padding: 24px 26px;
  margin: 18px 0;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.pnoe-recs-title {{
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
  color: #6B7280; text-transform: uppercase;
  border-bottom: 2px solid #E5E7EB;
  padding-bottom: 10px; margin-bottom: 18px;
}}
.pnoe-recs-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 14px;
}}
.pnoe-recs-col {{
  border-radius: 12px;
  padding: 16px 16px 18px;
  border: 1px solid;
  position: relative;
}}
.pnoe-recs-col::before {{
  content: ""; position: absolute; left: 16px; top: 16px;
  width: 4px; height: 18px; border-radius: 2px;
}}
.pnoe-recs-col.exercise  {{ background: #EFF6FF; border-color: #BFDBFE; }}
.pnoe-recs-col.exercise::before  {{ background: #3B82F6; }}
.pnoe-recs-col.nutrition {{ background: #ECFDF5; border-color: #A7F3D0; }}
.pnoe-recs-col.nutrition::before {{ background: #10B981; }}
.pnoe-recs-col.lifestyle {{ background: #FEF3F2; border-color: #FECDD3; }}
.pnoe-recs-col.lifestyle::before {{ background: #EF4444; }}
.pnoe-recs-head {{
  display: flex; align-items: center; gap: 9px;
  margin-bottom: 10px; padding-left: 14px;
}}
.pnoe-recs-icon {{ font-size: 17px; line-height: 1; }}
.pnoe-recs-label {{
  font-size: 11px; font-weight: 800;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: #1F2937;
}}
.pnoe-recs-body {{
  font-size: 13px; color: #374151; line-height: 1.6;
}}
.pnoe-refs {{
  margin-top: 12px; padding-top: 10px;
  border-top: 1px dashed rgba(0,0,0,0.10);
}}
.pnoe-refs-lbl {{
  font-size: 10px; font-weight: 700; letter-spacing: 0.10em;
  color: #6B7280; text-transform: uppercase; margin-bottom: 5px;
}}
.pnoe-refs ul {{
  list-style: none; padding: 0; margin: 0;
}}
.pnoe-refs li {{
  font-size: 11.5px; line-height: 1.45; margin-bottom: 5px;
  color: #374151;
}}
.pnoe-refs a:hover {{ text-decoration: underline !important; }}
@media (max-width: 768px) {{
  .pnoe-recs-grid {{ grid-template-columns: 1fr; gap: 11px; }}
  .pnoe-recs {{ padding: 20px 18px; }}
}}
</style>
<div class="pnoe-recs">
<div class="pnoe-recs-title">{tx['title']}</div>
<div class="pnoe-recs-grid">
<div class="pnoe-recs-col exercise">
<div class="pnoe-recs-head"><span class="pnoe-recs-icon">🏃</span><span class="pnoe-recs-label">{tx['ex_lbl']}</span></div>
<div class="pnoe-recs-body">{ex}</div>
{_refs_html("exercise")}
</div>
<div class="pnoe-recs-col nutrition">
<div class="pnoe-recs-head"><span class="pnoe-recs-icon">🥗</span><span class="pnoe-recs-label">{tx['nu_lbl']}</span></div>
<div class="pnoe-recs-body">{nu}</div>
{_refs_html("nutrition")}
</div>
<div class="pnoe-recs-col lifestyle">
<div class="pnoe-recs-head"><span class="pnoe-recs-icon">🌿</span><span class="pnoe-recs-label">{tx['li_lbl']}</span></div>
<div class="pnoe-recs-body">{li}</div>
{_refs_html("lifestyle")}
</div>
</div>
</div>
""", unsafe_allow_html=True)


def _compute_health_pillars(profile, vitals, status_map, report_text, lang):
    """Return (pillars_list, overall_score). Each pillar has a score 0-100 or
    None when no data was available. The overall is the mean of pillars that
    had data — never invented for missing inputs."""
    age = profile.get("age", 0) or 0
    # Accent-stripped so Greek history like "Υπέρταση" matches the "υπερτασ" pattern
    history = _strip_accents((profile.get("history") or "").lower())
    rep_low = _strip_accents((report_text or "").lower())

    def _ss(k):
        s = status_map.get(k)
        if s == "green":  return 100
        if s == "yellow": return 60
        if s == "red":    return 25
        return None

    # 1) 🫀 Cardiovascular: HR + BP + age/hypertension penalty
    cardio, cfact = [], []
    if _ss("hr") is not None:
        cardio.append(_ss("hr"));  cfact.append(f"HR {vitals.get('hr')}")
    if _ss("bp") is not None:
        cardio.append(_ss("bp"));  cfact.append(f"BP {vitals.get('bp_sys')}/{vitals.get('bp_dia')}")
    if cardio:
        sc = sum(cardio) / len(cardio)
        if age >= 75: sc = max(20, sc - 12); cfact.append("ηλικία ≥75" if lang=="el" else "age ≥75")
        elif age >= 65: sc = max(25, sc - 6)
        if any(w in history for w in ("υπερτασ","hypertens")):
            sc = max(20, sc - 8); cfact.append("ιστ. υπέρτασης" if lang=="el" else "hypertension hx")
        c_score = int(round(sc))
    else:
        c_score = None

    # 2) 🫁 Respiratory: SpO2 + BR + smoking/asthma flags
    resp, rfact = [], []
    if _ss("spo2") is not None:
        resp.append(_ss("spo2")); rfact.append(f"SpO₂ {vitals.get('spo2')}%")
    if _ss("br") is not None:
        resp.append(_ss("br"));   rfact.append(f"BR {vitals.get('br')}/min")
    if resp:
        sc = sum(resp) / len(resp)
        if any(w in history for w in ("καπν","smok","τσιγαρ")):
            sc = max(20, sc - 15); rfact.append("κάπνισμα" if lang=="el" else "smoking")
        if any(w in history for w in ("ασθμ","asthm","copd","χαπ")):
            sc = max(20, sc - 8);  rfact.append("ασθματικός" if lang=="el" else "asthma")
        r_score = int(round(sc))
    else:
        r_score = None

    # 3) ⚖️ Metabolic: BMI + temp + diabetes flag
    meta, mfact = [], []
    if _ss("bmi") is not None:
        meta.append(_ss("bmi"));  mfact.append(f"ΔΜΣ {vitals.get('bmi')}" if lang=="el" else f"BMI {vitals.get('bmi')}")
    if _ss("temp") is not None:
        meta.append(_ss("temp")); mfact.append(f"T {vitals.get('temp')}°C")
    if meta:
        sc = sum(meta) / len(meta)
        if any(w in history for w in ("διαβητ","diabet","τ2","t2")):
            sc = max(20, sc - 12); mfact.append("διαβήτης" if lang=="el" else "diabetes")
        m_score = int(round(sc))
    else:
        m_score = None

    # 4) 🩺 Symptom burden: from report content (red flags + severity terms)
    sb_score = 100
    sb_fact = []
    urgent = [_strip_accents(w) for w in
              ["επείγον","emergency","stroke","εγκεφαλικ","heart attack","έμφραγμα",
               "anaphylax","αναφυλαξ","unconscious","αναίσθητ","166","112"]]
    if any(w in rep_low for w in urgent):
        sb_score -= 50
        sb_fact.append("κόκκινες σημαίες" if lang=="el" else "red flags")
    severity = [_strip_accents(w) for w in
                ("σοβαρ","οξύς","έντον","severe","intense","acute")]
    if any(w in rep_low for w in severity):
        sb_score -= 12
        sb_fact.append("έντονα συμπτώματα" if lang=="el" else "intense symptoms")
    # Many differentials = more diagnostic uncertainty
    diff_rows = rep_low.count("|")
    if diff_rows >= 16:  # ≥4 rows in the markdown table
        sb_score -= 8
        sb_fact.append("πολλαπλές διαφορικές" if lang=="el" else "multiple differentials")
    sb_score = max(20, sb_score)
    if not sb_fact:
        sb_fact.append("ήπιο προφίλ" if lang=="el" else "mild profile")

    pillars = [
        {"key":"cardio","icon":"🫀",
         "label_el":"Καρδιαγγειακή","label_en":"Cardiovascular",
         "score":c_score,"factors":cfact,"available":c_score is not None},
        {"key":"resp","icon":"🫁",
         "label_el":"Αναπνευστική","label_en":"Respiratory",
         "score":r_score,"factors":rfact,"available":r_score is not None},
        {"key":"meta","icon":"⚖️",
         "label_el":"Μεταβολική","label_en":"Metabolic",
         "score":m_score,"factors":mfact,"available":m_score is not None},
        {"key":"symp","icon":"🩺",
         "label_el":"Συμπτωματικό φορτίο","label_en":"Symptom burden",
         "score":sb_score,"factors":sb_fact,"available":True},
    ]
    avail = [p for p in pillars if p["available"]]
    overall = int(round(sum(p["score"] for p in avail) / len(avail))) if avail else None
    return pillars, overall


def _grade_label(score, lang):
    """Map a 0-100 score to the PNOE 5-level grade label."""
    if score is None:
        return ("Δεν υπάρχουν δεδομένα", "#9CA3AF") if lang=="el" else ("No data", "#9CA3AF")
    if score >= 80: return (("Άριστο" if lang=="el" else "Excellent"), "#059669")
    if score >= 60: return (("Καλό"   if lang=="el" else "Good"),      "#10B981")
    if score >= 40: return (("Μέτριο" if lang=="el" else "Neutral"),   "#3B82F6")
    if score >= 20: return (("Χαμηλό" if lang=="el" else "Limited"),   "#F97316")
    return            (("Πολύ χαμηλό" if lang=="el" else "Severe limit."), "#DC2626")


def _pillar_scale_html(score):
    """A clean 5-segment scale (PNOE-style) for a pillar score, on white bg."""
    if score is None:
        return '<div style="height:10px;background:#F3F4F6;border-radius:5px;margin-top:6px"></div>'
    seg = max(0, min(4, int(score) // 20))
    colors = ["#DC2626","#F97316","#3B82F6","#10B981","#059669"]
    out = '<div style="display:flex;gap:4px;margin-top:6px">'
    for i in range(5):
        bg = colors[i] if i <= seg else "#E5E7EB"
        marker = "box-shadow:0 0 0 2px white inset" if i == seg else ""
        out += f'<div style="flex:1;height:10px;background:{bg};border-radius:5px;{marker}"></div>'
    out += '</div>'
    return out


def _render_health_pillars(profile, vitals, status_map, report_text, lang):
    """4-Pillar Health Profile card — replaces the placeholder wellness score
    with a transparent, factor-explained breakdown (PNOE 'Overview' inspired).
    Only shown when at least ONE measurement-based pillar (cardio/resp/meta)
    has data — symptom burden alone is not a 'health profile'."""
    pillars, overall = _compute_health_pillars(profile, vitals, status_map, report_text, lang)
    # Require objective measurements — don't fabricate a "wellness score" from
    # symptom-burden alone. If no vitals were taken, this card stays hidden.
    has_measurements = any(p["available"] for p in pillars if p["key"] in ("cardio","resp","meta"))
    if not has_measurements:
        return
    if lang == "el":
        title    = "📊 ΠΡΟΦΙΛ ΥΓΕΙΑΣ"
        ov_lbl   = "Συνολικό σκορ"
        no_data  = "δεν μετρήθηκε"
        method   = ("Υπολογίζεται από ζωτικά + ιστορικό + ευρήματα εκτίμησης. "
                    "Δεν αντικαθιστά εργαστηριακή μέτρηση.")
        factors_lbl = "Παράγοντες"
    else:
        title    = "📊 HEALTH PROFILE"
        ov_lbl   = "Overall score"
        no_data  = "not measured"
        method   = ("Computed from vitals + history + assessment findings. "
                    "Not a substitute for lab measurements.")
        factors_lbl = "Factors"
    ov_grade, ov_color = _grade_label(overall, lang)
    overall_disp = f"{overall}" if overall is not None else "—"

    # Build pillar rows
    rows_html = ""
    for p in pillars:
        label = p["label_el"] if lang == "el" else p["label_en"]
        grade, gcolor = _grade_label(p["score"], lang)
        score_disp = f"{p['score']}" if p["score"] is not None else "—"
        factors_disp = (" · ".join(p["factors"][:3])) if p["factors"] else no_data
        opacity = "1" if p["available"] else "0.55"
        rows_html += (
            f'<div style="padding:12px 0;border-top:1px solid #F3F4F6;opacity:{opacity}">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;gap:12px">'
            f'<div style="display:flex;align-items:center;gap:10px;min-width:0;flex:1">'
            f'<span style="font-size:20px;flex-shrink:0">{p["icon"]}</span>'
            f'<span style="font-size:13.5px;font-weight:700;color:#1F2937">{label}</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:10px;flex-shrink:0">'
            f'<span style="font-size:18px;font-weight:800;color:{gcolor};font-variant-numeric:tabular-nums">{score_disp}<span style="font-size:11px;color:#9CA3AF;font-weight:600">%</span></span>'
            f'<span style="background:{gcolor}15;color:{gcolor};font-size:10.5px;font-weight:700;padding:3px 9px;border-radius:99px;letter-spacing:0.04em;text-transform:uppercase">{grade}</span>'
            f'</div>'
            f'</div>'
            f'{_pillar_scale_html(p["score"])}'
            f'<div style="font-size:11px;color:#6B7280;margin-top:6px;line-height:1.5">'
            f'<span style="font-weight:700;letter-spacing:0.08em;text-transform:uppercase">{factors_lbl}:</span> {factors_disp}'
            f'</div>'
            f'</div>'
        )
    st.markdown(f"""
<style>
.hp-card {{
  background: white; border: 1px solid #E5E7EB; border-radius: 14px;
  padding: 22px 24px; margin: 18px 0;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.hp-title {{
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
  color: #6B7280; text-transform: uppercase;
  border-bottom: 2px solid #E5E7EB; padding-bottom: 10px; margin-bottom: 14px;
}}
.hp-overall {{
  display: flex; align-items: center; gap: 16px;
  background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
  border-radius: 12px; padding: 14px 18px; margin-bottom: 4px;
}}
.hp-overall .ov-num {{
  font-size: 38px; font-weight: 800; line-height: 1;
  color: {ov_color}; font-variant-numeric: tabular-nums;
}}
.hp-overall .ov-meta {{ flex: 1; min-width: 0; }}
.hp-overall .ov-lbl {{
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.12em;
  color: #6B7280; text-transform: uppercase;
}}
.hp-overall .ov-grade {{
  font-size: 16px; font-weight: 700; color: {ov_color}; margin-top: 2px;
}}
.hp-method {{
  font-size: 10.5px; color: #9CA3AF; margin-top: 12px;
  padding-top: 10px; border-top: 1px dashed #E5E7EB; line-height: 1.5;
}}
</style>
<div class="hp-card">
<div class="hp-title">{title}</div>
<div class="hp-overall">
<div class="ov-num">{overall_disp}<span style="font-size:18px;color:#9CA3AF;font-weight:600">%</span></div>
<div class="ov-meta"><div class="ov-lbl">{ov_lbl}</div><div class="ov-grade">{ov_grade}</div></div>
</div>
{rows_html}
<div class="hp-method">ℹ️ {method}</div>
</div>
""", unsafe_allow_html=True)


def render_emergency_resources(lang):
    """'Πού να απευθυνθώ' card with:
      - Emergency numbers (166 EKAB, 112 EU, 1066 fire dept) as click-to-call
      - Vrisko.gr links for ΕΟΠΥΥ doctors, on-duty hospitals, on-duty pharmacies
        (these are public Greek directories — same ones nextdeal.gr / vrisko link to)
      - Google Maps quick-search buttons for nearby facilities
    Rendered on the report screen so the user knows their next step after triage."""
    if lang == "el":
        tx = {
            "title":       "📍 ΠΟΥ ΝΑ ΑΠΕΥΘΥΝΘΩ",
            "subtitle":    "Επόμενα βήματα — εφημερεύοντα, ΕΟΠΥΥ γιατροί, φαρμακεία",
            "emerg_title": "🚨 ΣΕ ΕΠΕΙΓΟΥΣΑ ΑΝΑΓΚΗ",
            "ekab":        "ΕΚΑΒ Ασθενοφόρο",
            "eu_112":      "Ευρωπαϊκή Γραμμή Έκτακτης Ανάγκης",
            "pfy":         "Πρωτοβάθμια Φροντίδα Υγείας (1135)",
            "find_doc":    "🩺 Γιατρός ΕΟΠΥΥ",
            "find_doc_sub":"Συμβεβλημένοι γιατροί",
            "find_hosp":   "🚑 Εφημερεύοντα νοσοκομεία",
            "find_hosp_sub":"Σήμερα",
            "find_pharm":  "💊 Εφημερεύοντα φαρμακεία",
            "find_pharm_sub":"Διανυκτερεύοντα",
            "maps_title":  "Άνοιξε στο Google Maps",
            "maps_hosp":   "Νοσοκομείο κοντά μου",
            "maps_doc":    "Ιατρείο κοντά μου",
            "maps_pharm":  "Φαρμακείο κοντά μου",
        }
        # Greek Google Maps queries (browser geolocates from device)
        maps_q = {
            "hosp":  "νοσοκομείο",
            "doc":   "ιατρείο",
            "pharm": "φαρμακείο",
        }
    else:
        tx = {
            "title":       "📍 NEXT STEPS — WHERE TO GO",
            "subtitle":    "On-duty facilities, EOPYY doctors, pharmacies",
            "emerg_title": "🚨 IN AN EMERGENCY",
            "ekab":        "EKAB Ambulance",
            "eu_112":      "European Emergency Line",
            "pfy":         "Primary Care Helpline (1135)",
            "find_doc":    "🩺 EOPYY Doctor",
            "find_doc_sub":"Affiliated physicians",
            "find_hosp":   "🚑 On-duty hospitals",
            "find_hosp_sub":"Today",
            "find_pharm":  "💊 On-duty pharmacies",
            "find_pharm_sub":"Night/weekend",
            "maps_title":  "Open in Google Maps",
            "maps_hosp":   "Hospital near me",
            "maps_doc":    "Doctor's office near me",
            "maps_pharm":  "Pharmacy near me",
        }
        maps_q = {
            "hosp":  "hospital",
            "doc":   "doctor",
            "pharm": "pharmacy",
        }
    import urllib.parse as _up
    def _maps(q):
        return f"https://www.google.com/maps/search/?api=1&query={_up.quote(q)}"
    # Official Ministry of Health page for Athens hospital duty schedule.
    # vrisko.gr was a 3rd-party aggregator; moh.gov.gr is the authoritative source.
    URL_DOC   = "https://www.vrisko.gr/dir/giatroi-eopyy"
    URL_HOSP  = "https://www.moh.gov.gr/articles/citizen/efhmeries-nosokomeiwn/68-efhmeries-nosokomeiwn-attikhs"
    URL_PHARM = "https://www.vrisko.gr/efimeries-farmakeion"
    st.markdown(f"""
<style>
.er-card {{
  background: white; border: 1px solid #E5E7EB; border-radius: 14px;
  padding: 22px 24px; margin: 18px 0;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.er-title {{
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
  color: #6B7280; text-transform: uppercase;
  border-bottom: 2px solid #E5E7EB; padding-bottom: 10px; margin-bottom: 4px;
}}
.er-subtitle {{
  font-size: 12px; color: #9CA3AF; margin-bottom: 16px;
}}
.er-emerg {{
  background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
  border: 1px solid #FCA5A5; border-radius: 12px;
  padding: 14px 16px; margin-bottom: 16px;
}}
.er-emerg-title {{
  font-size: 10.5px; font-weight: 800; color: #991B1B;
  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px;
}}
.er-emerg-row {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 0; border-top: 1px dashed rgba(220,38,38,0.20);
  gap: 12px;
}}
.er-emerg-row:first-of-type {{ border-top: none; padding-top: 4px; }}
.er-emerg-label {{ font-size: 13px; color: #7F1D1D; font-weight: 600; flex: 1; min-width: 0; }}
.er-call-btn {{
  background: #DC2626; color: white; padding: 7px 16px; border-radius: 8px;
  font-weight: 700; font-size: 14px; text-decoration: none;
  font-variant-numeric: tabular-nums; white-space: nowrap; flex-shrink: 0;
}}
.er-call-btn:hover {{ background: #B91C1C; color: white; text-decoration: none; }}

.er-grid {{
  display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;
}}
.er-link {{
  display: block;
  background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px;
  padding: 14px; text-decoration: none; color: inherit;
  transition: all 0.15s;
}}
.er-link:hover {{
  background: white; border-color: #2D3FE7; text-decoration: none; color: inherit;
  transform: translateY(-1px); box-shadow: 0 2px 6px rgba(45,63,231,0.10);
}}
.er-link-title {{ font-size: 13.5px; font-weight: 700; color: #1F2937; margin-bottom: 3px; }}
.er-link-sub {{ font-size: 11px; color: #6B7280; }}

.er-maps {{
  margin-top: 16px; padding-top: 14px; border-top: 1px dashed #E5E7EB;
}}
.er-maps-title {{
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.12em;
  color: #6B7280; text-transform: uppercase; margin-bottom: 10px;
}}
.er-maps-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.er-maps-btn {{
  flex: 1; min-width: 140px;
  display: inline-block; padding: 9px 14px;
  background: white; border: 1px solid #C7D2FE; border-radius: 8px;
  color: #2D3FE7; font-size: 12.5px; font-weight: 600;
  text-decoration: none; text-align: center;
}}
.er-maps-btn:hover {{ background: #EFF6FF; color: #2D3FE7; text-decoration: none; }}

@media (max-width: 640px) {{
  .er-grid {{ grid-template-columns: 1fr; }}
  .er-emerg-row {{ flex-wrap: wrap; }}
  .er-maps-btn {{ min-width: 100%; }}
}}
</style>
<div class="er-card">
  <div class="er-title">{tx['title']}</div>
  <div class="er-subtitle">{tx['subtitle']}</div>

  <div class="er-emerg">
    <div class="er-emerg-title">{tx['emerg_title']}</div>
    <div class="er-emerg-row">
      <span class="er-emerg-label">{tx['ekab']}</span>
      <a class="er-call-btn" href="tel:166">📞 166</a>
    </div>
    <div class="er-emerg-row">
      <span class="er-emerg-label">{tx['eu_112']}</span>
      <a class="er-call-btn" href="tel:112">📞 112</a>
    </div>
    <div class="er-emerg-row">
      <span class="er-emerg-label">{tx['pfy']}</span>
      <a class="er-call-btn" href="tel:1135">📞 1135</a>
    </div>
  </div>

  <div class="er-grid">
    <a class="er-link" href="{URL_DOC}" target="_blank" rel="noopener">
      <div class="er-link-title">{tx['find_doc']}</div>
      <div class="er-link-sub">{tx['find_doc_sub']} ↗</div>
    </a>
    <a class="er-link" href="{URL_HOSP}" target="_blank" rel="noopener">
      <div class="er-link-title">{tx['find_hosp']}</div>
      <div class="er-link-sub">{tx['find_hosp_sub']} ↗</div>
    </a>
    <a class="er-link" href="{URL_PHARM}" target="_blank" rel="noopener">
      <div class="er-link-title">{tx['find_pharm']}</div>
      <div class="er-link-sub">{tx['find_pharm_sub']} ↗</div>
    </a>
  </div>

  <div class="er-maps">
    <div class="er-maps-title">📍 {tx['maps_title']}</div>
    <div class="er-maps-row">
      <a class="er-maps-btn" href="{_maps(maps_q['hosp'])}" target="_blank" rel="noopener">🚑 {tx['maps_hosp']}</a>
      <a class="er-maps-btn" href="{_maps(maps_q['doc'])}" target="_blank" rel="noopener">🩺 {tx['maps_doc']}</a>
      <a class="er-maps-btn" href="{_maps(maps_q['pharm'])}" target="_blank" rel="noopener">💊 {tx['maps_pharm']}</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_report():
    render_stepper("report")
    p=st.session_state.profile; lang=st.session_state.lang
    nm = p.get("name","")
    age = p.get("age")
    sub_el = f"για {nm}" + (f", {age} ετών" if age else "")
    sub_en = f"for {nm}" + (f", {age} years old" if age else "")
    render_doc_header(
        "Η εκτίμηση σου", "Your assessment",
        icon="📋",
        sub_el=(sub_el if nm else "Κλινική εκτίμηση με τεκμηρίωση"),
        sub_en=(sub_en if nm else "Clinical assessment with references"),
    )
    render_vitals_summary()
    if not st.session_state.report:
        conversation="\n".join(f"{'Patient' if m['role']=='user' else 'Asklepios'}: {m['content']}" for m in st.session_state.triage_chat)
        vitals_text="\n".join(f"- {k}: {v}" for k,v in st.session_state.vitals.items()) if st.session_state.vitals else "Not provided"
        vitals_analysis=st.session_state.vitals_analysis or "Not available"
        last_user=next((m["content"] for m in reversed(st.session_state.triage_chat) if m["role"]=="user"),"")
        search_query=last_user[:80]+" diagnosis management" if last_user else "symptom assessment management"
        with st.spinner("🔬 PubMed..." if lang=="el" else "🔬 Searching PubMed..."):
            refs=pubmed_search(search_query,n=3); st.session_state.report_pubmed=refs
        pubmed_ctx="\n".join(f"- {a['title']} ({a['journal']}, {a['date']}) {a['url']}" for a in refs) if refs else "None found."
        pp=p.get
        # Special-population flags that the report MUST respect
        _rep_flags = []
        if pp("pregnancy"):
            _rep_flags.append("PREGNANT — exclude Category D/X drugs; flag teratogenic risks; recommend OB-GYN consultation")
        if pp("for_whom") == "other":
            _rep_flags.append("Caregiver-mode: report addresses the caregiver; use third person for the patient")
        _age_r = pp("age", 0) or 0
        if _age_r < 18:
            _rep_flags.append(f"PEDIATRIC ({_age_r} yo) — use pediatric vital ranges, dosing by weight, age-appropriate red flags")
        _rep_flags_str = ("\nSPECIAL CONSIDERATIONS: " + " | ".join(_rep_flags)) if _rep_flags else ""
        report_prompt=f"""Generate a concise clinical assessment for:
PATIENT: {pp('name')}, {pp('age')}yo {pp('sex')}{_rep_flags_str}
HISTORY: {pp('history','none')} | ALLERGIES: {pp('allergies','none')} | MEDS: {pp('meds_raw','none')}
VITALS: {vitals_text}
VITALS ANALYSIS: {vitals_analysis}
CONSULTATION: {conversation}
PUBMED: {pubmed_ctx}
Write these sections IN THIS ORDER, using EXACTLY these headers as written (do not abbreviate or drop letters):
{"1. ΚΥΡΙΟ ΠΑΡΑΠΟΝΟ  2. ΙΣΤΟΡΙΚΟ  3. ΕΚΤΙΜΗΣΗ (Πρωτεύουσα Διάγνωση + Διαφορικές Διαγνώσεις)  4. ΘΕΡΑΠΕΥΤΙΚΟ ΠΛΑΝΟ  5. ΚΟΚΚΙΝΕΣ ΣΗΜΑΙΕΣ  6. ΒΙΒΛΙΟΓΡΑΦΙΑ" if lang=="el" else "1. CHIEF COMPLAINT  2. HISTORY  3. ASSESSMENT (Primary Diagnosis + Differentials)  4. TREATMENT PLAN  5. RED FLAGS  6. REFERENCES"}
For the differentials use a markdown table with EXACTLY 3 columns and these short headers: {"| Διάγνωση | % | Σχόλιο |" if lang=="el" else "| Diagnosis | % | Comment |"} (keep the probability header as just "%", and put values like "~8%"). Keep cell text short.

After section 6 (References), append EXACTLY this delimited block — same format, no extra text inside the delimiters:
<<<RECS
CONDITION: [the primary clinical condition in 2-4 ENGLISH words, MeSH-friendly — e.g. "Hypertension", "Migraine", "Type 2 Diabetes", "Gastroesophageal Reflux", "Anxiety Disorder". Just the noun phrase, no extra text. This is used to fetch matching guideline literature.]
EXERCISE: [2-3 sentences of PERSONALISED exercise advice for this specific patient — based on age, conditions, symptoms. Direct and actionable. {"Σε Ελληνικά." if lang=="el" else "In English."} No generic platitudes.]
NUTRITION: [2-3 sentences of personalised nutrition advice for this patient — specific foods/changes that target the assessed conditions. {"Σε Ελληνικά." if lang=="el" else "In English."}]
LIFESTYLE: [2-3 sentences on sleep, stress, smoking, alcohol — tailored to this case. {"Σε Ελληνικά." if lang=="el" else "In English."}]
RECS>>>

Language: {"Greek" if lang=="el" else "English"}. Be direct. End with a one-line AI disclaimer."""
        with st.spinner("Δημιουργία αναφοράς..." if lang=="el" else "Generating report..."):
            result=claude([{"role":"user","content":report_prompt}],system=kira_system(),max_tokens=4000,timeout=120)
            if result.startswith("⚠️"):
                st.error(result)
                if st.button("🔄 Retry"): st.rerun()
                return
            # Parse out the PNOE-style RECS block ONCE on generation. The cleaned
            # report (without delimiters) is what shows on-screen and in exports;
            # the recs dict drives the 3-column visual card.
            _clean, _recs = _extract_recs(result)
            st.session_state.report = _clean
            st.session_state.report_recs = _recs
            # Fetch high-evidence PubMed refs PER PILLAR (Exercise/Nutrition/Lifestyle)
            # using MeSH + Practice-Guideline/Systematic-Review/Meta-Analysis filters.
            # Runs the 3 queries in parallel to keep total latency reasonable.
            _condition = (_recs or {}).get("condition", "").strip()
            if _condition:
                from concurrent.futures import ThreadPoolExecutor as _TPE
                with st.spinner("📚 " + ("Αναζήτηση οδηγιών ανά πυλώνα..." if lang=="el"
                                          else "Searching guideline-level evidence per pillar...")):
                    with _TPE(max_workers=3) as _ex:
                        _futs = {p: _ex.submit(pubmed_pillar_search, _condition, p, 2)
                                 for p in ("exercise","nutrition","lifestyle")}
                        st.session_state.report_recs_refs = {p: f.result() for p,f in _futs.items()}
            else:
                st.session_state.report_recs_refs = {}
    if not st.session_state.report:
        if st.button("🔄 "+("Δοκιμή ξανά" if lang=="el" else "Retry"),type="primary"): st.rerun()
        return
    # ── Doctor's-report style: PATIENT INFO doc-card + CLINICAL ASSESSMENT header ──
    # Inspired by the medical-report template (USGH-style): blue/red accent boxes
    # for allergies + medications side-by-side, with medical history above. The
    # actual Claude assessment renders below with restyled markdown section headers.
    history_raw = (p.get("history") or "").strip()
    history = history_raw if history_raw else "—"
    allergies_raw = (p.get("allergies") or "").strip()
    allergies = allergies_raw if allergies_raw else "—"
    meds_raw = (p.get("meds_raw") or "").strip()
    meds_list = [m.strip() for m in meds_raw.split(",") if m.strip()]
    meds_html = "<br>".join(f"• {m}" for m in meds_list) if meds_list else "—"
    if lang == "el":
        TX = {
            "patient_info": "Στοιχεία Ασθενή",
            "history_lbl": "Ιατρικό Ιστορικό",
            "allergies_lbl": "Αλλεργίες",
            "meds_lbl": "Φάρμακα",
            "assessment_title": "Κλινική Αξιολόγηση",
        }
    else:
        TX = {
            "patient_info": "Patient Information",
            "history_lbl": "Medical History",
            "allergies_lbl": "Allergies",
            "meds_lbl": "Medications",
            "assessment_title": "Clinical Assessment",
        }
    st.markdown(f"""
<style>
.report-card {{
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  padding: 26px 28px;
  margin: 6px 0 16px;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.report-card-title {{
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
  color: #6B7280; text-transform: uppercase;
  border-bottom: 2px solid #E5E7EB;
  padding-bottom: 10px; margin-bottom: 18px;
  display: flex; align-items: center; gap: 8px;
}}
.history-block {{
  background: #F9FAFB; border: 1px solid #E5E7EB;
  border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;
  font-size: 13.5px; color: #374151; line-height: 1.55;
}}
.history-block .hb-lbl {{
  font-size: 10.5px; font-weight: 700; color: #6B7280;
  text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 6px;
}}
.aller-meds {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
}}
.aller-box, .meds-box {{
  border-radius: 12px; padding: 16px 18px;
  font-size: 13.5px; line-height: 1.55;
}}
.aller-box {{ background: #FEF2F2; border: 1px solid #FECACA; }}
.meds-box  {{ background: #DBEAFE; border: 1px solid #93C5FD; }}
.aller-box .am-lbl, .meds-box .am-lbl {{
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; margin-bottom: 8px;
}}
.aller-box .am-lbl {{ color: #991B1B; }}
.meds-box  .am-lbl {{ color: #1E40AF; }}
.aller-box .am-val {{ color: #7F1D1D; font-weight: 500; }}
.meds-box  .am-val {{ color: #1E3A8A; font-weight: 500; }}

/* Clinical Assessment section header (separates patient info from Claude content) */
.assessment-section-header {{
  background: linear-gradient(135deg, #EFF6FF 0%, #E0E7FF 100%);
  border: 1px solid #C7D2FE;
  border-left: 4px solid #2D3FE7;
  border-radius: 12px;
  padding: 16px 22px;
  margin: 16px 0 14px;
  display: flex; align-items: center; gap: 12px;
  font-family: 'Inter', system-ui, sans-serif;
}}
.assessment-section-header .ash-icon {{
  font-size: 24px;
}}
.assessment-section-header .ash-title {{
  font-size: 13px; font-weight: 800; letter-spacing: 0.14em;
  color: #1E3A8A; text-transform: uppercase;
}}

/* Style markdown section headers inside the Claude report so each section
 * ("ΚΥΡΙΟ ΠΑΡΑΠΟΝΟ", "ΙΣΤΟΡΙΚΟ", "ΕΚΤΙΜΗΣΗ" ...) reads like a medical report block */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2 {{
  color: #2D3FE7 !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.10em !important;
  border-bottom: 1.5px solid #DBEAFE !important;
  padding-bottom: 6px !important;
  margin: 22px 0 12px !important;
}}
[data-testid="stMarkdownContainer"] h3 {{
  color: #4338CA !important;
  font-size: 13.5px !important;
  font-weight: 700 !important;
  margin: 16px 0 8px !important;
}}
@media (max-width: 640px) {{
  .report-card {{ padding: 20px 18px; }}
  .aller-meds {{ grid-template-columns: 1fr; gap: 10px; }}
  .assessment-section-header {{ padding: 12px 16px; }}
}}
</style>
<div class="report-card">
  <div class="report-card-title">📑 {TX['patient_info']}</div>
  <div class="history-block">
    <div class="hb-lbl">📋 {TX['history_lbl']}</div>
    <div>{history}</div>
  </div>
  <div class="aller-meds">
    <div class="aller-box">
      <div class="am-lbl">🔴 {TX['allergies_lbl']}</div>
      <div class="am-val">{allergies}</div>
    </div>
    <div class="meds-box">
      <div class="am-lbl">💊 {TX['meds_lbl']}</div>
      <div class="am-val">{meds_html}</div>
    </div>
  </div>
</div>
<div class="assessment-section-header">
  <span class="ash-icon">📋</span>
  <span class="ash-title">{TX['assessment_title']}</span>
</div>
""", unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    # Photo findings card — if the user uploaded any photos during triage, the
    # AI vision analyses become visible evidence in the final report. Each card
    # shows scan type + Claude's interpretation. (Florence-2 description is
    # already woven into the analysis so we don't duplicate it.)
    _pfs = st.session_state.get("photo_findings") or []
    if isinstance(_pfs, list) and _pfs:
        _pf_title = ("📷 ΕΥΡΗΜΑΤΑ ΑΠΟ ΦΩΤΟΓΡΑΦΙΕΣ" if lang=="el"
                     else "📷 PHOTO FINDINGS")
        _pf_count = len(_pfs)
        import html as _html_pf, re as _re_pf
        def _flat_pf(t): return _re_pf.sub(r"\s+", " ", (t or "").strip())
        _cards_html = ""
        for i, pf in enumerate(_pfs, 1):
            _label = _html_pf.escape(pf.get("scan_label","—"))
            _analysis = _flat_pf(pf.get("analysis",""))
            # Keep markdown bold/headers in the analysis readable inside the card —
            # convert ** ** → <strong>, leave the rest as text after escape.
            _analysis = _html_pf.escape(_analysis)
            _analysis = _re_pf.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _analysis)
            _cards_html += (
                f'<div class="pf-item">'
                f'<div class="pf-head"><span class="pf-num">{i}</span><span class="pf-label">{_label}</span></div>'
                f'<div class="pf-body">{_analysis}</div>'
                f'</div>'
            )
        st.markdown(
            f'<style>'
            f'.pf-card{{background:white;border:1px solid #E5E7EB;border-radius:14px;padding:22px 24px;margin:18px 0;font-family:Inter,system-ui,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.04)}}'
            f'.pf-title{{font-size:11px;font-weight:700;letter-spacing:0.14em;color:#6B7280;text-transform:uppercase;border-bottom:2px solid #E5E7EB;padding-bottom:10px;margin-bottom:14px}}'
            f'.pf-item{{padding:14px 0;border-bottom:1px solid #F3F4F6}}'
            f'.pf-item:last-child{{border-bottom:none;padding-bottom:0}}'
            f'.pf-head{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}'
            f'.pf-num{{background:#DBEAFE;color:#1E40AF;font-size:11px;font-weight:700;width:22px;height:22px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center}}'
            f'.pf-label{{font-size:13.5px;font-weight:700;color:#111827}}'
            f'.pf-body{{font-size:13px;color:#374151;line-height:1.6}}'
            f'.pf-body strong{{color:#1F2937}}'
            f'</style>'
            f'<div class="pf-card"><div class="pf-title">{_pf_title} · {_pf_count}</div>{_cards_html}</div>',
            unsafe_allow_html=True,
        )
    # Lab findings card — mirrors the photo card style with a green accent for
    # laboratory data. Same in-memory-only privacy: contents are session-only.
    _lfs = st.session_state.get("lab_findings") or []
    if isinstance(_lfs, list) and _lfs:
        _lf_title = ("🧪 ΕΥΡΗΜΑΤΑ ΕΡΓΑΣΤΗΡΙΑΚΩΝ ΕΞΕΤΑΣΕΩΝ" if lang=="el"
                     else "🧪 LAB FINDINGS")
        _lf_count = len(_lfs)
        import html as _html_lf, re as _re_lf
        def _flat_lf(t): return _re_lf.sub(r"\s+", " ", (t or "").strip())
        _lf_cards = ""
        for i, lf in enumerate(_lfs, 1):
            _fname = _html_lf.escape(lf.get("file_name","—"))
            _an = _html_lf.escape(_flat_lf(lf.get("analysis","")))
            _an = _re_lf.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _an)
            _lf_cards += (
                f'<div class="lf-item">'
                f'<div class="lf-head"><span class="lf-num">{i}</span>'
                f'<span class="lf-label">📄 {_fname}</span></div>'
                f'<div class="lf-body">{_an}</div>'
                f'</div>'
            )
        st.markdown(
            f'<style>'
            f'.lf-card{{background:white;border:1px solid #E5E7EB;border-radius:14px;padding:22px 24px;margin:18px 0;font-family:Inter,system-ui,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.04)}}'
            f'.lf-title{{font-size:11px;font-weight:700;letter-spacing:0.14em;color:#6B7280;text-transform:uppercase;border-bottom:2px solid #E5E7EB;padding-bottom:10px;margin-bottom:14px}}'
            f'.lf-item{{padding:14px 0;border-bottom:1px solid #F3F4F6}}'
            f'.lf-item:last-child{{border-bottom:none;padding-bottom:0}}'
            f'.lf-head{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}'
            f'.lf-num{{background:#D1FAE5;color:#065F46;font-size:11px;font-weight:700;width:22px;height:22px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center}}'
            f'.lf-label{{font-size:13.5px;font-weight:700;color:#111827}}'
            f'.lf-body{{font-size:13px;color:#374151;line-height:1.6}}'
            f'.lf-body strong{{color:#1F2937}}'
            f'</style>'
            f'<div class="lf-card"><div class="lf-title">{_lf_title} · {_lf_count}</div>{_lf_cards}</div>',
            unsafe_allow_html=True,
        )
    # PNOE-style 3-pillar Recommendations card (Exercise / Nutrition / Lifestyle)
    if st.session_state.get("report_recs"):
        _render_recs_card(st.session_state.report_recs, lang,
                          refs=st.session_state.get("report_recs_refs") or {})
    # Where-to-go card: emergency numbers + nearby clinics/pharmacies finder.
    # Placed right after the personalised recs so the user has all the info
    # needed to take the next step.
    render_emergency_resources(lang)
    if st.session_state.report_pubmed:
        with st.expander(f"🔬 {t('pubmed')} ({len(st.session_state.report_pubmed)})"):
            for a in st.session_state.report_pubmed:
                st.markdown(f"**[{a['title']}]({a['url']})**  \n*{a['authors']} — {a['journal']}, {a['date']}*")
    if get_openai_key():
        with st.expander(f"🤖 {t('second_opinion')}"):
            if not st.session_state.report_gpt:
                if st.button(("Λάβε δεύτερη γνώμη GPT-4o" if lang=="el" else "Get GPT-4o second opinion"),
                             type="secondary", key="gpt_get"):
                    with st.spinner("GPT-4o reviewing..."):
                        _gpt_prompt = (
                            f"Patient: {p.get('name')}, {p.get('age')}yo {p.get('sex','')}\n"
                            f"History: {p.get('history','none')} | Allergies: {p.get('allergies','none')} | Meds: {p.get('meds_raw','none')}\n\n"
                            f"Claude clinical assessment:\n{st.session_state.report}\n\n"
                            f"As an independent clinical reviewer: do you AGREE with this assessment? "
                            f"What specific ADDITIONS or CORRECTIONS would you make (differentials missed, "
                            f"treatment refinements, red flags overlooked, drug-interaction concerns)? "
                            f"Be concise — bullet points OK. Respond in {'Greek' if lang=='el' else 'English'}."
                        )
                        st.session_state.report_gpt = gpt4o(prompt=_gpt_prompt, system=kira_system(), max_tokens=900)
                    st.rerun()
            else:
                st.markdown(st.session_state.report_gpt)
                # Integration: if the second opinion adds value, the user can fold
                # it into the main report so it shows up in the on-screen assessment
                # AND in every downstream export (PDF/HTML/TXT/WhatsApp).
                st.divider()
                if st.session_state.get("_gpt_integrated"):
                    st.success("✓ " + ("Ενσωματώθηκε στην τελική εκτίμηση παραπάνω και στα exports."
                                       if lang=="el" else
                                       "Integrated into the final assessment above and in all exports."))
                else:
                    if st.button(("➕ Ενσωμάτωση στην τελική εκτίμηση" if lang=="el"
                                  else "➕ Integrate into final assessment"),
                                 type="primary", use_container_width=True, key="gpt_integrate"):
                        _hdr = "## " + ("ΔΕΥΤΕΡΗ ΓΝΩΜΗ (GPT-4o)" if lang=="el"
                                        else "SECOND OPINION (GPT-4o)")
                        st.session_state.report = (
                            (st.session_state.report or "").rstrip()
                            + "\n\n---\n\n" + _hdr + "\n\n"
                            + (st.session_state.report_gpt or "").strip()
                        )
                        st.session_state["_gpt_integrated"] = True
                        st.rerun()
                    st.caption(("💡 Προσθέτει τη δεύτερη γνώμη ως ξεχωριστή ενότητα στην αναφορά "
                                "και σε όλα τα exports (PDF/TXT/WhatsApp)."
                                if lang=="el" else
                                "💡 Adds the second opinion as a separate section in the report "
                                "and in every export (PDF/TXT/WhatsApp)."))
    if len(st.session_state.medications)>=2:
        with st.expander("💊 RxNorm" + (" — Έλεγχος Αλληλεπιδράσεων" if lang=="el" else " — Interactions")):
            with st.spinner("RxNorm..."): rxr=rxnorm_interactions([m["name"] for m in st.session_state.medications])
            if rxr: st.markdown(rxr)
    # ── 4-Pillar Health Profile (replaces the old placeholder wellness score).
    # Honest, factor-explained — Cardiovascular / Respiratory / Metabolic /
    # Symptom burden — each backed by the vitals + history items that drove it.
    v=st.session_state.vitals
    _status_map = classify_vitals(dict(v), age=st.session_state.profile.get("age")) if v else {}
    _render_health_pillars(st.session_state.profile, v, _status_map,
                           st.session_state.report, lang)
    urgent_kw=["chest pain","πόνος στήθους","stroke","εγκεφαλικό","anaphylaxis","αναφυλαξία","166","112","emergency","επείγον","unconscious","αναίσθητος"]
    if any(kw in st.session_state.report.lower() for kw in urgent_kw):
        st.markdown('<div class="red-flags-urgent">🚨 Η αναφορά περιέχει <b>επείγουσες ενδείξεις</b>. Καλέστε <b>166</b> ή <b>112</b> αμέσως αν ισχύουν.</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="emergency">{t("emergency")}</div>',unsafe_allow_html=True)
    st.markdown('<div class="disclaimer-red">AI-generated. Δεν αντικαθιστά ιατρική γνώμη.</div>',unsafe_allow_html=True)

    # ── Feedback (👍/👎) — quality signal only, no medical data stored ──────────
    st.markdown("---")
    if st.session_state.get("fb_sent"):
        st.success("Ευχαριστούμε για το feedback!" if lang=="el" else "Thanks for your feedback!")
    else:
        st.caption("Σου φάνηκε χρήσιμη η εκτίμηση; (μας βοηθάει να βελτιωνόμαστε)" if lang=="el" else "Was this assessment helpful? (helps us improve)")
        rating = st.session_state.get("fb_rating", "")
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button(("👍 Χρήσιμη" if lang=="el" else "👍 Helpful"), key="fb_up", use_container_width=True,
                         type=("primary" if rating=="up" else "secondary")):
                st.session_state["fb_rating"]="up"; st.rerun()
        with fc2:
            if st.button(("👎 Όχι χρήσιμη" if lang=="el" else "👎 Not helpful"), key="fb_down", use_container_width=True,
                         type=("primary" if rating=="down" else "secondary")):
                st.session_state["fb_rating"]="down"; st.rerun()
        if rating:
            comment = st.text_area(("Τι θα βελτίωνες; (προαιρετικό)" if lang=="el" else "What would you improve? (optional)"),
                                   key="fb_comment", height=80)
            if st.button(("Αποστολή" if lang=="el" else "Submit"), key="fb_submit", type="primary"):
                save_feedback(rating, comment)
                st.session_state["fb_sent"]=True; st.rerun()

    fname=f"asklepios_report_{p.get('name','patient')}_{datetime.now().strftime('%Y%m%d')}"
    c1,c2,c3,c4=st.columns(4)
    with c1:
        if st.button("← "+("Νέα Αξιολόγηση" if lang=="el" else "New Assessment"),use_container_width=True):
            delete_draft(st.session_state.get("auth_user",""))
            for k,vv in defaults.items(): st.session_state[k]=vv
            for fbk in ("fb_comment","fb_rating","fb_sent","photo_added","photo_findings",
                        "_draft_hash","_from_facescan","_scan_injected","_vitals_nudge_off",
                        "_gpt_integrated","_photo_preview",
                        "lab_added","lab_findings","_lab_preview"): st.session_state.pop(fbk, None)
            st.rerun()
    with c2:
        # TXT: report + recs (plain text) so the file is self-contained
        _txt_parts = [st.session_state.report or ""]
        _r = st.session_state.get("report_recs")
        if _r and any(_r.get(k) for k in ("exercise","nutrition","lifestyle")):
            _hdr = ("ΕΞΑΤΟΜΙΚΕΥΜΕΝΕΣ ΣΥΣΤΑΣΕΙΣ" if lang=="el" else "PERSONALISED RECOMMENDATIONS")
            _lbls = (("Φυσική Δραστηριότητα","Διατροφή","Τρόπος Ζωής") if lang=="el"
                     else ("Exercise","Nutrition","Lifestyle"))
            _txt_parts += [
                "", "", "## " + _hdr,
                f"🏃 {_lbls[0]}: " + _r.get("exercise","—"),
                f"🥗 {_lbls[1]}: " + _r.get("nutrition","—"),
                f"🌿 {_lbls[2]}: " + _r.get("lifestyle","—"),
            ]
        _txt_full = "\n".join(_txt_parts)
        st.download_button("📄 TXT",data=_txt_full,file_name=fname+".txt",mime="text/plain",use_container_width=True)
    with c3:
        _recs_for_html = dict(st.session_state.get("report_recs") or {})
        if _recs_for_html:
            _recs_for_html["_refs"] = st.session_state.get("report_recs_refs") or {}
        _pf_for_html = st.session_state.get("photo_findings") or []
        if not isinstance(_pf_for_html, list):
            _pf_for_html = []
        _lf_for_html = st.session_state.get("lab_findings") or []
        if not isinstance(_lf_for_html, list):
            _lf_for_html = []
        st.download_button("📄 PDF/HTML",data=generate_html_report(st.session_state.profile,st.session_state.vitals,st.session_state.report,st.session_state.report_pubmed,lang=lang,recs=_recs_for_html,photo_findings=_pf_for_html,lab_findings=_lf_for_html),file_name=fname+".html",mime="text/html",use_container_width=True,help="Open in browser → Ctrl+P → Save as PDF")
    with c4:
        import re as _re_wa
        wa_lines=[f"🩺 Asklepios AI Nurse",
                  f"Ασθενής: {p.get('name','')} {p.get('age','')}y · {p.get('sex','')}"]
        vbits=[]
        if v.get("hr"):     vbits.append(f"HR {v['hr']}bpm")
        if v.get("bp_sys"): vbits.append(f"BP {v['bp_sys']}/{v.get('bp_dia','?')}mmHg")
        if v.get("br"):     vbits.append(f"BR {v['br']}/min")
        if v.get("spo2"):   vbits.append(f"SpO2 {v['spo2']}%")
        if v.get("temp"):   vbits.append(f"T {v['temp']}°C")
        if v.get("bmi"):    vbits.append(f"ΔΜΣ {v['bmi']}")
        if vbits: wa_lines.append("Ζωτικά: "+", ".join(vbits))
        # Clean markdown from report so it reads well in WhatsApp
        rep=_re_wa.sub(r"[#*>`|]", "", st.session_state.report or "").strip()
        rep=_re_wa.sub(r"\n{3,}", "\n\n", rep)
        # Cap length — wa.me pre-fill fails on very long URLs
        if len(rep)>1500:
            rep=rep[:1500].rsplit("\n",1)[0].rstrip()+"\n…(πλήρης αναφορά στο PDF)"
        if rep:
            wa_lines+=["", rep]
        # PNOE-style recs in WhatsApp (plain emoji-prefixed lines)
        _r2 = st.session_state.get("report_recs")
        if _r2 and any(_r2.get(k) for k in ("exercise","nutrition","lifestyle")):
            _lbls2 = (("Άσκηση","Διατροφή","Τρόπος ζωής") if lang=="el"
                      else ("Exercise","Nutrition","Lifestyle"))
            wa_lines += ["", ("📍 Συστάσεις:" if lang=="el" else "📍 Recommendations:")]
            if _r2.get("exercise"):  wa_lines.append(f"🏃 {_lbls2[0]}: {_r2['exercise']}")
            if _r2.get("nutrition"): wa_lines.append(f"🥗 {_lbls2[1]}: {_r2['nutrition']}")
            if _r2.get("lifestyle"): wa_lines.append(f"🌿 {_lbls2[2]}: {_r2['lifestyle']}")
        wa_lines+=["", "---", "⚠️ AI-generated. asklepiosainurse.up.railway.app"]
        msg="\n".join(wa_lines)
        wa_url="https://wa.me/?text="+urllib.parse.quote(msg)
        st.markdown(f'<a href="{wa_url}" target="_blank" style="display:block;text-align:center;padding:8px;border-radius:8px;text-decoration:none;font-weight:600;font-size:13px;color:white;background:#25D366">WhatsApp</a>',unsafe_allow_html=True)

# ── COOKIE MANAGER (once) — persistent login + in-progress profile draft ──────
if _STX_OK and auth_enabled():
    try:
        CM = stx.CookieManager()
    except Exception:
        CM = None

# Read the login cookie (single component call).
_all_cookies = {}
if CM is not None:
    try:
        _all_cookies = CM.get_all() or {}
    except Exception:
        _all_cookies = {}

# Restore login from the signed cookie (keeps the user signed in across reloads /
# new tabs — e.g. the tab returning from the external face scan).
if auth_enabled() and not is_logged_in():
    _ctok = _all_cookies.get(COOKIE_NAME)
    _em = _read_token(_ctok) if _ctok else None
    if _em:
        st.session_state["auth_user"] = _em

# Restore the in-progress assessment from the ENCRYPTED server-side draft ONLY
# when returning from the face scan (which sets _from_facescan on the very first
# run of the new tab). General re-opens of the app stay clean — the user does NOT
# pick up an old conversation just because they opened the app again.
#
# Order matters: on run 1 (URL has facescan param), this block sees _from_facescan
# still False (set later by the facescan block below), so it skips. The facescan
# block sets _from_facescan=True and st.rerun() → on run 2 this block fires.
if (auth_enabled() and is_logged_in()
        and st.session_state.get("_from_facescan")
        and not st.session_state.profile.get("name")
        and not st.session_state.get("_draft_loaded")):
    st.session_state["_draft_loaded"] = True
    _dd = load_draft(st.session_state.get("auth_user", ""))
    if _dd and (_dd.get("profile") or {}).get("name"):
        st.session_state.profile = _dd["profile"]
        if _dd.get("lang"):
            st.session_state.lang = _dd["lang"]
        if _dd.get("triage_chat"):
            st.session_state.triage_chat = _dd["triage_chat"]
        if _dd.get("vitals_analysis"):
            st.session_state.vitals_analysis = _dd["vitals_analysis"]
        if _dd.get("medications"):
            st.session_state.medications = _dd["medications"]
        else:
            _mr = st.session_state.profile.get("meds_raw", "")
            st.session_state.medications = [{"name": m.strip(), "freq": "", "notes": ""}
                                            for m in _mr.split(",") if m.strip()] if _mr else []
        # One-shot: the draft has served its purpose, delete it so it does NOT
        # resurrect on later re-opens.
        delete_draft(st.session_state.get("auth_user", ""))

# If we came back from the face scan during an ongoing conversation, drop the
# measurement into the chat so Asklepios continues the SAME assessment with it
# (instead of the result just sitting silently in the vitals badge).
if (st.session_state.get("_from_facescan") and st.session_state.triage_chat
        and not st.session_state.get("_scan_injected")):
    _v = st.session_state.vitals
    _bits = []
    if _v.get("hr"):
        _bits.append((f"καρδιακός ρυθμός {_v['hr']} bpm" if st.session_state.lang=="el"
                      else f"heart rate {_v['hr']} bpm"))
    if _v.get("br"):
        _bits.append((f"αναπνοές {_v['br']}/min" if st.session_state.lang=="el"
                      else f"breathing {_v['br']}/min"))
    if _bits:
        _m = (("Μέτρησα τα ζωτικά μου με τη σάρωση: " if st.session_state.lang=="el"
               else "I measured my vitals with the scan: ") + ", ".join(_bits) + ".")
        st.session_state.triage_chat.append({"role": "user", "content": _m})
        st.session_state["_scan_injected"] = True
        st.session_state["_scan_reply_pending"] = True

# ── FACESCAN INTERCEPTION ─────────────────────────────────────────────────────
try:
    _raw = st.query_params.get("facescan","")
    if _raw:
        _scanned = json.loads(urllib.parse.unquote(_raw))
        if _scanned and isinstance(_scanned, dict):
            # Filter out null values — only keep actual measurements
            _clean = {k: v for k, v in _scanned.items()
                      if v is not None and v != 0 and k not in ("quality","wellness")}
            # Keep quality and wellness as-is even if 0
            if "quality"  in _scanned: _clean["quality"]  = _scanned["quality"]
            if "wellness" in _scanned: _clean["wellness"] = _scanned["wellness"]
            if _clean:
                st.session_state.vitals = _clean
                st.session_state["_from_facescan"] = True
                st.session_state["_fs_banner"] = True
                st.session_state["_scan_injected"] = False
                st.session_state.screen = "triage" if st.session_state.profile.get("name") else "intake"
            st.query_params.clear()
            st.rerun()
except Exception:
    pass

# If we returned from the face scan and the profile draft has since been restored
# (the cookie can take a render to arrive), skip the now-prefilled intake form and
# jump straight to the assessment.
if (st.session_state.get("_from_facescan") and st.session_state.vitals
        and st.session_state.profile.get("name") and st.session_state.screen == "intake"):
    st.session_state.screen = "triage"
    st.session_state["_from_facescan"] = False

# ── LOGIN GATE ────────────────────────────────────────────────────────────────
# Login-first: every visitor is identified in Supabase → Authentication → Users.
if auth_enabled() and not is_logged_in():
    render_login_screen()
    st.stop()

# ── PERSIST login cookie on a CLEAN render pass ───────────────────────────────
# The login cookie write on the verify *click* + immediate st.rerun() is unreliable
# (the rerun aborts the stx browser write); a normal render that completes lands it.
if CM is not None and is_logged_in() and not st.session_state.get("_cookie_synced"):
    _save_login_cookie(st.session_state.get("auth_user", ""))
    st.session_state["_cookie_synced"] = True
# NOTE: the encrypted draft is NOT saved on every clean render. It is saved only
# when about to leave for an external page (face scan) via
# _save_session_for_external_nav(). After the round-trip it is one-shot deleted.
# This way the user does NOT accumulate a saved conversation across general
# re-opens — re-entering the app starts clean.

screen=st.session_state.screen
render_topbar()
if st.session_state.pop("_fs_banner", False):
    v_loaded = st.session_state.vitals
    metrics  = [f"HR:{v_loaded['hr']}bpm" if "hr" in v_loaded else "",
                f"BR:{v_loaded['br']}/min" if "br" in v_loaded else "",
                f"HRV:{v_loaded['hrv']}ms" if "hrv" in v_loaded else ""]
    metrics_str = " · ".join(m for m in metrics if m)
    lang = st.session_state.lang
    msg = (f"✅ Σάρωση φορτώθηκε! {metrics_str}" if lang=="el"
           else f"✅ Face scan loaded! {metrics_str}")
    st.success(msg)
if   screen=="home":   render_home()
elif screen=="intake": render_intake()
elif screen=="vitals": render_vitals()
elif screen=="triage": render_triage()
elif screen=="report": render_report()
else: render_home()
