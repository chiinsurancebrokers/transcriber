# 🎙️ Greek Audio Transcriber

Transcribe long Greek audio recordings (40–60 min WAV files) using **ElevenLabs Scribe** (primary) with automatic fallback to **OpenAI Whisper**.

---

## How it works

| Step | Detail |
|------|--------|
| Upload | WAV / MP3 / M4A / OGG, up to 1 GB |
| ElevenLabs | Sends the full file to Scribe v1 (`language_code=el`) |
| Fallback | If ElevenLabs fails, splits audio into 10-min MP3 chunks and transcribes with Whisper-1 |
| Output | Copy transcript in-app, or download as `.txt` or `.srt` |

---

## 🚀 Deploy to Streamlit Cloud (free, 5 minutes)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Greek transcriber"
gh repo create greek-transcriber --public --push  # or use GitHub Desktop
```

### 2. Go to Streamlit Cloud
- Open **https://share.streamlit.io**
- Click **"New app"**
- Select your GitHub repo → branch `main` → `app.py`

### 3. Add your API keys (Secrets)
In the Streamlit Cloud app settings → **"Secrets"** tab, paste:
```toml
ELEVENLABS_API_KEY = "xi-..."
OPENAI_API_KEY     = "sk-..."
```

### 4. Deploy!
Click **"Deploy"** — your app will be live at:
`https://<your-username>-greek-transcriber-app-xxxx.streamlit.app`

---

## Run locally

```bash
pip install -r requirements.txt
# Install ffmpeg (required by pydub):
#   macOS:   brew install ffmpeg
#   Ubuntu:  sudo apt install ffmpeg
#   Windows: https://ffmpeg.org/download.html

# Copy and fill in your keys:
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

streamlit run app.py
```

---

## API limits

| API | File size | Duration | Greek |
|-----|-----------|----------|-------|
| ElevenLabs Scribe v1 | up to 1 GB | up to ~4.5 h | ✅ `el` |
| OpenAI Whisper-1 | 25 MB per chunk | unlimited (chunked) | ✅ `el` |

---

## Files

```
greek-transcriber/
├── app.py                         # Main Streamlit app
├── requirements.txt               # Python dependencies
├── packages.txt                   # System packages (ffmpeg)
├── .streamlit/
│   ├── config.toml                # Upload size limit + theme
│   └── secrets.toml.example       # API key template
└── README.md
```
