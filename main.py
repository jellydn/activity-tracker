"""
Activity Tracker - OCR Screenshot Logger
Single-file FastAPI backend with RapidOCR, SQLite, modern Python stack.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# ─── Config ───────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "data" / "activity.db"
UPLOAD_DIR = Path(__file__).parent / "data" / "thumbnails"
OCR_LANG = os.getenv("OCR_LANG", "en")

# ─── App Detection ────────────────────────────────────────────────────────────

APP_RULES: list[tuple[str, list[str], str]] = [
    # (name, patterns, category)
    ("VS Code", ["visual studio code", "vscode", "code - insiders"], "editor"),
    ("Cursor", ["cursor", "cursor.com"], "editor"),
    ("Vim/Neovim", ["nvim", "vim", "neovim"], "editor"),
    ("Terminal", ["terminal", "iterm", "zsh", "bash", "powershell", "cmd.exe"], "editor"),
    ("JetBrains", ["intellij", "pycharm", "webstorm", "jetbrains"], "editor"),
    ("Sublime", ["sublime text", "subl"], "editor"),
    ("Chrome", ["chrome://", "google chrome", "chromium"], "browser"),
    ("Firefox", ["firefox", "mozilla firefox"], "browser"),
    ("Safari", ["safari"], "browser"),
    ("Edge", ["microsoft edge", "edge://"], "browser"),
    ("Brave", ["brave"], "browser"),
    ("Slack", ["slack.com", "slack"], "communication"),
    ("Discord", ["discord.com", "discord"], "communication"),
    ("Telegram", ["telegram.org", "telegram"], "communication"),
    ("WhatsApp", ["web.whatsapp.com", "whatsapp"], "communication"),
    ("Gmail", ["mail.google.com", "gmail"], "communication"),
    ("Outlook", ["outlook.live.com", "outlook.office.com", "outlook"], "communication"),
    ("Twitter/X", ["x.com", "twitter.com"], "social"),
    ("Reddit", ["reddit.com", "old.reddit"], "social"),
    ("YouTube", ["youtube.com", "youtu.be"], "social"),
    ("LinkedIn", ["linkedin.com"], "social"),
    ("GitHub", ["github.com"], "dev"),
    ("GitLab", ["gitlab.com"], "dev"),
    ("Bitbucket", ["bitbucket.org"], "dev"),
    ("Notion", ["notion.so", "notion"], "productivity"),
    ("Figma", ["figma.com", "figma"], "productivity"),
    ("Jira", ["atlassian.net", "jira"], "productivity"),
    ("Linear", ["linear.app"], "productivity"),
    ("Trello", ["trello.com"], "productivity"),
    ("Google Docs", ["docs.google.com"], "productivity"),
    ("Google Sheets", ["sheets.google.com"], "productivity"),
    ("Excel", ["excel", "spreadsheet"], "productivity"),
    ("ChatGPT", ["chat.openai.com", "chatgpt"], "ai"),
    ("Claude", ["claude.ai"], "ai"),
    ("Perplexity", ["perplexity.ai"], "ai"),
    ("Jupyter", ["jupyter", "notebook", "localhost:8888"], "dev"),
    ("Docker", ["docker", "container"], "dev"),
    ("Kubernetes", ["kubernetes", "k8s", "kubectl"], "dev"),
    ("AWS", ["aws.amazon.com", "console.aws"], "dev"),
    ("GCP", ["console.cloud.google.com", "cloud.google"], "dev"),
    ("Spotify", ["open.spotify.com", "spotify"], "media"),
    ("Zoom", ["zoom.us"], "communication"),
    ("Google Meet", ["meet.google.com"], "communication"),
    ("Finder", ["finder"], "system"),
    ("File Explorer", ["file explorer", "windows explorer"], "system"),
    ("Settings", ["settings", "system preferences", "system settings"], "system"),
]


def detect_app(text: str) -> tuple[str, str, str]:
    """Return (name, category, matched_pattern) or ("Unknown", "unknown", "")."""
    lower = text.lower()
    for name, patterns, category in APP_RULES:
        for pat in patterns:
            if pat.lower() in lower:
                return name, category, pat
    return "Unknown", "unknown", ""


def detect_url(text: str) -> str | None:
    m = re.search(r"https?://[^\s\]\)\"']+", text)
    return m.group(0) if m else None


def detect_language(text: str) -> str:
    code_kw = [
        "function", "const ", "let ", "var ", "import ", "export ",
        "class ", "def ", "public ", "private ", "#include", "package ",
        "select ", "insert ", "create table", "git ", "npm ", "pip ",
        "docker", "kubectl", "curl ", "ssh ", "sudo ", "apt ", "brew ",
    ]
    lower = text.lower()
    score = sum(1 for kw in code_kw if kw in lower)
    if score >= 3:
        return "Source Code"
    if score >= 1:
        return "Mixed"
    return "Text / UI"


def extract_title(text: str) -> str:
    skip = {"file", "edit", "view", "help", "window", "new tab", "search", "home"}
    for line in text.split("\n"):
        t = line.strip()
        if 2 < len(t) < 120 and not any(t.lower().startswith(s) for s in skip):
            return t[:80]
    return "Unknown"


def detect_tags(text: str) -> list[str]:
    lower = text.lower()
    tags: list[str] = []
    if any(w in lower for w in ["error", "exception", "traceback", "failed", "panic"]):
        tags.append("Debugging")
    if any(w in lower for w in ["git ", "commit", "merge", "pull request", "branch"]):
        tags.append("Git")
    if any(w in lower for w in ["npm ", "pip ", "install", "package", "dependency"]):
        tags.append("Package Mgmt")
    if any(w in lower for w in ["docker", "container", "kubernetes", "k8s"]):
        tags.append("DevOps")
    if any(w in lower for w in ["test", "assert", "expect", "jest", "pytest"]):
        tags.append("Testing")
    if any(w in lower for w in ["api", "endpoint", "request", "response", "rest", "graphql"]):
        tags.append("API")
    if any(w in lower for w in ["database", "sql", "query", "mongo", "postgres"]):
        tags.append("Database")
    if any(w in lower for w in ["css", "html", "react", "vue", "tailwind", "style"]):
        tags.append("Frontend")
    if any(w in lower for w in ["deploy", "ci/", "pipeline", "build"]):
        tags.append("Deployment")
    if any(w in lower for w in ["config", ".env", "settings", "yaml", "json"]):
        tags.append("Config")
    return tags or ["General"]


# ─── Database ─────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            app TEXT NOT NULL DEFAULT 'Unknown',
            app_category TEXT NOT NULL DEFAULT 'unknown',
            url TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '[]',
            text TEXT NOT NULL DEFAULT '',
            text_length INTEGER NOT NULL DEFAULT 0,
            confidence REAL NOT NULL DEFAULT 0,
            thumbnail TEXT NOT NULL DEFAULT '',
            source_width INTEGER NOT NULL DEFAULT 0,
            source_height INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_records_timestamp ON records(timestamp);
        CREATE INDEX IF NOT EXISTS idx_records_app ON records(app);
    """)
    return conn


def insert_record(conn: sqlite3.Connection, rec: dict) -> int:
    cur = conn.execute(
        """INSERT INTO records
           (timestamp, title, app, app_category, url, language, tags,
            text, text_length, confidence, thumbnail, source_width, source_height)
           VALUES (:timestamp, :title, :app, :app_category, :url, :language,
                   :tags, :text, :text_length, :confidence, :thumbnail,
                   :source_width, :source_height)""",
        rec,
    )
    conn.commit()
    return cur.lastrowid or 0


def get_all_records(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM records ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_stats(conn: sqlite3.Connection) -> dict:
    now = time.time()
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_ts = today_start.timestamp()
    week_ago = now - 7 * 86400

    total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    today = conn.execute("SELECT COUNT(*) FROM records WHERE timestamp >= ?", (today_ts,)).fetchone()[0]
    week = conn.execute("SELECT COUNT(*) FROM records WHERE timestamp >= ?", (week_ago,)).fetchone()[0]
    apps = conn.execute("SELECT COUNT(DISTINCT app) FROM records WHERE app != 'Unknown'").fetchone()[0]

    return {"total": total, "today": today, "week": week, "apps": apps}


def delete_all_records(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM records")
    conn.commit()


# ─── OCR ──────────────────────────────────────────────────────────────────────

_ocr: RapidOCR | None = None


def get_ocr() -> RapidOCR:
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR(lang=OCR_LANG)
    return _ocr


def run_ocr(image_bytes: bytes) -> tuple[str, float]:
    img = Image.open(io.BytesIO(image_bytes))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    ocr = get_ocr()
    result, elapse = ocr(img_bytes.read())
    if not result:
        return "", 0.0
    texts = [item[1] for item in result]
    # confidence is item[2] in each result entry
    confs = [item[2] for item in result if len(item) > 2]
    avg_conf = sum(confs) / len(confs) * 100 if confs else 0.0
    return "\n".join(texts), avg_conf


def make_thumbnail(image_bytes: bytes, size: int = 200) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


# ─── FastAPI App ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    get_ocr()  # warm up OCR
    yield


app = FastAPI(title="Activity Tracker", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: Annotated[UploadFile, File()]):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20MB)")

    # OCR
    text, confidence = run_ocr(contents)

    # Image info
    img = Image.open(io.BytesIO(contents))
    w, h = img.size

    # Analysis
    app_name, app_cat, _ = detect_app(text)
    url = detect_url(text)
    language = detect_language(text)
    title = extract_title(text)
    tags = detect_tags(text)
    thumbnail = make_thumbnail(contents)

    # Save
    conn = get_db()
    rec = {
        "timestamp": time.time(),
        "title": title,
        "app": app_name,
        "app_category": app_cat,
        "url": url or "",
        "language": language,
        "tags": json.dumps(tags),
        "text": text[:3000],
        "text_length": len(text),
        "confidence": round(confidence, 1),
        "thumbnail": thumbnail,
        "source_width": w,
        "source_height": h,
    }
    rid = insert_record(conn, rec)
    conn.close()

    return JSONResponse({
        "id": rid,
        **rec,
        "tags": tags,
    })


@app.get("/api/records")
async def list_records():
    conn = get_db()
    records = get_all_records(conn)
    conn.close()
    for r in records:
        r["tags"] = json.loads(r["tags"])
    return records


@app.get("/api/stats")
async def stats():
    conn = get_db()
    s = get_stats(conn)
    conn.close()
    return s


@app.delete("/api/records")
async def clear_all():
    conn = get_db()
    delete_all_records(conn)
    conn.close()
    return {"ok": True}


@app.get("/api/export")
async def export_data():
    conn = get_db()
    records = get_all_records(conn, limit=10000)
    conn.close()
    for r in records:
        r["tags"] = json.loads(r["tags"])
        r["timestamp_iso"] = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).isoformat()
    return JSONResponse(records)


# ─── Frontend ─────────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Activity Tracker</title>
<style>
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;--text2:#8b949e;--accent:#58a6ff;--accent-soft:rgba(88,166,255,.12);--green:#3fb950;--red:#f85149;--orange:#d29922}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.wrap{max-width:960px;margin:0 auto;padding:24px 16px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid var(--border)}
header h1{font-size:20px;font-weight:600}header h1 span{color:var(--accent)}
.hdr-actions{display:flex;gap:8px}
button{font-family:inherit;font-size:13px;font-weight:500;padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;transition:all .15s}
button:hover{border-color:var(--accent);background:var(--accent-soft)}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
button.danger{border-color:var(--red);color:var(--red)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}
.sc{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
.sc .v{font-size:24px;font-weight:700;color:var(--accent)}
.sc .l{font-size:12px;color:var(--text2);margin-top:2px}
.upload{border:2px dashed var(--border);border-radius:12px;padding:48px 24px;text-align:center;cursor:pointer;transition:all .2s;margin-bottom:24px;background:var(--surface)}
.upload:hover,.upload.drag{border-color:var(--accent);background:var(--accent-soft)}
.upload svg{width:48px;height:48px;color:var(--text2);margin-bottom:12px}
.upload p{color:var(--text2);font-size:14px;margin-top:8px}
.upload .hint{font-size:12px;margin-top:4px;opacity:.6}
#fi{display:none}
.ocr-sec{display:none;gap:16px;margin-bottom:24px}
.ocr-sec.on{display:grid;grid-template-columns:1fr 1fr}
@media(max-width:640px){.ocr-sec.on{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px}
.panel h3{font-size:13px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}
.pv{max-width:100%;max-height:300px;border-radius:6px;border:1px solid var(--border)}
.pb{width:100%;height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:8px}
.pb .f{height:100%;background:var(--accent);border-radius:2px;transition:width .3s;width:0%}
.pt{font-size:12px;color:var(--text2);margin-top:4px}
.of{font-family:'SF Mono',Monaco,monospace;font-size:12px;line-height:1.6;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow-y:auto;color:var(--text2)}
.of.has{color:var(--text)}
.ar{display:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:24px}
.ar.on{display:block}
.ar h3{font-size:13px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}
.am{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:12px}
.mi{background:var(--bg);border-radius:6px;padding:10px 12px}
.mi label{font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px}
.mi .v{font-size:14px;font-weight:500;margin-top:2px}
.tl{margin-top:8px}
.tl-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.tl-h h2{font-size:16px;font-weight:600}
.tl-h .c{font-size:12px;color:var(--text2)}
.tl-list{display:flex;flex-direction:column;gap:8px}
.ti{display:flex;gap:12px;padding:12px;background:var(--surface);border:1px solid var(--border);border-radius:8px;transition:border-color .15s;cursor:pointer}
.ti:hover{border-color:var(--accent)}
.ti-th{width:64px;height:48px;border-radius:4px;object-fit:cover;border:1px solid var(--border);flex-shrink:0}
.ti-c{flex:1;min-width:0}
.ti-t{font-size:11px;color:var(--text2);margin-bottom:2px}
.ti-tl{font-size:13px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ti-s{font-size:12px;color:var(--text2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
.ti-tags{display:flex;gap:4px;margin-top:6px}
.ti-tags .tag{font-size:10px;padding:1px 6px}
.tl-empty{text-align:center;padding:48px 24px;color:var(--text2);font-size:14px}
.tl-empty svg{width:40px;height:40px;margin-bottom:12px;opacity:.4}
.tag{font-size:11px;padding:2px 8px;border-radius:10px;background:var(--accent-soft);color:var(--accent);border:1px solid rgba(88,166,255,.2)}
.tag.green{background:rgba(63,185,80,.12);color:var(--green);border-color:rgba(63,185,80,.2)}
.tag.orange{background:rgba(210,153,34,.12);color:var(--orange);border-color:rgba(210,153,34,.2)}
.mo{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;align-items:center;justify-content:center;padding:24px}
.mo.on{display:flex}
.mc{background:var(--surface);border:1px solid var(--border);border-radius:12px;max-width:720px;width:100%;max-height:90vh;overflow-y:auto;padding:24px}
.mc h2{font-size:16px;margin-bottom:16px}
.mc-cl{float:right;background:none;border:none;color:var(--text2);font-size:20px;cursor:pointer;padding:0;line-height:1}
.mc-cl:hover{color:var(--text);background:none}
.mc-i{width:100%;border-radius:6px;border:1px solid var(--border);margin-bottom:12px}
.mc-tx{font-family:'SF Mono',Monaco,monospace;font-size:12px;line-height:1.6;white-space:pre-wrap;word-break:break-word;background:var(--bg);border-radius:6px;padding:12px;max-height:300px;overflow-y:auto}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>
<div class="wrap">
<header>
<h1><span>&#9776;</span> Activity Tracker</h1>
<div class="hdr-actions">
<button onclick="exportData()">&#8682; Export</button>
<button class="danger" onclick="clearAll()">&#128465; Clear</button>
</div>
</header>
<div class="stats">
<div class="sc"><div class="v" id="s-total">0</div><div class="l">Screenshots</div></div>
<div class="sc"><div class="v" id="s-today">0</div><div class="l">Today</div></div>
<div class="sc"><div class="v" id="s-apps">0</div><div class="l">Apps</div></div>
<div class="sc"><div class="v" id="s-week">0</div><div class="l">This Week</div></div>
</div>
<div class="upload" id="up">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
<strong>Drop screenshot here or click to upload</strong>
<p>PNG, JPG, WEBP &middot; Ctrl+V to paste</p>
<p class="hint">Win+Shift+S / Cmd+Shift+4 to capture</p>
<input type="file" id="fi" accept="image/*">
</div>
<div class="ocr-sec" id="ocr-sec">
<div class="panel"><h3>Preview</h3><img class="pv" id="pv" alt="Preview"></div>
<div class="panel"><h3>OCR</h3><div class="pb"><div class="f" id="pf"></div></div><div class="pt" id="pt">Initializing...</div><div class="of" id="of"></div></div>
</div>
<div class="ar" id="ar">
<h3>Detected Activity</h3>
<div class="am" id="am"></div>
<div id="atags" style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px"></div>
</div>
<div class="tl">
<div class="tl-h"><h2>Timeline</h2><span class="c" id="tl-c">0 entries</span></div>
<div class="tl-list" id="tl-l">
<div class="tl-empty" id="tl-e">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
<p>No activity yet. Upload a screenshot.</p>
</div>
</div>
</div>
</div>
<div class="mo" id="mo" onclick="closeMo(event)">
<div class="mc">
<button class="mc-cl" onclick="document.getElementById('mo').classList.remove('on')">&#10005;</button>
<h2 id="mo-t">Detail</h2>
<img class="mc-i" id="mo-i" alt="">
<div class="mc-tx" id="mo-tx"></div>
</div>
</div>
<script>
const $=s=>document.querySelector(s);
const up=$("#up"),fi=$("#fi"),ocrSec=$("#ocr-sec"),pv=$("#pv"),pf=$("#pf"),pt=$("#pt"),of=$("#of"),ar=$("#ar"),am=$("#am"),atags=$("#atags"),tlList=$("#tl-l"),tlEmpty=$("#tl-e"),tlC=$("#tl-c"),mo=$("#mo");

up.onclick=()=>fi.click();
up.ondragover=e=>{e.preventDefault();up.classList.add("drag")};
up.ondragleave=()=>up.classList.remove("drag");
up.ondrop=e=>{e.preventDefault();up.classList.remove("drag");const f=e.dataTransfer.files[0];if(f?.type.startsWith("image/"))process(f)};
fi.onchange=e=>{const f=e.target.files[0];if(f)process(f)};
document.onpaste=e=>{for(const i of e.clipboardData.items){if(i.type.startsWith("image/")){e.preventDefault();process(i.getAsFile());return}}};

async function process(f){
  ocrSec.classList.add("on");ar.classList.remove("on");of.textContent="";of.classList.remove("has");pf.style.width="0%";
  pv.src=URL.createObjectURL(f);
  const fd=new FormData();fd.append("file",f);
  pt.textContent="Uploading...";
  try{
    const r=await fetch("/api/upload",{method:"POST",body:fd});
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    of.textContent=d.text||"(No text detected)";of.classList.add("has");
    pf.style.width="100%";pt.textContent=`Done - ${d.text_length} chars, ${d.confidence.toFixed(0)}% conf`;
    showActivity(d);await loadTimeline();
  }catch(e){pt.textContent="Error: "+e.message;of.textContent=e.message}
}

function showActivity(d){
  ar.classList.add("on");
  const meta=[{l:"App",v:d.app},{l:"Category",v:d.app_category},{l:"Type",v:d.language},{l:"Size",v:`${d.source_width}x${d.source_height}`},{l:"Time",v:new Date(d.timestamp*1000).toLocaleTimeString()}];
  if(d.url)meta.push({l:"URL",v:d.url});
  am.innerHTML=meta.map(m=>`<div class="mi"><label>${m.l}</label><div class="v">${m.v}</div></div>`).join("");
  atags.innerHTML=d.tags.map(t=>`<span class="tag">${t}</span>`).join("");
}

async function loadTimeline(){
  const[records,stats]=await Promise.all([fetch("/api/records").then(r=>r.json()),fetch("/api/stats").then(r=>r.json())]);
  $("#s-total").textContent=stats.total;$("#s-today").textContent=stats.today;$("#s-apps").textContent=stats.apps;$("#s-week").textContent=stats.week;
  tlC.textContent=`${records.length} entries`;
  if(!records.length){tlEmpty.style.display="";tlList.innerHTML="";tlList.appendChild(tlEmpty);return}
  tlEmpty.style.display="none";tlList.innerHTML="";
  for(const r of records.slice(0,50)){
    const el=document.createElement("div");el.className="ti";el.onclick=()=>showDetail(r);
    const t=new Date(r.timestamp*1000);
    const ts=t.toLocaleDateString()===new Date().toLocaleDateString()?t.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}):t.toLocaleDateString([],{month:'short',day:'numeric'})+" "+t.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
    el.innerHTML=`${r.thumbnail?`<img class="ti-th" src="data:image/jpeg;base64,${r.thumbnail}">`:`<div class="ti-th" style="background:var(--bg);display:flex;align-items:center;justify-content:center;color:var(--text2);font-size:10px">No img</div>`}<div class="ti-c"><div class="ti-t">${ts} &middot; ${r.app}</div><div class="ti-tl">${r.title}</div><div class="ti-s">${(r.text||"").substring(0,100)}...</div><div class="ti-tags">${(r.tags||[]).map(t=>`<span class="tag">${t}</span>`).join("")}</div></div>`;
    tlList.appendChild(el);
  }
}

function showDetail(r){
  $("#mo-t").textContent=r.title;
  $("#mo-i").src=r.thumbnail?`data:image/jpeg;base64,${r.thumbnail}`:"";
  $("#mo-tx").textContent=r.text||"";
  mo.classList.add("on");
}
function closeMo(e){if(e.target===e.currentTarget)mo.classList.remove("on")}

async function exportData(){
  const r=await fetch("/api/export");const d=await r.json();
  if(!d.length)return alert("No data");
  const blob=new Blob([JSON.stringify(d,null,2)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`activity-${new Date().toISOString().slice(0,10)}.json`;a.click();
}
async function clearAll(){
  if(!confirm("Delete all records?"))return;
  await fetch("/api/records",{method:"DELETE"});await loadTimeline();ar.classList.remove("on");
}
loadTimeline();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host="0.0.0.0", port=port)
