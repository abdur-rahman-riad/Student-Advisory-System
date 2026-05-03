import os, json, time, logging
from typing import Optional
from functools import lru_cache

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sas")

# App
app = FastAPI(title="Student Advisory System API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config from environment
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")          # primary key
GEMINI_API_KEY_2 = os.environ.get("GEMINI_API_KEY_2", "")        # fallback key
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_BASE      = "https://generativelanguage.googleapis.com/v1beta"

# Google Sheets config
SHEET_ID         = os.environ.get("SHEET_ID", "")                # spreadsheet ID
SHEET_TAB        = os.environ.get("SHEET_TAB", "Sheet1")         # tab / worksheet name
# Service account JSON stored as env var (stringify the whole JSON)
GCP_SA_JSON      = os.environ.get("GCP_SA_JSON", "")             # service account JSON string

# In-memory data store
_store = {
    "students":     [],          # [{id, name}]
    "records":      {},          # {student_id: {fields: {...}, context: "...", chunks: [...]}}
    "last_loaded":  0,
    "id_column":    "Student ID",
}

# Google Sheets loader
def _get_gsheet_client():
    if not GCP_SA_JSON:
        raise RuntimeError("GCP_SA_JSON env var not set")
    sa_info = json.loads(GCP_SA_JSON)
    scopes  = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def _detect_id_column(headers: list[str]) -> str:
    """Auto-detect which column holds the student ID."""
    priority = ["Student ID", "ID", "student_id", "Roll", "Roll No", "Roll Number"]
    for p in priority:
        if p in headers:
            return p
    return headers[0]


def _row_to_context(fields: dict) -> str:
    """Convert flat field dict to a rich text block for RAG chunking."""
    lines = ["=== STUDENT RECORD ==="]
    for k, v in fields.items():
        if v and str(v).strip():
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def load_data_from_sheet():
    """Pull all rows from Google Sheets and rebuild in-memory store."""
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID env var not set")

    log.info("Loading data from Google Sheets: %s / %s", SHEET_ID, SHEET_TAB)
    client = _get_gsheet_client()
    sh     = client.open_by_key(SHEET_ID)
    ws     = sh.worksheet(SHEET_TAB)
    rows   = ws.get_all_records()          # list of dicts keyed by header

    if not rows:
        raise RuntimeError("Sheet is empty or has no data rows")

    headers   = list(rows[0].keys())
    id_col    = _detect_id_column(headers)
    students  = []
    records   = {}

    for row in rows:
        sid  = str(row.get(id_col, "")).strip()
        name = str(row.get("Name", row.get("Student Name", sid))).strip()
        if not sid:
            continue
        fields  = {k: str(v).strip() for k, v in row.items() if v != ""}
        context = _row_to_context(fields)
        students.append({"id": sid, "name": name})
        records[sid] = {
            "id":      sid,
            "fields":  fields,
            "context": context,
            "chunks":  _chunk_context(context),
        }

    _store["students"]    = students
    _store["records"]     = records
    _store["last_loaded"] = time.time()
    _store["id_column"]   = id_col
    log.info("Loaded %d students", len(students))


def _chunk_context(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Split context text into overlapping chunks for RAG."""
    parts  = text.split("\n\n")
    chunks = []
    cur    = ""
    for p in parts:
        if len(cur) + len(p) + 2 <= size:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 10]


# Gemini proxy 
async def call_gemini(messages: list, system_prompt: str,
                      max_tokens: int = 1024, temperature: float = 0.1) -> dict:
    """
    Call Gemini generateContent. Falls back to GEMINI_API_KEY_2 on 429.
    """
    keys = [k for k in [GEMINI_API_KEY, GEMINI_API_KEY_2] if k]
    if not keys:
        raise HTTPException(503, "No Gemini API key configured")

    # Convert messages to Gemini format
    contents = []
    for m in messages:
        role  = "model" if m.get("role") == "model" else "user"
        parts = m.get("parts", [{"text": m.get("content", "")}])
        contents.append({"role": role, "parts": parts})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }

    last_err = None
    async with httpx.AsyncClient(timeout=30) as client:
        for key in keys:
            url = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent?key={key}"
            try:
                r = await client.post(url, json=payload)
                if r.status_code == 429 and len(keys) > 1:
                    log.warning("Primary Gemini key rate-limited, trying fallback")
                    last_err = r.text
                    continue
                if r.status_code != 200:
                    raise HTTPException(r.status_code, f"Gemini error: {r.text[:300]}")
                data = r.json()
                candidate = data["candidates"][0]["content"]["parts"][0]["text"]
                usage     = data.get("usageMetadata", {})
                return {
                    "text":         candidate,
                    "input_tokens":  usage.get("promptTokenCount", 0),
                    "output_tokens": usage.get("candidatesTokenCount", 0),
                }
            except httpx.RequestError as e:
                raise HTTPException(503, f"Network error reaching Gemini: {e}")

    raise HTTPException(429, f"All Gemini keys rate-limited: {last_err}")


# Pydantic models
class ChatRequest(BaseModel):
    student_id:    str
    messages:      list
    system_prompt: str
    max_tokens:    int = 1024
    temperature:   float = 0.1


# Routes
@app.on_event("startup")
async def startup():
    """Load data on boot; fail gracefully so the server still starts."""
    try:
        load_data_from_sheet()
    except Exception as e:
        log.error("Startup data load failed: %s", e)


@app.get("/api/health")
async def health():
    return {
        "status":          "ok",
        "students_cached": len(_store["students"]),
        "last_loaded":     _store["last_loaded"],
        "sheet_id":        SHEET_ID[:10] + "…" if SHEET_ID else "not set",
        "sheet_tab":       SHEET_TAB,
        "model":           GEMINI_MODEL,
    }


@app.get("/api/students")
async def list_students():
    if not _store["students"]:
        try:
            load_data_from_sheet()
        except Exception as e:
            raise HTTPException(503, str(e))
    return {"students": _store["students"]}


@app.get("/api/student/{student_id}")
async def get_student(student_id: str):
    rec = _store["records"].get(student_id)
    if not rec:
        raise HTTPException(404, f"Student '{student_id}' not found")
    return rec


@app.post("/api/reload")
async def reload_data():
    try:
        load_data_from_sheet()
        return {"status": "ok", "students": len(_store["students"])}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Verify student exists (don't trust client)
    if req.student_id not in _store["records"]:
        raise HTTPException(404, f"Student '{req.student_id}' not in cache")

    result = await call_gemini(
        messages      = req.messages,
        system_prompt = req.system_prompt,
        max_tokens    = min(req.max_tokens, 2048),
        temperature   = req.temperature,
    )
    return result


# Serve static frontend 
@app.get("/")
async def root():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
