# Signature System — FastAPI Backend (frontend contract ke mutabiq)
# Ye backend Signature_System frontend ke BACKEND_INTEGRATION.md wale
# contract par bana hai — endpoints, field names, response shapes sab wohi.
#
#   POST /api/v1/auth/login                     (JSON: email + password)
#   GET  /api/v1/auth/me
#   POST /api/v1/auth/logout
#   POST /api/v1/auth/register                  (extra — user banane ke liye;
#                                                frontend ise call nahi karta,
#                                                lekin login ke liye user
#                                                pehle banna zaroori hai)
#   POST /api/v1/signature-detection            (multipart field: document)
#   GET  /api/v1/signature-detection/{task_id}
#   GET  /api/v1/signature-detection            (history, ?page=&page_size=)
#   POST /api/v1/signature-verification         (fields: original_signature,
#                                                suspected_signature)
#   GET  /api/v1/signature-verification/{task_id}
#   GET  /api/v1/signature-verification         (history)
#
# Task lifecycle: queued -> processing -> completed / failed.
# POST foran task_id deta hai; model ka kaam background me hota hai aur
# frontend GET se poll karke result leta hai (contract section 4).
#
# Chalane ke liye:
#   pip install -r requirements.txt
#   python -m uvicorn main:app --reload --port 8000
# (python -m zaroori hai — PC par Scripts folder PATH me nahi)
# Frontend: .env me VITE_API_BASE_URL=http://localhost:8000 (default yahi hai)
#
# MODEL FILE: is folder me team ki final model file rakhein — in naamo me se
# koi ek code khud pehchan lega: signature_model.pt, combined_best.pt,
# document_image_best.pt, signature_closeup_best.pt, best.pt
# (ya MODEL_PATH env var dein). Size 40,593,964 bytes = hamari verified
# train2-2 model (mAP50 0.916).
#
# NOTE (imaandari): Detection asal YOLO model se hoti hai. Verification ka
# Siamese model TRAIN HO CHUKA hai (notebook: Siemes_Network_Model.ipynb,
# test accuracy 96.7%) — lekin us ki weights file (siamese.weights.h5) is
# folder me ho tabhi verification chalti hai (siamese.py dekhein, saath me
# TensorFlow chahiye: requirements-verification.txt). Weights file na ho
# to verification tasks contract ke tareeqay se "failed" hote hain saaf
# error ke saath (code: VERIFICATION_MODEL_NOT_CONNECTED). Koi nakli
# score/verdict kabhi nahi banta.

import base64
import hashlib
import hmac
import io
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from pydantic import BaseModel

import siamese

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ANNOTATED_DIR = os.path.join(BASE_DIR, "annotated")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

MODEL_CANDIDATES = [
    "signature_model.pt",
    "combined_best.pt",
    "document_image_best.pt",
    "signature_closeup_best.pt",
    "best.pt",
]
EXPECTED_MODEL_SIZE = 40593964  # train2-2 wale verified model ka size (bytes)

DB_PATH = os.path.join(BASE_DIR, "app.db")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
TOKEN_HOURS = 24

MAX_DOC_BYTES = 10 * 1024 * 1024  # detection document: 10 MB (contract)
MAX_SIG_BYTES = 5 * 1024 * 1024  # verification signatures: 5 MB (contract)
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

app = FastAPI(
    title="Signature System API",
    description="Signature_System frontend ke contract (BACKEND_INTEGRATION.md) par bana backend.",
    version="1.0",
)

# CORS — frontend (Vite, port 5173) se calls ke liye (contract section 2)
# Live frontend (masalan Vercel) ke liye us ka URL ALLOWED_ORIGINS env var
# me dein, comma se alag karke — masalan:
#   ALLOWED_ORIGINS=https://signature-system.vercel.app
# Na dein to sirf localhost wale defaults chalte hain (local demo waisa
# ka waisa). "*" kabhi na dein — login wala API hai.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
] or ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Database (SQLite default / PostgreSQL optional) -------------
# DATABASE_URL na ho to purana SQLite (app.db) hi chalta hai — kuch tabdeel
# nahi, purana data mehfooz. DATABASE_URL ho to saara data (users, tasks,
# results) PostgreSQL me save hota hai (Neon waghera ki free cloud database).
# Tareeqa: is folder me ek .env file banayein, us me ek line:
#   DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
# (.env.example dekhein. .env APNI file hai — is me password hota hai, ye
# file kisi ko bhejni/share nahi hai.)


def _load_env_file():
    path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()


def _clean_db_url(url: str) -> str:
    """DATABASE_URL ki aam chhoti ghaltiyan khud theek kar dein:
    ird-gird lage nishan (< > aur quotes), dohra 'DATABASE_URL=' prefix,
    aur channel_binding wala hissa (kuch systems par wo connection rok
    deta hai — Neon baghair is ke bhi bilkul theek chalta hai)."""
    u = (url or "").strip().strip('<>"\'').strip()
    if u.lower().startswith("database_url="):
        u = u[len("database_url=") :].strip().strip('<>"\'').strip()
    if "channel_binding=" in u:
        u = re.sub(r"[&?]channel_binding=[^&]*", "", u)
        u = u.replace("?&", "?")
        if "?" not in u and "&" in u:
            u = u.replace("&", "?", 1)
    return u


DATABASE_URL = _clean_db_url(os.environ.get("DATABASE_URL", ""))
USE_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))

if USE_PG:
    import psycopg
    from psycopg.rows import dict_row

_INTEGRITY_ERRORS = (sqlite3.IntegrityError,)
if USE_PG:
    _INTEGRITY_ERRORS = _INTEGRITY_ERRORS + (psycopg.IntegrityError,)


class _Cur:
    """Dono backends ke cursors ke liye ek jaisa wrapper."""

    def __init__(self, cur):
        self._cur = cur

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def lastrowid(self):
        return getattr(self._cur, "lastrowid", None)


class _Conn:
    """Ek jaisa chhota interface: execute(sql, params) / commit() / close().
    SQL har jagah '?' placeholder se likha hai — PostgreSQL ke liye ye
    wrapper khud use '%s' me badal deta hai."""

    def __init__(self):
        if USE_PG:
            self._con = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        else:
            self._con = sqlite3.connect(DB_PATH)
            self._con.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        if USE_PG:
            sql = sql.replace("?", "%s")
        return _Cur(self._con.execute(sql, tuple(params)))

    def commit(self):
        self._con.commit()

    def close(self):
        self._con.close()


def db():
    return _Conn()


def init_db():
    con = db()
    if USE_PG:
        con.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "id SERIAL PRIMARY KEY, "
            "email TEXT UNIQUE NOT NULL, "
            "full_name TEXT DEFAULT '', "
            "password_hash TEXT NOT NULL)"
        )
    else:
        con.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "email TEXT UNIQUE NOT NULL, "
            "full_name TEXT DEFAULT '', "
            "password_hash TEXT NOT NULL)"
        )
    con.execute("CREATE TABLE IF NOT EXISTS revoked_tokens (token TEXT PRIMARY KEY)")
    con.execute(
        "CREATE TABLE IF NOT EXISTS tasks ("
        "id TEXT PRIMARY KEY, "
        "module TEXT NOT NULL, "
        "username TEXT NOT NULL, "
        "status TEXT NOT NULL, "
        "message TEXT, "
        "created_at TEXT NOT NULL, "
        "result TEXT, "
        "error TEXT)"
    )
    con.commit()
    con.close()


init_db()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------- Passwords + JWT (stdlib) ----------------


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return salt.hex() + "$" + dk.hex()


def check_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), 100_000
        )
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(email: str) -> str:
    header = _b64e(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64e(
        json.dumps({"sub": email, "exp": int(time.time()) + TOKEN_HOURS * 3600}).encode()
    )
    sig = hmac.new(
        SECRET_KEY.encode(), f"{header}.{payload}".encode(), hashlib.sha256
    ).digest()
    return f"{header}.{payload}.{_b64e(sig)}"


def read_token(token: str):
    try:
        header, payload, sig = token.split(".")
        expected = hmac.new(
            SECRET_KEY.encode(), f"{header}.{payload}".encode(), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(_b64e(expected), sig):
            return None
        data = json.loads(_b64d(payload))
        if data.get("exp", 0) < int(time.time()):
            return None
        return data.get("sub")
    except Exception:
        return None


from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)):
    con = db()
    revoked = con.execute(
        "SELECT token FROM revoked_tokens WHERE token = ?", (token,)
    ).fetchone()
    row = None
    email = None if revoked else read_token(token)
    if email:
        row = con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return row


def user_view(row) -> dict:
    return {"id": row["id"], "email": row["email"], "full_name": row["full_name"]}


# ---------------- Auth endpoints ----------------


class RegisterIn(BaseModel):
    email: str
    password: str
    full_name: str = ""


class LoginIn(BaseModel):
    email: str
    password: str


@app.post("/api/v1/auth/register")
def register(body: RegisterIn):
    """Naya user banayein (frontend ise call nahi karta — pehla user banane
    ke liye /docs ya script se istemal karein)."""
    email = body.email.strip().lower()
    if "@" not in email or len(body.password) < 4:
        raise HTTPException(
            status_code=400,
            detail="Sahi email dein aur password kam az kam 4 characters ka ho.",
        )
    con = db()
    try:
        if USE_PG:
            row = con.execute(
                "INSERT INTO users (email, full_name, password_hash) "
                "VALUES (?, ?, ?) RETURNING id",
                (email, body.full_name.strip(), hash_password(body.password)),
            ).fetchone()
            uid = row["id"]
        else:
            cur = con.execute(
                "INSERT INTO users (email, full_name, password_hash) VALUES (?, ?, ?)",
                (email, body.full_name.strip(), hash_password(body.password)),
            )
            uid = cur.lastrowid
        con.commit()
    except _INTEGRITY_ERRORS:
        raise HTTPException(status_code=400, detail="Ye email pehle se registered hai.")
    finally:
        con.close()
    return {"id": uid, "email": email, "full_name": body.full_name.strip()}


@app.post("/api/v1/auth/login")
def login(body: LoginIn):
    """Contract: JSON {email, password} -> {access_token, token_type, user}."""
    email = body.email.strip().lower()
    con = db()
    row = con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    con.close()
    if not row or not check_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return {
        "access_token": make_token(email),
        "token_type": "bearer",
        "user": user_view(row),
    }


@app.get("/api/v1/auth/me")
def me(current_user=Depends(get_current_user)):
    return user_view(current_user)


@app.post("/api/v1/auth/logout")
def logout(
    token: str = Depends(oauth2_scheme), current_user=Depends(get_current_user)
):
    con = db()
    if USE_PG:
        con.execute(
            "INSERT INTO revoked_tokens (token) VALUES (?) "
            "ON CONFLICT (token) DO NOTHING",
            (token,),
        )
    else:
        con.execute("INSERT OR IGNORE INTO revoked_tokens (token) VALUES (?)", (token,))
    con.commit()
    con.close()
    return {"message": "Logged out."}


# ---------------- Detection model ----------------

_model = None


def find_model_path():
    env = os.environ.get("MODEL_PATH")
    if env:
        return env
    for name in MODEL_CANDIDATES:
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, MODEL_CANDIDATES[0])


def get_model():
    global _model
    if _model is None:
        path = find_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Model file nahi mili — backend folder me in naamo me se ek "
                "file rakhein: " + ", ".join(MODEL_CANDIDATES)
            )
        size = os.path.getsize(path)
        print(f"[model] {path} | size: {size} bytes")
        if size == EXPECTED_MODEL_SIZE:
            print("[model] Size match — ye hamari verified train2-2 model file hai.")
        from ultralytics import YOLO

        _model = YOLO(path)
    return _model


def predict_on_image(img: Image.Image):
    """Ek image par model chalain -> (detections list, annotated PIL image)."""
    tmp = os.path.join(UPLOAD_DIR, "_predict_tmp.png")
    img.save(tmp)
    model = get_model()
    res = model.predict(tmp, conf=0.25, imgsz=1024, verbose=False)[0]
    detections = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        detections.append(
            {
                "confidence": round(float(b.conf[0]), 3),
                "bounding_box": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
            }
        )
    annotated = Image.fromarray(res.plot()[:, :, ::-1])
    return detections, annotated


def load_document_pages(path: str, ext: str):
    """Document ko pages ki images me badlein. PDF ke liye PyMuPDF (fitz)."""
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PDF support ke liye PyMuPDF install karein: pip install PyMuPDF"
            )
        pages = []
        doc = fitz.open(path)
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        doc.close()
        return pages
    return [Image.open(path).convert("RGB")]


# ---------------- Task helpers ----------------


def create_task(module: str, username: str, message: str) -> str:
    prefix = "det" if module == "detection" else "ver"
    task_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
    con = db()
    con.execute(
        "INSERT INTO tasks (id, module, username, status, message, created_at) "
        "VALUES (?, ?, ?, 'queued', ?, ?)",
        (task_id, module, username, message, now_iso()),
    )
    con.commit()
    con.close()
    return task_id


def update_task(task_id: str, **fields):
    con = db()
    sets = ", ".join(f"{k} = ?" for k in fields)
    con.execute(f"UPDATE tasks SET {sets} WHERE id = ?", (*fields.values(), task_id))
    con.commit()
    con.close()


def task_detail(row) -> dict:
    out = {"task_id": row["id"], "status": row["status"], "created_at": row["created_at"]}
    if row["message"]:
        out["message"] = row["message"]
    if row["status"] == "completed" and row["result"]:
        out["result"] = json.loads(row["result"])
    if row["status"] == "failed" and row["error"]:
        out["error"] = json.loads(row["error"])
    return out


def fail_task(task_id: str, code: str, message: str):
    update_task(
        task_id,
        status="failed",
        message=None,
        error=json.dumps({"code": code, "message": message}),
    )


def get_owned_task(module: str, task_id: str, username: str):
    con = db()
    row = con.execute(
        "SELECT * FROM tasks WHERE id = ? AND module = ? AND username = ?",
        (task_id, module, username),
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found.")
    return row


def list_tasks(module: str, username: str, page: int, page_size: int, extra_key: str):
    con = db()
    total = con.execute(
        "SELECT COUNT(*) AS c FROM tasks WHERE module = ? AND username = ?",
        (module, username),
    ).fetchone()["c"]
    rows = con.execute(
        "SELECT * FROM tasks WHERE module = ? AND username = ? "
        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (module, username, page_size, (page - 1) * page_size),
    ).fetchall()
    con.close()
    items = []
    for r in rows:
        item = {"task_id": r["id"], "status": r["status"], "created_at": r["created_at"]}
        if r["result"]:
            try:
                res = json.loads(r["result"])
                if extra_key == "signature_count":
                    item["signature_count"] = res.get("signature_count")
                else:
                    item["similarity_score"] = res.get("similarity_score")
                    item["verdict"] = res.get("verdict")
            except Exception:
                pass
        if extra_key == "similarity_score" and "similarity_score" not in item:
            item["similarity_score"] = None
        items.append(item)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


async def save_upload(file: UploadFile, max_bytes: int, allowed_exts, label: str) -> str:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"{label}: sirf {', '.join(sorted(allowed_exts))} files chalti hain.",
        )
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"{label}: file {max_bytes // (1024 * 1024)} MB se bari hai.",
        )
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return path


# ---------------- Detection module ----------------


def process_detection(task_id: str, doc_path: str, ext: str, base_url: str):
    update_task(task_id, status="processing", message="Detecting signatures...")
    try:
        pages = load_document_pages(doc_path, ext)
        if not pages:
            raise RuntimeError("Document se koi page nahi mila.")
        all_detections = []
        first_size = None
        for i, page_img in enumerate(pages, start=1):
            dets, annotated = predict_on_image(page_img)
            if i == 1:
                first_size = page_img.size
                annotated.save(os.path.join(ANNOTATED_DIR, f"{task_id}.jpg"), "JPEG")
            for d in dets:
                d["page"] = i
                all_detections.append(d)
        for n, d in enumerate(all_detections, start=1):
            d_ordered = {"signature_id": n, **d}
            all_detections[n - 1] = d_ordered
        result = {
            "signature_count": len(all_detections),
            "detections": all_detections,
            "image_width": first_size[0],
            "image_height": first_size[1],
            "annotated_image_url": f"{base_url}api/v1/files/annotated/{task_id}.jpg",
        }
        update_task(
            task_id, status="completed", message=None, result=json.dumps(result)
        )
    except FileNotFoundError as e:
        fail_task(task_id, "MODEL_NOT_AVAILABLE", str(e))
    except Exception as e:
        fail_task(task_id, "PROCESSING_ERROR", f"Unable to process the uploaded file. ({e})")


@app.post("/api/v1/signature-detection")
async def submit_detection(
    background: BackgroundTasks,
    request: Request,
    document: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    ext = os.path.splitext(document.filename or "")[1].lower()
    path = await save_upload(
        document, MAX_DOC_BYTES, IMAGE_EXTS | {".pdf"}, "document"
    )
    task_id = create_task(
        "detection", current_user["email"], "Detection request submitted successfully"
    )
    base_url = str(request.base_url)
    background.add_task(process_detection, task_id, path, ext, base_url)
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Detection request submitted successfully",
    }


@app.get("/api/v1/signature-detection/{task_id}")
def get_detection(task_id: str, current_user=Depends(get_current_user)):
    return task_detail(get_owned_task("detection", task_id, current_user["email"]))


@app.get("/api/v1/signature-detection")
def detection_history(
    page: int = 1, page_size: int = 10, current_user=Depends(get_current_user)
):
    return list_tasks(
        "detection", current_user["email"], page, page_size, "signature_count"
    )


@app.get("/api/v1/files/annotated/{task_id}.jpg")
def annotated_file(task_id: str):
    """Annotated image — bina login ke khulti hai (contract: <img> ke liye)."""
    path = os.path.join(ANNOTATED_DIR, f"{task_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Annotated image not found.")
    return FileResponse(path, media_type="image/jpeg")


# ---------------- Verification module ----------------


def process_verification(task_id: str, original_path: str, suspected_path: str):
    update_task(task_id, status="processing", message="Verifying signatures...")
    # Weights file (siamese.weights.h5) na ho to wahi purani imandari wali
    # failed status — nakli score kabhi nahi banta.
    if not siamese.weights_present():
        fail_task(
            task_id,
            "VERIFICATION_MODEL_NOT_CONNECTED",
            "The verification model is not connected yet. Only signature "
            "detection is available at this time.",
        )
        return
    try:
        result = siamese.verify_pair(original_path, suspected_path)
    except RuntimeError as e:
        fail_task(task_id, "VERIFICATION_MODEL_LOAD_FAILED", str(e))
        return
    except Exception as e:
        fail_task(
            task_id, "PROCESSING_ERROR", f"Unable to verify the signatures. ({e})"
        )
        return
    update_task(
        task_id, status="completed", message=None, result=json.dumps(result)
    )


@app.post("/api/v1/signature-verification")
async def submit_verification(
    background: BackgroundTasks,
    original_signature: UploadFile = File(...),
    suspected_signature: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    original_path = await save_upload(
        original_signature, MAX_SIG_BYTES, IMAGE_EXTS, "original_signature"
    )
    suspected_path = await save_upload(
        suspected_signature, MAX_SIG_BYTES, IMAGE_EXTS, "suspected_signature"
    )
    task_id = create_task(
        "verification",
        current_user["email"],
        "Verification request submitted successfully",
    )
    background.add_task(process_verification, task_id, original_path, suspected_path)
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Verification request submitted successfully",
    }


@app.get("/api/v1/signature-verification/{task_id}")
def get_verification(task_id: str, current_user=Depends(get_current_user)):
    return task_detail(get_owned_task("verification", task_id, current_user["email"]))


@app.get("/api/v1/signature-verification")
def verification_history(
    page: int = 1, page_size: int = 10, current_user=Depends(get_current_user)
):
    return list_tasks(
        "verification", current_user["email"], page, page_size, "similarity_score"
    )


# ---------------- Home / health ----------------


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html><body style="font-family: sans-serif; margin: 40px;">
  <h2>Signature System API</h2>
  <p>Backend chal raha hai. Contract: frontend ke BACKEND_INTEGRATION.md ke mutabiq.
  Test ke liye <a href="/docs"><b>/docs</b></a> kholein.</p>
  <ul>
    <li>POST /api/v1/auth/login · GET /api/v1/auth/me · POST /api/v1/auth/logout</li>
    <li>POST /api/v1/signature-detection · GET /api/v1/signature-detection/{task_id} · GET /api/v1/signature-detection</li>
    <li>POST /api/v1/signature-verification · GET /api/v1/signature-verification/{task_id} · GET /api/v1/signature-verification</li>
  </ul>
</body></html>"""


@app.get("/health")
def health():
    p = find_model_path()
    return {
        "status": "ok",
        "model_file_expected": os.path.basename(p),
        "model_file_present": os.path.exists(p),
        "verification_weights_present": siamese.weights_present(),
        "database": "postgresql" if USE_PG else "sqlite",
    }
