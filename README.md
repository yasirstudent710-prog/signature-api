# Signature System — Backend (FastAPI)

Ye backend **Signature_System frontend** ke `BACKEND_INTEGRATION.md` wale
contract par bana hai — endpoints, upload field names aur response shapes
bilkul wohi jo frontend call karta hai. Andar trained YOLO11m detection
model (mAP50 **0.916**) chalta hai.

> Pehle wala combined design (`POST /api/v1/signature-verifications`,
> document + reference) frontend ab **nahi** bulata — us ki jagah detection
> aur verification ke **alag alag** route groups hain, jaise is build me hain.

## Endpoints (contract)

| Method | Endpoint | Kaam |
|---|---|---|
| POST | `/api/v1/auth/login` | JSON `{email, password}` → `{access_token, token_type, user}` |
| GET | `/api/v1/auth/me` | Logged-in user `{id, email, full_name}` |
| POST | `/api/v1/auth/logout` | Logout — token khatam |
| POST | `/api/v1/auth/register` | **Extra** — naya user (frontend call nahi karta; pehla user banane ke liye) |
| POST | `/api/v1/signature-detection` | `document` file (pdf/png/jpg, 10 MB tak) → foran `task_id` |
| GET | `/api/v1/signature-detection/{task_id}` | Status poll karein — completed par `result` (signature_count, detections bounding_box pixels me, annotated_image_url) |
| GET | `/api/v1/signature-detection` | History — `{items, total, page, page_size}` |
| POST | `/api/v1/signature-verification` | `original_signature` + `suspected_signature` (5 MB tak) → `task_id` |
| GET | `/api/v1/signature-verification/{task_id}` | Status / result |
| GET | `/api/v1/signature-verification` | History |

Task lifecycle: `queued → processing → completed / failed`
(failed par `error: {code, message}` ata hai — contract section 4/7).
Model ka kaam **background** me hota hai; POST request rukti nahi.

CORS frontend origin (`http://localhost:5173`) ke liye on hai, aur
annotated image wali file bina login ke khulti hai (contract ki shart).

## Model file

Backend folder me team ki final model file rakhein — code in naamo me se
jo mile khud utha lega: `signature_model.pt`, `combined_best.pt`,
`document_image_best.pt`, `signature_closeup_best.pt`, `best.pt`
(ya `MODEL_PATH` env var). Pehli detection par terminal me size print
hota hai — **40,593,964 bytes** wali hamari verified train2-2 model hai.
PDF documents ke liye **PyMuPDF** chahiye (requirements me shamil hai).

## Chalane ka tareeqa

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
> `python -m` zaroori hai — PC par Python ka Scripts folder PATH me nahi hai.

**Pehla user banana** (ek hi dafa): http://127.0.0.1:8000/docs par
`/api/v1/auth/register` se apna email/password register karein — phir
frontend (http://localhost:5173) par usi email se login hoga.
Frontend ka `.env` default `VITE_API_BASE_URL=http://localhost:8000` hai,
wohi port jo upar hai — kuch badalna nahi.

Poora contract test (script):

```bash
python test_api.py gold_test_page.jpg
```

## Verification ke baray me (imaandari wali baat)

Detection asal model se hoti hai. **Verification wala Siamese model TRAIN
HO CHUKA hai** (notebook `Siemes_Network_Model.ipynb` — 5,000 signatures,
100 persons, person-wise split, **test accuracy 96.7%**). Verification
chalane ke liye do cheezein chahiye:

1. **`siamese.weights.h5`** is folder me rakhein (Colab training ke baad
   wahan save hoti hai — Drive se download karke yahan le ayen)
2. TensorFlow install karein: `pip install -r requirements-verification.txt`
   (bari download hai, one-time; baqi API is ke baghair bhi chalti hai)

Dono ho jayen to verification tasks asal result dete hain:
`similarity_score` (0–100) + `verdict` — score **≥55 genuine**, **≤45
forged**, darmiyan **inconclusive** (API-side band, `siamese.py` me ek
line se badal sakta hai). Weights file na ho to verification purani tarah
`failed` hoti hai saaf error ke saath: `VERIFICATION_MODEL_NOT_CONNECTED`.
Koi nakli score ya verdict **nahi** banta.

## Database — SQLite ya PostgreSQL

Default: **SQLite** (`app.db` isi folder me) — users, tasks aur results
isi PC par save hote hain. Koi setting nahi chahiye.

**PostgreSQL par le jana ho to** (masalan Neon ki free cloud database):

1. [neon.com](https://neon.com) par account banayein → naya project
   banayein → Dashboard me **Connect** se **connection string** copy karein
2. Is folder me `.env` naam ki file banayein (`.env.example` copy karke)
   aur us me ek line likhein:
   `DATABASE_URL=postgresql://...` (copy ki hui string)
   — **ye file kisi ko bhejni/upload nahi karni, is me password hota hai**
3. `pip install -r requirements.txt` (psycopg shamil hai) aur server
   restart karein — tables Postgres me khud ban jati hain

Tasdeeq: `http://127.0.0.1:8000/health` par `"database": "postgresql"`
ana chahiye (warna `"sqlite"`). **Purana SQLite data Postgres me khud
transfer nahi hota** — users ko naye database me dobara register karna
hoga. Wapas SQLite par jana ho to `.env` se line hata kar restart kar
dein; purana `app.db` waisa ka waisa mehfooz rahega. Uploaded files aur
annotated images har surat me isi folder (`uploads/`, `annotated/`) me
rehti hain — database me sirf records (users/tasks/results) jate hain.

## Files

- `main.py` — poori API (auth + detection + verification tasks)
- `siamese.py` — verification model ka loader (weights: `siamese.weights.h5`)
- `requirements.txt` — libraries (PyMuPDF + PostgreSQL ke liye psycopg)
- `requirements-verification.txt` — sirf verification ke liye TensorFlow
- `test_api.py` — contract ka poora flow test
- `.env.example` — PostgreSQL connection string ki file ka namuna
- `app.db` — SQLite mode me khud banti hai (users + tasks) · `uploads/`, `annotated/` — khud bante hain
