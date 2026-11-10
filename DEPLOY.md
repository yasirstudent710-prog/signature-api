# Signature System — LIVE karne ka plan (frontend + backend)

> Pehle ye samajh lein: **live** ka matlab hai ke system internet par kisi
> ko bhi link se khule — sirf aap ke PC par nahi. Is ke **do hisse** hain:
> **frontend** (React — asan, muft) aur **backend** (FastAPI + YOLO +
> TensorFlow — bhaari, is liye soch kar).

## STEP 0 — Live se PEHLE, ye do kaam (aaj hi)

**1. Verification test (Step 8) pehle mukammal karein.**
Localhost par system 100% chalta sabit ho, phir live ki bari. Frontend
(`http://localhost:5173`) → login → **Verification** page → genuine jori
(`original_1_1.png` + `original_1_2.png`, dono `Desktop\signature\
signatures_labeled_5000\original` se) → Submit. Score zyada + verdict
**genuine** ana chahiye. Pehli dafa model load me 1 minute lag sakta hai.

**2. Neon password reset — live se PEHLE, baad me nahi.**
Purana Neon password subah chat me aa gaya tha aur ab bhi chal raha hai.
Live backend usi password se chalega to khatra hoga. Is liye **pehle**
reset karein (neon.com → project `my-first-project` → **Roles** →
`neondb_owner` → **Reset password**), phir **nayi connection string**
Notepad me save karein — **chat me mat bhejna**. Server band karke
`.env` badal kar dobara chalain aur `127.0.0.1:8000/health` par
`"database": "postgresql"` dekh kar tasdeeq karein. **Nayi** string hi
live wale Step 2 me Secret banegi — purani wali kabhi nahi.

## STEP 1 — GitHub login (aap ka apna step, is ke baghair live nahi)

Live deployment ke liye code **GitHub** par jana hoga. Ye login main aap
ki jagah nahi kar sakta. GitHub par do repositories (ya ek repo, do
folders) banayein:

- `signature-api` — is folder ki files (neeche wali list)
- `Signature_System-main` — frontend (React) wali files

**`signature-api` me ye files honi chahiye** (sab is waqt tayyar hain):
`main.py`, `siamese.py`, `requirements.txt`,
`requirements-verification.txt`, `test_api.py`, `README.md`,
`.env.example`, `Dockerfile`, `.dockerignore`, `.gitignore`,
`siamese.weights.h5` (1,358,408 bytes — zip me pehle se hai) aur
**`signature_model.pt` (40,593,964 bytes — sirf aap ke PC par hai,
is zip me nahi; use alag se upload karna hoga)**.

**KABHI GitHub par mat dalna:** `.env` (is me Neon password hai),
`app.db`, `uploads/`, `annotated/` — `.gitignore` ye khud rokta hai.

## STEP 2 — Backend live: Hugging Face Spaces (Docker)

**Imaandari wali baat (nayi maloomat):** pehle ye samjha tha ke HF Spaces
Docker muft hai — HF ki **aaj ki docs** kehti hain ke Docker/Gradio
Spaces banane ke liye **paid plan chahiye** (cpu-basic hardware ki hourly
cost phir bhi FREE hai, aur free account par 2 Gradio Spaces ZeroGPU par
chal sakte hain). Is liye Space banate waqt agar payment/PRO mange to
hairan na hon — **pehle check karein, phir faisla karein**. Agar paid
nahi karna to dusra rasta Railway/Render ka **paid plan (~$5–7 mahina)**
hai. **Render ka free plan (512 MB RAM) is backend ke liye nahi chalega**
— YOLO + TensorFlow dono ek saath is me nahi samate.

Tareeqa (Docker Space):
1. [huggingface.co/new-space](https://huggingface.co/new-space) → Space
   ka naam dein → **SDK: Docker** chunain → **Create** (Private rakhain
   to behtar hai — model files andar hain).
2. Space ke README.md ke sab se upar ye 4 lines hon (Space banate hi
   HF khud likh deta hai — na hon to khud likh dein):
   ```yaml
   ---
   title: Signature System API
   sdk: docker
   app_port: 7860
   ---
   ```
3. Apne `signature-api` wali files Space repo me push/upload karein —
   **aur `signature_model.pt` bhi** (Space ki Files wali screen se
   upload kar sakte hain; size check: **40,593,964 bytes** hona chahiye —
   ghalat file (825,011 KB wali) kabhi nahi).
4. Space ke **Settings → Secrets** me ye 3 secrets dalain (ye **kabhi
   repo/chat me nahi**):
   - `DATABASE_URL` = nayi Neon connection string (Step 0 wali, pooling OFF)
   - `SECRET_KEY` = koi lambi random line (apni marzi se likh lein —
     localhost wala `dev-secret-key-change-me` live par nahi chalega)
   - `ALLOWED_ORIGINS` = frontend ka URL — **ye Step 3 ke baad milega**,
     is liye pehle backend banayein, phir frontend, phir wapas aakar ye
     secret dal kar Space **restart** karein.
5. Build khatam ho to Space ka link kholein — jaise
   `https://huggingface.co/spaces/<aapka-naam>/<space-naam>` — aur
   `/health` check karein: `https://<space-link>/health` par
   `model_file_present: true`, `verification_weights_present: true`,
   `"database": "postgresql"` — teeno `true` hone chahiye.
   **Pehli build me 10–20 minute lag sakte hain** (TensorFlow + torch
   bari downloads hain) — ghabrana nahi.

## STEP 3 — Frontend live: Vercel (muft)

1. [vercel.com](https://vercel.com) → GitHub se login → **Add New →
   Project** → `Signature_System-main` wali repo chunain → **Deploy**.
2. Deploy se **pehle**, Vercel ke **Environment Variables** me ye ek
   line dalain (yehi frontend ko live backend se jorta hai):
   - `VITE_API_BASE_URL` = `https://<aapka-space-link>` (Step 2 wala,
     aakhir me `/` nahi)
   - Local `.env` me abhi `http://localhost:8000` likha hai — Vercel
     wali value build ke waqt us par **override** kar deti hai, file
     badalne ki zaroorat nahi.
3. Deploy ho jaye to Vercel aap ko link dega, masalan
   `https://signature-system.vercel.app` — **yehi URL Step 2 ke
   `ALLOWED_ORIGINS` secret me dalna hai**, phir Space restart karein.
   Us ke baad hi frontend live backend se baat kar payega (warna CORS
   use rok dega — yehi wajah hai ke `main.py` me CORS ab `ALLOWED_ORIGINS`
   env var se parhta hai, localhost par pehle jaisa hi chalta hai).

## STEP 4 — Live par aakhri check

1. Vercel link kholein → **login** karein. **Neon users localhost aur
   live dono par ek hi hain** (dono ek hi Neon database use karte hain),
   is liye wohi email/password chalega jo localhost par banaya tha —
   agar "incorrect password" aye to live site par dobara **register**
   karein.
2. **Detection** page par ek signed document upload karein, phir
   **Verification** page par genuine jori — dono ke results live par
   bhi ane chahiye.
3. Bas — ab link kisi ko bhi bheja ja sakta hai.

## Imaandari wali hadain (live par ye farq hoga)

- **Free tier par Space sota hai:** kuch dair koi na khole to Space
  band ho jata hai — pehli khulne me 1–2 minute lag sakte hain, aur
  pehli verification me model load ka waqt alag.
- **Uploaded/annotated files mehfooz nahi rehti:** Space ki disk
  restart par saaf ho jati hai — `uploads/` aur `annotated/` wali
  tasaveer gayab ho jayengi. **History aur users mehfooz rehte hain**
  (wo Neon PostgreSQL me hain), sirf files dobara upload karni hongi.
- **Pehli dafa model load hone me waqt lagta hai** — localhost jaisa
  tez nahi hoga.
- Live hone ke baad bhi **koi bhi shakhs sirf link se** system khol
  sakta hai, lekin **login ke baghair andar nahi ja sakta** — register
  wala rasta khula hai, is liye apne team ke logon ko hi link dein.

## Is package me kya badla (10 Nov, dopahar)

- `main.py` — CORS ab `ALLOWED_ORIGINS` env var se parhta hai
  (default ab bhi localhost:5173 — local demo waisa ka waisa).
- `Dockerfile` — naya (Python 3.12, stable TensorFlow, port 7860).
- `.dockerignore` / `.gitignore` — naye (`.env`, `app.db`, uploads
  kabhi live/GitHub par nahi jayenge).
- `DEPLOY.md` — ye file (poora plan).
- Kuch bhi delete ya replace nahi hua — `signature_model.pt`
  (39,643 KB) aur `.env` ko haath nahi laga.
