from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid, time, os, stripe
from google import genai
from google.genai import types
from collections import defaultdict
from jose import jwt as jose_jwt, JWTError
from datetime import datetime, timedelta

app = FastAPI(title="Lingi Coach API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY", "")
STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_MONTHLY_PRICE  = os.environ.get("STRIPE_MONTHLY_PRICE", "")
STRIPE_YEARLY_PRICE   = os.environ.get("STRIPE_YEARLY_PRICE", "")
JWT_SECRET            = os.environ.get("JWT_SECRET", "change-in-production")
FRONTEND_URL          = os.environ.get("FRONTEND_URL", "http://localhost:3000")

MODEL = "gemini-3.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)
stripe.api_key = STRIPE_SECRET_KEY

sessions: dict = {}
rate_limits: dict = defaultdict(list)
RATE_LIMIT = 30
SESSION_TTL = 1800

SUPPORTED_LANGUAGES = {
    "bn": "Bengali", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "ur": "Urdu", "ar": "Arabic", "sw": "Swahili",
    "es": "Spanish", "pt": "Portuguese", "fr": "French", "id": "Indonesian",
    "tr": "Turkish", "vi": "Vietnamese", "th": "Thai", "ms": "Malay",
    "tl": "Filipino", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "it": "Italian", "nl": "Dutch", "pl": "Polish",
}


def check_rate(ip: str) -> bool:
    now = time.time()
    rate_limits[ip] = [t for t in rate_limits[ip] if now - t < 60]
    if len(rate_limits[ip]) >= RATE_LIMIT:
        return False
    rate_limits[ip].append(now)
    return True


def clean_sessions():
    now = time.time()
    for sid in [k for k, v in sessions.items() if now - v["last"] > SESSION_TTL]:
        del sessions[sid]


def issue_jwt(email: str, plan: str) -> str:
    days = 366 if plan == "yearly" else 32
    payload = {
        "sub": email, "plan": plan, "premium": True,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=days),
    }
    return jose_jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_jwt(token: str) -> Optional[dict]:
    try:
        return jose_jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        return None


def build_learn_prompt(native_language: str, scenario_title: str,
                       ai_role: str, situation: str) -> str:
    return f"""You are Lingi Coach, a warm and patient AI English speaking coach.

## MANDATORY — EU AI ACT ARTICLE 50
You are an artificial intelligence (Google Gemini). Never claim to be human.

## YOUR ROLE
You are playing: {ai_role}
Scenario: {scenario_title}
Situation: {situation}

## LANGUAGE RULES
- All your setup, instructions, corrections, encouragement: in {native_language}.
- During roleplay you speak English (as the {ai_role} would in real life).
- After each student attempt, step out of character briefly to correct in {native_language}.
- Then continue the roleplay in English.

## FLOW
1. Set the scene in {native_language} — one or two sentences max.
2. Start roleplay immediately as {ai_role}, speaking English.
3. Student responds:
   - Correct → brief praise in {native_language}, continue.
   - Almost correct → name error in {native_language}, write **correct version in bold**, continue.
   - Wrong → correct in {native_language}, write **correct version in bold**, encourage, continue.
4. Keep responses under 80 words. No lectures.

## PRIVACY
Never ask for personal information."""


def build_translate_prompt(from_lang: str, to_lang: str) -> str:
    return f"""You are a professional interpreter between {from_lang} and {to_lang}.
If the message is in {from_lang}, translate to {to_lang}.
If the message is in {to_lang}, translate to {from_lang}.
Return ONLY the translation. No labels, no preamble."""


# ── Pydantic models ──────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    native_language_code: str
    scenario_id: str
    scenario_title: str
    ai_role: str
    situation: str
    consent_given: bool
    age_verified: bool

class ChatRequest(BaseModel):
    session_id: str
    message: str

class TranslateRequest(BaseModel):
    text: str
    from_lang_code: str
    to_lang_code: str

class CheckoutRequest(BaseModel):
    plan: str
    email: str

class VerifyTokenRequest(BaseModel):
    token: str

class SessionResponse(BaseModel):
    session_id: str
    message: str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "gdpr": True, "eu_ai_act": True}

@app.get("/languages")
def languages():
    return {"languages": [{"code": k, "name": v} for k, v in SUPPORTED_LANGUAGES.items()]}


@app.post("/session/start", response_model=SessionResponse)
async def start_session(req: StartRequest, request: Request):
    if not check_rate(request.client.host):
        raise HTTPException(429, "Too many requests.")
    if not req.consent_given or not req.age_verified:
        raise HTTPException(400, "Consent required.")
    if req.native_language_code not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, "Unsupported language.")
    clean_sessions()

    native = SUPPORTED_LANGUAGES[req.native_language_code]
    sid = str(uuid.uuid4())
    system_prompt = build_learn_prompt(native, req.scenario_title, req.ai_role, req.situation)

    try:
        # Start chat with system instruction
        chat = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        resp = chat.send_message("Begin the session now.")
        sessions[sid] = {
            "chat": chat,
            "lang": req.native_language_code,
            "last": time.time(),
            "count": 1,
        }
        return SessionResponse(session_id=sid, message=resp.text)
    except Exception as e:
        raise HTTPException(500, f"Failed to start session: {str(e)}")


@app.post("/session/chat", response_model=SessionResponse)
async def chat_message(req: ChatRequest, request: Request):
    if not check_rate(request.client.host):
        raise HTTPException(429, "Rate limit.")
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found.")
    s = sessions[req.session_id]
    if time.time() - s["last"] > SESSION_TTL:
        del sessions[req.session_id]
        raise HTTPException(410, "Session expired.")
    if s["count"] >= 100:
        raise HTTPException(400, "Session limit reached.")
    msg = (req.message or "").strip()
    if not msg or len(msg) > 1000:
        raise HTTPException(400, "Invalid message.")
    try:
        resp = s["chat"].send_message(msg)
        s["last"] = time.time()
        s["count"] += 1
        return SessionResponse(session_id=req.session_id, message=resp.text)
    except Exception as e:
        raise HTTPException(500, f"AI error: {str(e)}")


@app.post("/translate")
async def translate(req: TranslateRequest, request: Request):
    if not check_rate(request.client.host):
        raise HTTPException(429, "Rate limit.")
    from_lang = SUPPORTED_LANGUAGES.get(req.from_lang_code, req.from_lang_code)
    to_lang   = SUPPORTED_LANGUAGES.get(req.to_lang_code, req.to_lang_code)
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=req.text,
            config=types.GenerateContentConfig(
                system_instruction=build_translate_prompt(from_lang, to_lang),
            ),
        )
        return {"translation": resp.text.strip()}
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {str(e)}")


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"deleted": True}


@app.post("/payment/create-checkout")
async def create_checkout(req: CheckoutRequest, request: Request):
    if not check_rate(request.client.host):
        raise HTTPException(429, "Rate limit.")
    price_id = STRIPE_YEARLY_PRICE if req.plan == "yearly" else STRIPE_MONTHLY_PRICE
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=req.email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/?payment=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/?payment=cancelled",
            metadata={"plan": req.plan},
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(500, f"Payment error: {str(e)}")


@app.post("/payment/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(400, "Invalid webhook.")
    if event["type"] == "checkout.session.completed":
        obj   = event["data"]["object"]
        email = obj.get("customer_email", "")
        plan  = obj.get("metadata", {}).get("plan", "monthly")
        token = issue_jwt(email, plan)
        print(f"PREMIUM TOKEN for {email}: {token}")  # wire to SendGrid for production
    return {"received": True}


@app.post("/payment/verify-token")
async def verify_token(req: VerifyTokenRequest):
    payload = verify_jwt(req.token)
    if not payload or not payload.get("premium"):
        raise HTTPException(401, "Invalid or expired token.")
    return {"premium": True, "plan": payload.get("plan"), "expires": payload.get("exp")}


@app.get("/privacy-policy")
def privacy():
    return {
        "controller": "LOVEUAD LTD (Co. 16838046)",
        "contact": "kanchan.g12@gmail.com",
        "ai_system": f"Google Gemini ({MODEL}) — EU AI Act Article 50 compliant",
        "data_stored": "None. Sessions ephemeral, RAM only, deleted after 30 min.",
        "legal_basis": "GDPR Article 6(1)(a) explicit consent",
        "age_requirement": "13+. COPPA compliant.",
    }
