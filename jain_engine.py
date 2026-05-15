"""
Jain Contacts Campaign Engine
Flask-based long-running service for AI-powered email outreach.
"""

import os
import time
import json
import logging
import smtplib
import imaplib
import email
import threading
import re
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# AI providers
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# SMTP / IMAP
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")

# Campaign defaults
DAILY_EMAIL_LIMIT = int(os.getenv("DAILY_EMAIL_LIMIT", "50"))
FOLLOW_UP_INTERVAL_HOURS = int(os.getenv("FOLLOW_UP_INTERVAL_HOURS", "48"))
MAX_FOLLOW_UPS = int(os.getenv("MAX_FOLLOW_UPS", "3"))
INBOX_POLL_INTERVAL_SECONDS = int(os.getenv("INBOX_POLL_INTERVAL_SECONDS", "300"))

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_state = {
    "campaigns": {},          # campaign_id -> campaign dict
    "daily_sent": {},         # date_str -> count
    "inbox_thread_started": False,
    "followup_thread_started": False,
}

# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _supabase_request(method: str, path: str, payload: dict | None = None) -> dict:
    """Thin wrapper around the Supabase REST API."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    resp = requests.request(method, url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json() if resp.text else {}


def db_get_contacts(campaign_id: str) -> list:
    try:
        return _supabase_request("GET", f"contacts?campaign_id=eq.{campaign_id}&select=*")
    except Exception as exc:
        logger.error("db_get_contacts error: %s", exc)
        return []


def db_update_contact(contact_id: str, updates: dict) -> None:
    try:
        _supabase_request("PATCH", f"contacts?id=eq.{contact_id}", updates)
    except Exception as exc:
        logger.error("db_update_contact error: %s", exc)


def db_log_email(record: dict) -> None:
    try:
        _supabase_request("POST", "email_logs", record)
    except Exception as exc:
        logger.error("db_log_email error: %s", exc)


def db_get_campaign(campaign_id: str) -> dict | None:
    try:
        rows = _supabase_request("GET", f"campaigns?id=eq.{campaign_id}&select=*")
        return rows[0] if rows else None
    except Exception as exc:
        logger.error("db_get_campaign error: %s", exc)
        return None


def db_update_campaign(campaign_id: str, updates: dict) -> None:
    try:
        _supabase_request("PATCH", f"campaigns?id=eq.{campaign_id}", updates)
    except Exception as exc:
        logger.error("db_update_campaign error: %s", exc)

# ---------------------------------------------------------------------------
# AI helpers
# ---------------------------------------------------------------------------

def _call_openrouter(prompt: str, model: str = "meta-llama/llama-3.1-8b-instruct:free") -> str:
    """Call OpenRouter chat completions API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_anthropic(prompt: str, model: str = "claude-3-haiku-20240307") -> str:
    """Call Anthropic Messages API."""
    import anthropic as anthropic_sdk
    client = anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def generate_personalized_email(contact: dict, campaign: dict, sequence_step: int = 1) -> tuple[str, str]:
    """
    Generate a personalized subject + body for a contact.
    Returns (subject, body).
    """
    name = contact.get("first_name") or contact.get("name", "there")
    company = contact.get("company", "your company")
    role = contact.get("role", "")
    campaign_context = campaign.get("context", "")
    sender_name = campaign.get("sender_name", "The Team")

    step_label = {1: "initial", 2: "first follow-up", 3: "second follow-up"}.get(sequence_step, "follow-up")

    prompt = (
        f"Write a concise, professional {step_label} cold outreach email.\n"
        f"Recipient: {name}, {role} at {company}.\n"
        f"Campaign context: {campaign_context}\n"
        f"Sender name: {sender_name}\n"
        f"Requirements:\n"
        f"- Personalise the opening line to the recipient's role/company\n"
        f"- Keep the body under 120 words\n"
        f"- End with a soft call-to-action (e.g. 'Would you be open to a quick call?')\n"
        f"- Do NOT include a subject line in the body\n"
        f"Return JSON with keys 'subject' and 'body' only."
    )

    raw = ""
    try:
        if OPENROUTER_API_KEY:
            raw = _call_openrouter(prompt, model="meta-llama/llama-3.1-8b-instruct:free")
        elif ANTHROPIC_API_KEY:
            raw = _call_anthropic(prompt)
        else:
            raise RuntimeError("No AI provider configured")

        # Extract JSON from the response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("subject", "Quick question"), data.get("body", raw)
    except Exception as exc:
        logger.warning("AI generation failed (%s), using fallback template", exc)

    # Fallback template
    subject = f"Quick question for {name}"
    body = (
        f"Hi {name},\n\n"
        f"I came across {company} and thought there might be a great fit with what we're working on.\n\n"
        f"{campaign_context}\n\n"
        f"Would you be open to a quick 15-minute call to explore this?\n\n"
        f"Best,\n{sender_name}"
    )
    return subject, body


def classify_reply(email_body: str) -> str:
    """
    Classify an inbound reply as one of:
    'interested', 'not_interested', 'objection', 'auto_reply', 'other'
    """
    prompt = (
        f"Classify the following email reply into exactly one category:\n"
        f"- interested (wants to meet / learn more)\n"
        f"- not_interested (explicit opt-out or rejection)\n"
        f"- objection (has a concern but not a hard no)\n"
        f"- auto_reply (out-of-office or automated message)\n"
        f"- other\n\n"
        f"Reply text:\n{email_body[:1000]}\n\n"
        f"Return only the category word, nothing else."
    )
    try:
        if OPENROUTER_API_KEY:
            result = _call_openrouter(prompt, model="meta-llama/llama-3.1-8b-instruct:free")
        elif ANTHROPIC_API_KEY:
            result = _call_anthropic(prompt)
        else:
            return "other"
        category = result.strip().lower().split()[0]
        valid = {"interested", "not_interested", "objection", "auto_reply", "other"}
        return category if category in valid else "other"
    except Exception as exc:
        logger.warning("classify_reply failed: %s", exc)
        return "other"


def generate_objection_response(contact: dict, campaign: dict, objection_text: str) -> tuple[str, str]:
    """Generate a reply to handle an objection."""
    name = contact.get("first_name") or contact.get("name", "there")
    sender_name = campaign.get("sender_name", "The Team")
    prompt = (
        f"Write a short, empathetic email reply addressing this objection from {name}:\n"
        f"\"{objection_text[:500]}\"\n\n"
        f"Keep it under 80 words. Acknowledge their concern, briefly reframe the value, "
        f"and suggest a low-commitment next step.\n"
        f"Sender name: {sender_name}\n"
        f"Return JSON with keys 'subject' and 'body' only."
    )
    raw = ""
    try:
        if OPENROUTER_API_KEY:
            raw = _call_openrouter(prompt, model="meta-llama/llama-3.1-8b-instruct:free")
        elif ANTHROPIC_API_KEY:
            raw = _call_anthropic(prompt)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("subject", f"Re: your note"), data.get("body", raw)
    except Exception as exc:
        logger.warning("generate_objection_response failed: %s", exc)
    subject = f"Re: your note, {name}"
    body = (
        f"Hi {name},\n\nThanks for the honest reply — I completely understand.\n\n"
        f"Happy to keep things brief. Would a 10-minute call work to see if there's any fit?\n\n"
        f"Best,\n{sender_name}"
    )
    return subject, body

# ---------------------------------------------------------------------------
# Email sending / receiving
# ---------------------------------------------------------------------------

def _daily_sent_count() -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _state_lock:
        return _state["daily_sent"].get(today, 0)


def _increment_daily_sent() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _state_lock:
        _state["daily_sent"][today] = _state["daily_sent"].get(today, 0) + 1


def send_email(to_address: str, subject: str, body: str, reply_to: str | None = None) -> bool:
    """Send a plain-text email via SMTP. Returns True on success."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        logger.error("SMTP credentials not configured")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_address
    msg["Subject"] = subject
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_address, msg.as_string())
        logger.info("Email sent to %s | subject: %s", to_address, subject)
        _increment_daily_sent()
        return True
    except Exception as exc:
        logger.error("send_email failed to %s: %s", to_address, exc)
        return False


def _decode_header_value(value: str) -> str:
    parts = decode_header(value)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def fetch_inbox_replies(since_hours: int = 24) -> list[dict]:
    """
    Fetch recent emails from the inbox via IMAP.
    Returns a list of dicts with keys: from, subject, body, date.
    """
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        logger.warning("IMAP credentials not configured — skipping inbox fetch")
        return []

    results = []
    try:
        since_date = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).strftime("%d-%b-%Y")
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as mail:
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            mail.select("INBOX")
            _, data = mail.search(None, f'(SINCE "{since_date}")')
            ids = data[0].split()
            for uid in ids[-100:]:  # cap at 100 most recent
                _, msg_data = mail.fetch(uid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)
                from_addr = _decode_header_value(msg.get("From", ""))
                subject = _decode_header_value(msg.get("Subject", ""))
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                            break
                else:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
                results.append({"from": from_addr, "subject": subject, "body": body, "date": msg.get("Date", "")})
    except Exception as exc:
        logger.error("fetch_inbox_replies error: %s", exc)
    return results

# ---------------------------------------------------------------------------
# Campaign execution
# ---------------------------------------------------------------------------

def _extract_email_address(raw: str) -> str:
    """Extract bare email address from a 'Name <addr>' string."""
    match = re.search(r"<([^>]+)>", raw)
    return match.group(1).strip() if match else raw.strip()


def process_campaign_step(campaign_id: str) -> dict:
    """
    Send the next batch of emails for a campaign, respecting the daily limit.
    Returns a summary dict.
    """
    campaign = db_get_campaign(campaign_id)
    if not campaign:
        return {"error": f"Campaign {campaign_id} not found"}

    if campaign.get("status") not in ("active", "running"):
        return {"skipped": True, "reason": "campaign not active"}

    contacts = db_get_contacts(campaign_id)
    sent_count = 0
    skipped_count = 0
    errors = []

    for contact in contacts:
        if _daily_sent_count() >= DAILY_EMAIL_LIMIT:
            logger.info("Daily email limit (%d) reached", DAILY_EMAIL_LIMIT)
            break

        status = contact.get("status", "pending")
        if status in ("replied", "unsubscribed", "bounced", "not_interested"):
            skipped_count += 1
            continue

        sequence_step = contact.get("sequence_step", 0)
        last_emailed_at = contact.get("last_emailed_at")

        # Determine if it's time to send
        if sequence_step > 0 and last_emailed_at:
            try:
                last_dt = datetime.fromisoformat(last_emailed_at.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) - last_dt < timedelta(hours=FOLLOW_UP_INTERVAL_HOURS):
                    skipped_count += 1
                    continue
            except ValueError:
                pass

        if sequence_step >= MAX_FOLLOW_UPS + 1:
            db_update_contact(contact["id"], {"status": "sequence_complete"})
            skipped_count += 1
            continue

        to_email = contact.get("email", "")
        if not to_email:
            skipped_count += 1
            continue

        subject, body = generate_personalized_email(contact, campaign, sequence_step + 1)
        success = send_email(to_email, subject, body)

        if success:
            sent_count += 1
            db_update_contact(contact["id"], {
                "status": "emailed",
                "sequence_step": sequence_step + 1,
                "last_emailed_at": datetime.now(timezone.utc).isoformat(),
            })
            db_log_email({
                "campaign_id": campaign_id,
                "contact_id": contact["id"],
                "to_email": to_email,
                "subject": subject,
                "sequence_step": sequence_step + 1,
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "status": "sent",
            })
        else:
            errors.append(to_email)

        # Small delay between sends to avoid rate limits
        time.sleep(2)

    return {"sent": sent_count, "skipped": skipped_count, "errors": errors}


def process_inbox_replies(campaign_id: str | None = None) -> dict:
    """
    Fetch inbox, classify replies, update contacts, and handle objections.
    """
    replies = fetch_inbox_replies(since_hours=INBOX_POLL_INTERVAL_SECONDS // 3600 + 1)
    processed = 0
    for reply in replies:
        from_addr = _extract_email_address(reply["from"])
        classification = classify_reply(reply["body"])
        logger.info("Reply from %s classified as: %s", from_addr, classification)

        # Find matching contact across all active campaigns
        try:
            rows = _supabase_request("GET", f"contacts?email=eq.{from_addr}&select=*")
        except Exception:
            rows = []

        for contact in rows:
            cid = contact.get("campaign_id")
            if campaign_id and cid != campaign_id:
                continue

            campaign = db_get_campaign(cid) if cid else None

            if classification == "interested":
                db_update_contact(contact["id"], {"status": "interested", "replied_at": datetime.now(timezone.utc).isoformat()})
                logger.info("🎯 Meeting interest detected from %s", from_addr)

            elif classification == "not_interested":
                db_update_contact(contact["id"], {"status": "not_interested"})

            elif classification == "objection" and campaign:
                db_update_contact(contact["id"], {"status": "objection_received"})
                subj, body = generate_objection_response(contact, campaign, reply["body"])
                send_email(from_addr, subj, body)
                db_update_contact(contact["id"], {"status": "objection_handled"})

            elif classification == "auto_reply":
                pass  # ignore auto-replies

            else:
                db_update_contact(contact["id"], {"status": "replied"})

            processed += 1

    return {"replies_fetched": len(replies), "processed": processed}

# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

def _inbox_monitor_loop() -> None:
    """Continuously poll the inbox for replies."""
    logger.info("Inbox monitor thread started (interval=%ds)", INBOX_POLL_INTERVAL_SECONDS)
    while True:
        try:
            result = process_inbox_replies()
            logger.info("Inbox poll: %s", result)
        except Exception as exc:
            logger.error("Inbox monitor error: %s", exc)
        time.sleep(INBOX_POLL_INTERVAL_SECONDS)


def _followup_loop() -> None:
    """Periodically trigger follow-up sends for all active campaigns."""
    logger.info("Follow-up scheduler thread started")
    while True:
        try:
            with _state_lock:
                active_campaigns = [
                    cid for cid, c in _state["campaigns"].items()
                    if c.get("status") == "active"
                ]
            for cid in active_campaigns:
                logger.info("Running follow-up step for campaign %s", cid)
                result = process_campaign_step(cid)
                logger.info("Follow-up result for %s: %s", cid, result)
        except Exception as exc:
            logger.error("Follow-up loop error: %s", exc)
        # Check every hour
        time.sleep(3600)


def _ensure_background_threads() -> None:
    """Start background threads exactly once."""
    with _state_lock:
        if not _state["inbox_thread_started"]:
            t = threading.Thread(target=_inbox_monitor_loop, daemon=True, name="inbox-monitor")
            t.start()
            _state["inbox_thread_started"] = True

        if not _state["followup_thread_started"]:
            t = threading.Thread(target=_followup_loop, daemon=True, name="followup-scheduler")
            t.start()
            _state["followup_thread_started"] = True

# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "jain-campaign-engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "daily_sent": _daily_sent_count(),
        "daily_limit": DAILY_EMAIL_LIMIT,
    })


@app.route("/campaigns", methods=["GET"])
def list_campaigns():
    """List all campaigns tracked in memory."""
    with _state_lock:
        return jsonify({"campaigns": list(_state["campaigns"].values())})


@app.route("/campaigns/<campaign_id>/launch", methods=["POST"])
def launch_campaign(campaign_id: str):
    """
    Launch or resume a campaign.
    Body (JSON, all optional):
      - context: str — campaign pitch / context for AI
      - sender_name: str — name to sign emails with
    """
    body = request.get_json(silent=True) or {}
    campaign = db_get_campaign(campaign_id)
    if not campaign:
        # Create a minimal in-memory campaign record
        campaign = {
            "id": campaign_id,
            "context": body.get("context", ""),
            "sender_name": body.get("sender_name", "The Team"),
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        campaign.update({
            "context": body.get("context", campaign.get("context", "")),
            "sender_name": body.get("sender_name", campaign.get("sender_name", "The Team")),
            "status": "active",
        })
        db_update_campaign(campaign_id, {"status": "active"})

    with _state_lock:
        _state["campaigns"][campaign_id] = campaign

    _ensure_background_threads()

    # Kick off the first send immediately in a background thread
    def _run():
        result = process_campaign_step(campaign_id)
        logger.info("Initial campaign send for %s: %s", campaign_id, result)

    threading.Thread(target=_run, daemon=True, name=f"campaign-{campaign_id}").start()

    return jsonify({"status": "launched", "campaign_id": campaign_id}), 202


@app.route("/campaigns/<campaign_id>/pause", methods=["POST"])
def pause_campaign(campaign_id: str):
    """Pause an active campaign."""
    with _state_lock:
        if campaign_id in _state["campaigns"]:
            _state["campaigns"][campaign_id]["status"] = "paused"
    db_update_campaign(campaign_id, {"status": "paused"})
    return jsonify({"status": "paused", "campaign_id": campaign_id})


@app.route("/campaigns/<campaign_id>/resume", methods=["POST"])
def resume_campaign(campaign_id: str):
    """Resume a paused campaign."""
    with _state_lock:
        if campaign_id in _state["campaigns"]:
            _state["campaigns"][campaign_id]["status"] = "active"
    db_update_campaign(campaign_id, {"status": "active"})
    _ensure_background_threads()
    return jsonify({"status": "resumed", "campaign_id": campaign_id})


@app.route("/campaigns/<campaign_id>/status", methods=["GET"])
def campaign_status(campaign_id: str):
    """Get the current status of a campaign."""
    with _state_lock:
        campaign = _state["campaigns"].get(campaign_id)
    if not campaign:
        campaign = db_get_campaign(campaign_id)
    if not campaign:
        return jsonify({"error": "Campaign not found"}), 404
    contacts = db_get_contacts(campaign_id)
    status_counts: dict[str, int] = {}
    for c in contacts:
        s = c.get("status", "pending")
        status_counts[s] = status_counts.get(s, 0) + 1
    return jsonify({
        "campaign_id": campaign_id,
        "campaign_status": campaign.get("status"),
        "contact_counts": status_counts,
        "total_contacts": len(contacts),
        "daily_sent_today": _daily_sent_count(),
    })


@app.route("/campaigns/<campaign_id>/send-step", methods=["POST"])
def manual_send_step(campaign_id: str):
    """Manually trigger one send step for a campaign (useful for testing)."""
    with _state_lock:
        campaign = _state["campaigns"].get(campaign_id)
    if not campaign:
        campaign = db_get_campaign(campaign_id)
        if campaign:
            with _state_lock:
                _state["campaigns"][campaign_id] = campaign
    if not campaign:
        return jsonify({"error": "Campaign not found"}), 404
    result = process_campaign_step(campaign_id)
    return jsonify(result)


@app.route("/inbox/poll", methods=["POST"])
def manual_inbox_poll():
    """Manually trigger an inbox poll (useful for testing)."""
    campaign_id = (request.get_json(silent=True) or {}).get("campaign_id")
    result = process_inbox_replies(campaign_id)
    return jsonify(result)


@app.route("/campaigns/<campaign_id>/contacts", methods=["GET"])
def get_contacts(campaign_id: str):
    """List all contacts for a campaign."""
    contacts = db_get_contacts(campaign_id)
    return jsonify({"campaign_id": campaign_id, "contacts": contacts, "total": len(contacts)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Jain Campaign Engine starting on port 5000...")
    _ensure_background_threads()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

