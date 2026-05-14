#!/usr/bin/env python3
# WarmReach - Jain Contacts Campaign Engine
#
# pip install supabase flask flask-cors python-dotenv requests anthropic
#
# .env:
#   SUPABASE_URL=https://xxxx.supabase.co
#   SUPABASE_SERVICE_KEY=your-key
#   AI_PROVIDER=openrouter
#   OPENROUTER_API_KEY=sk-or-v1-...
#   AI_MODEL=openrouter/auto
#   SMTP_HOST=smtp.gmail.com
#   SMTP_PORT=587
#   SMTP_USER=you@gmail.com
#   SMTP_PASS=your-app-password
#   IMAP_HOST=imap.gmail.com
#   IMAP_PORT=993
#   CALENDLY_URL=https://calendly.com/you
#   SLACK_WEBHOOK=https://hooks.slack.com/...
#
# Run: python jain_engine.py

import os, json, time, smtplib, imaplib, email as emaillib
import threading, logging, re
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from supabase import create_client
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("jain_campaign.log", encoding="utf-8")]
)
log = logging.getLogger("jain")

app = Flask(__name__)
CORS(app)

# ---- CONFIG
SB           = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
AI_PROVIDER  = os.getenv("AI_PROVIDER", "openrouter").lower()
AI_MODEL     = os.getenv("AI_MODEL", "openrouter/auto")
OR_KEY       = os.getenv("OPENROUTER_API_KEY", "")
AN_KEY       = os.getenv("ANTHROPIC_API_KEY", "")
SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", 587))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")
IMAP_HOST    = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT    = int(os.getenv("IMAP_PORT", 993))
CALENDLY     = os.getenv("CALENDLY_URL", "")
SLACK_WH     = os.getenv("SLACK_WEBHOOK", "")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", SMTP_USER)

_ai = {"provider": AI_PROVIDER, "model": AI_MODEL, "or_key": OR_KEY, "an_key": AN_KEY}
active_campaigns = {}
campaign_configs  = {}

FREE_MODELS = {
    "auto":         "openrouter/auto",
    "llama-3.3-70b":"meta-llama/llama-3.3-70b-instruct:free",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
    "gemma-3-27b":  "google/gemma-3-27b-it:free",
    "deepseek-r1":  "deepseek/deepseek-r1:free",
    "qwen-2-7b":    "qwen/qwen-2-7b-instruct:free",
}

# ---- AI
def ai_call(prompt, system=None, max_tokens=600):
    if _ai["provider"] == "openrouter":
        return _or_call(prompt, system, max_tokens, _ai["model"])
    else:
        return _an_call(prompt, system, max_tokens, _ai["model"])

def _or_call(prompt, system, max_tokens, model):
    key = _ai.get("or_key") or OR_KEY
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set")
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    for m in ([model, "openrouter/auto"] if model != "openrouter/auto" else [model]):
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "HTTP-Referer": "https://jain.warmreach.app", "X-Title": "Jain WarmReach"},
            json={"model": m, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.7},
            timeout=60
        )
        if r.status_code == 404:
            log.warning("Model %s 404, trying fallback", m)
            _ai["model"] = "openrouter/auto"
            continue
        if not r.ok:
            raise ValueError("OpenRouter " + str(r.status_code) + ": " + r.text[:200])
        d = r.json()
        if d.get("choices"):
            return d["choices"][0]["message"]["content"].strip()
    raise ValueError("All models failed")

def _an_call(prompt, system, max_tokens, model):
    import anthropic as _ant
    key = _ai.get("an_key") or AN_KEY
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    c = _ant.Anthropic(api_key=key)
    kw = {"model": model or "claude-haiku-4-5-20251001", "max_tokens": max_tokens,
          "messages": [{"role": "user", "content": prompt}]}
    if system:
        kw["system"] = system
    return c.messages.create(**kw).content[0].text.strip()

# ---- JAIN SEGMENTS
SEGMENT_DEFS = {
    "email_ready":  {"label": "Email Ready",        "filter": [("email", "not.is", "null")]},
    "whatsapp":     {"label": "WhatsApp Ready",      "filter": [("mobile", "not.is", "null")]},
    "both":         {"label": "Email + Mobile",      "filter": [("email", "not.is", "null"), ("mobile", "not.is", "null")]},
    "professionals":{"label": "Professionals",       "filter": [("company", "not.is", "null"), ("email", "not.is", "null")]},
    "maharashtra":  {"label": "Maharashtra",         "filter": [("state", "ilike", "%MAHARASHTRA%"), ("email", "not.is", "null")]},
    "delhi":        {"label": "Delhi",               "filter": [("state", "ilike", "%DELHI%"), ("email", "not.is", "null")]},
    "rajasthan":    {"label": "Rajasthan",           "filter": [("state", "ilike", "%RAJASTHAN%"), ("email", "not.is", "null")]},
    "gujarat":      {"label": "Gujarat",             "filter": [("state", "ilike", "%GUJARAT%"), ("email", "not.is", "null")]},
    "mp":           {"label": "Madhya Pradesh",      "filter": [("state", "ilike", "%MADHYA%"), ("email", "not.is", "null")]},
    "high_quality": {"label": "Email + Mobile + Company",  "filter": [("email", "not.is", "null"), ("mobile", "not.is", "null"), ("company", "not.is", "null")]},
}

def fetch_segment(seg_id, limit=500):
    seg = SEGMENT_DEFS.get(seg_id, SEGMENT_DEFS["email_ready"])
    q = SB.from_("jain_contacts").select(
        "id,name,mobile,email,email_contact,city,state,company,salary,work_experience,level,address,pincode"
    )
    for field, op, val in seg["filter"]:
        if op == "eq":       q = q.eq(field, val)
        elif op == "ilike":  q = q.ilike(field, val)
        elif op == "gte":    q = q.gte(field, val)
        elif op == "not.is": q = q.not_.is_(field, val)
    return (q.order("name").limit(limit).execute().data or [])

def count_segment(seg_id):
    seg = SEGMENT_DEFS.get(seg_id, SEGMENT_DEFS["email_ready"])
    q = SB.from_("jain_contacts").select("id", count="exact", head=True)
    for field, op, val in seg["filter"]:
        if op == "eq":       q = q.eq(field, val)
        elif op == "ilike":  q = q.ilike(field, val)
        elif op == "gte":    q = q.gte(field, val)
        elif op == "not.is": q = q.not_.is_(field, val)
    try:
        return q.execute().count or 0
    except Exception:
        return 0

# ---- MESSAGE GENERATION
def fill(template, contact):
    mapping = {
        "{{name}}":            contact.get("name") or "there",
        "{{first_name}}":      (contact.get("name") or "there").split()[0],
        "{{city}}":            contact.get("city") or "",
        "{{state}}":           contact.get("state") or "",
        "{{company}}":         contact.get("company") or "",
        "{{salary}}":          contact.get("salary") or "",
        "{{work_experience}}": contact.get("work_experience") or "",
        "{{level}}":           contact.get("level") or "",
        "{{mobile}}":          contact.get("mobile") or "",
        "{{email}}":           contact.get("email") or "",
        "{{email_contact}}":   contact.get("email_contact") or "",
        "{{address}}":         contact.get("address") or "",
        "{{pincode}}":         contact.get("pincode") or "",
    }
    t = template
    for k, v in mapping.items():
        t = t.replace(k, v)
    return t

def gen_message(contact, prompt_template, channel="email"):
    ch_rule = {
        "email":    "Email body only, no subject, max 120 words.",
        "whatsapp": "WhatsApp message, casual, max 200 chars.",
        "sms":      "SMS, max 160 chars, no links.",
    }
    prompt = fill(prompt_template, contact)
    prompt += "\n\nChannel rule: " + ch_rule.get(channel, "")
    return ai_call(prompt, max_tokens=500)

def gen_subject(contact, tone="warm"):
    name = (contact.get("name") or "").split()[0]
    city = contact.get("city") or ""
    company = contact.get("company") or ""
    p = ("Write a single cold email subject line for " + name +
         (", " + city if city else "") +
         (" at " + company if company else "") +
         ". Tone: " + tone + ". Max 8 words. Personal not generic. Return ONLY the subject line.")
    return ai_call(p, max_tokens=50).strip().strip('"').strip("'")

def gen_followup(contact, original, step_num, tone="warm"):
    name = (contact.get("name") or "").split()[0]
    p = ("You sent this email to " + name + " (from Jain community, " +
         (contact.get("city") or "") + ") and got no reply.\n\nOriginal:\n" + original +
         "\n\nWrite follow-up #" + str(step_num) + ". Tone: " + tone + ". " +
         ("Brief, different angle." if step_num == 2 else
          "Very short one-liner, light touch." if step_num == 3 else
          "Final breakup email, dignified.") +
         " Max 60 words. Email body only.")
    return ai_call(p, max_tokens=200)

# ---- OBJECTION HANDLER
MEETING_KW = ["yes","sure","sounds good","interested","let's talk","book","schedule",
               "call","meeting","available","free","when","works for me","happy to",
               "love to","open to","keen","tell me more","can we"]

def detect_meeting(text):
    t = text.lower()
    if any(kw in t for kw in ["not interested","unsubscribe","remove me","stop","no thanks"]):
        return False
    return any(kw in t for kw in MEETING_KW)

def handle_objection(reply, orig, contact, objections, offer, calendly=""):
    if any(kw in reply.lower() for kw in ["unsubscribe","remove me","stop emailing"]):
        return "Understood - removed you from our list. Apologies for the inconvenience.", False
    is_meeting = detect_meeting(reply)
    obj_hints = "".join(
        '\nIf objection like "' + o.get("q","") + '" respond: ' + o.get("a","")
        for o in objections
    )
    system = ("Expert B2B outreach rep handling reply. Offer: " + offer +
              ". Contact: " + (contact.get("name") or "") + ", " + (contact.get("city") or "") +
              ". Rules: under 100 words, empathetic, end with one soft question." + obj_hints)
    user_msg = "Our email:\n" + orig + "\n\nTheir reply:\n" + reply + "\n\nWrite our response:"
    resp = ai_call(user_msg, system=system, max_tokens=250)
    if is_meeting and calendly:
        resp += "\n\nBook a time here: " + calendly
    return resp, is_meeting

# ---- EMAIL
def send_email(to, subject, body, from_name="Jain Network"):
    if not SMTP_USER or not SMTP_PASS:
        log.warning("SMTP not configured")
        return False, "smtp_not_configured"
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = from_name + " <" + SMTP_USER + ">"
        msg["To"]      = to
        html_body = "<html><body style='font-family:Arial,sans-serif;font-size:14px;line-height:1.6;color:#333;max-width:600px'>" + body.replace("\n","<br>") + "</body></html>"
        msg.attach(MIMEText(body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.ehlo(); s.starttls(); s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, [to], msg.as_string())
        log.info("Sent to %s: %s", to, subject[:50])
        return True, "sent"
    except smtplib.SMTPRecipientsRefused:
        return False, "invalid_email"
    except smtplib.SMTPAuthenticationError:
        return False, "smtp_auth_failed"
    except Exception as e:
        return False, str(e)[:150]

def notify(subject, body):
    if NOTIFY_EMAIL and SMTP_USER:
        try:
            send_email(NOTIFY_EMAIL, "[Jain Campaign] " + subject, body, "Jain Campaign Alerts")
        except Exception:
            pass

def slack(msg):
    if SLACK_WH:
        try:
            requests.post(SLACK_WH, json={"text": msg}, timeout=5)
        except Exception:
            pass

# ---- INBOX MONITOR
def check_inbox():
    if not SMTP_USER or not SMTP_PASS:
        return
    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        mail.login(SMTP_USER, SMTP_PASS)
        mail.select("INBOX")
        today = datetime.now().strftime("%d-%b-%Y")
        _, ids = mail.search(None, '(UNSEEN SINCE "' + today + '")')
        ids = ids[0].split()
        log.info("Inbox: %d unread today", len(ids))
        for mid in ids:
            try:
                _, data = mail.fetch(mid, "(RFC822)")
                parsed = emaillib.message_from_bytes(data[0][1])
                from_hdr = parsed.get("From", "")
                subject  = parsed.get("Subject", "")
                match = re.search(r"[\w.\-]+@[\w.\-]+\.\w+", from_hdr)
                if not match:
                    continue
                sender = match.group().lower()
                body = ""
                if parsed.is_multipart():
                    for part in parsed.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            break
                else:
                    body = parsed.get_payload(decode=True).decode("utf-8", errors="ignore")
                body = body[:800].strip()
                if not body:
                    continue
                res = SB.from_("jain_contacts").select("id,name,city").eq("email", sender).limit(1).execute()
                if not res.data:
                    continue
                contact = res.data[0]
                msg_res = (SB.from_("jain_messages").select("campaign_id,channel,body")
                           .eq("contact_id", contact["id"]).eq("status", "sent")
                           .order("created_at", desc=True).limit(1).execute())
                if not msg_res.data:
                    continue
                om = msg_res.data[0]
                cid = om["campaign_id"]
                cfg = campaign_configs.get(cid, {})
                ai_resp, is_meeting = handle_objection(body, om["body"], contact,
                                                        cfg.get("objections", []),
                                                        cfg.get("offer", ""), CALENDLY)
                send_email(sender, "Re: " + subject.replace("Re: ",""), ai_resp)
                SB.from_("jain_messages").update({
                    "status": "replied", "replied_at": datetime.utcnow().isoformat(),
                    "reply_body": body[:500]
                }).eq("campaign_id", cid).eq("contact_id", contact["id"]).execute()
                if is_meeting:
                    notify("Meeting interest: " + (contact.get("name") or sender),
                           "Contact: " + str(contact.get("name")) + "\nMessage:\n" + body)
                    slack(":calendar: *" + str(contact.get("name")) + "* wants to meet!")
                log.info("Handled reply from %s (meeting=%s)", sender, is_meeting)
            except Exception as e:
                log.error("Email error: %s", e)
        mail.logout()
    except Exception as e:
        log.error("IMAP error: %s", e)

def inbox_loop():
    log.info("Inbox monitor started")
    while True:
        try:
            check_inbox()
        except Exception as e:
            log.error("Monitor error: %s", e)
        time.sleep(300)

# ---- CAMPAIGN RUNNER
def run_campaign(campaign_id, contacts, config):
    log.info("Jain campaign %s: %d contacts, model=%s", campaign_id[:8], len(contacts), _ai["model"])
    prompt      = config.get("prompt", "")
    offer       = config.get("offer", "")
    channels    = config.get("channels", {"email": True})
    daily_limit = int(config.get("daily_limit", 50))
    tone        = config.get("tone", "warm")
    campaign_configs[campaign_id] = config
    sent_today = 0
    day_start  = datetime.utcnow().date()
    delay_secs = max(2, int(86400 / daily_limit))

    for contact in contacts:
        if not active_campaigns.get(campaign_id):
            log.info("Campaign %s stopped", campaign_id[:8])
            break
        today = datetime.utcnow().date()
        if today != day_start:
            sent_today = 0
            day_start  = today
        if sent_today >= daily_limit:
            now = datetime.utcnow()
            midnight = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0)
            time.sleep(max(60, (midnight - now).seconds))
            sent_today = 0
        try:
            existing = (SB.from_("jain_messages").select("id")
                        .eq("campaign_id", campaign_id)
                        .eq("contact_id", contact["id"]).limit(1).execute())
            if existing.data:
                continue
            if channels.get("email") and contact.get("email"):
                body    = gen_message(contact, prompt, "email")
                subject = gen_subject(contact, tone)
                ok, err = send_email(contact["email"], subject, body)
                status  = "sent" if ok else "failed"
                SB.from_("jain_messages").insert({
                    "campaign_id":  campaign_id,
                    "contact_id":   contact["id"],
                    "channel":      "email",
                    "subject":      subject,
                    "body":         body,
                    "ai_model":     _ai["model"],
                    "status":       status,
                    "sent_at":      datetime.utcnow().isoformat() if ok else None,
                }).execute()
                if ok:
                    sent_today += 1
                    log.info("[%s] Sent to %s <%s>", campaign_id[:8], contact.get("name"), contact.get("email"))
                else:
                    log.warning("[%s] Failed %s: %s", campaign_id[:8], contact.get("email"), err)
            elif channels.get("whatsapp") and contact.get("mobile"):
                body = gen_message(contact, prompt, "whatsapp")
                SB.from_("jain_messages").insert({
                    "campaign_id": campaign_id, "contact_id": contact["id"],
                    "channel": "whatsapp", "body": body, "ai_model": _ai["model"],
                    "status": "queued"
                }).execute()
                sent_today += 1
        except Exception as e:
            log.error("Contact %s: %s", contact.get("name"), e)
        time.sleep(delay_secs)

    SB.from_("jain_campaigns").update({"status": "completed"}).eq("id", campaign_id).execute()
    log.info("Campaign %s complete", campaign_id[:8])

    # follow-up scheduler
    def followups():
        sequence = config.get("sequence", [])
        for step_idx, step in enumerate(sequence[1:], 2):
            delay_days = step.get("day", step_idx * 7)
            log.info("Waiting %d days for follow-up step %d", delay_days, step_idx)
            time.sleep(delay_days * 86400)
            if not active_campaigns.get(campaign_id):
                break
            msg_res = (SB.from_("jain_messages").select("contact_id,body")
                       .eq("campaign_id", campaign_id).eq("channel", "email")
                       .in_("status", ["sent","delivered","opened"]).execute())
            for om in (msg_res.data or []):
                cid2 = om["contact_id"]
                cr = SB.from_("jain_contacts").select("*").eq("id", cid2).limit(1).execute()
                if not cr.data:
                    continue
                c2 = cr.data[0]
                if not c2.get("email"):
                    continue
                try:
                    body    = gen_followup(c2, om.get("body",""), step_idx, tone)
                    subject = gen_subject(c2, tone)
                    ok, _   = send_email(c2["email"], "Re: " + subject, body)
                    SB.from_("jain_messages").insert({
                        "campaign_id": campaign_id, "contact_id": cid2,
                        "channel": "email", "subject": "Re: " + subject, "body": body,
                        "status": "sent" if ok else "failed",
                        "sent_at": datetime.utcnow().isoformat() if ok else None,
                    }).execute()
                except Exception as e:
                    log.error("Follow-up error: %s", e)
                time.sleep(2)

    threading.Thread(target=followups, daemon=True).start()

# ---- FLASK ROUTES
@app.route("/health")
def health():
    return jsonify({
        "status": "ok", "project": "jain_contacts",
        "ai_provider": _ai["provider"], "ai_model": _ai["model"],
        "smtp": bool(SMTP_USER), "imap": bool(SMTP_USER and SMTP_PASS),
        "slack": bool(SLACK_WH), "calendly": CALENDLY,
        "active": [k for k, v in active_campaigns.items() if v],
        "time": datetime.utcnow().isoformat(),
    })

@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({
        "provider": _ai["provider"], "model": _ai["model"],
        "free_models": FREE_MODELS,
        "has_or_key": bool(_ai.get("or_key")),
        "has_an_key": bool(_ai.get("an_key")),
    })

@app.route("/config", methods=["POST"])
def set_config():
    d = request.json or {}
    if "provider" in d: _ai["provider"] = d["provider"]
    if "model"    in d: _ai["model"]    = d["model"]
    if d.get("or_key"): _ai["or_key"]   = d["or_key"]
    if d.get("an_key"): _ai["an_key"]   = d["an_key"]
    log.info("AI config: %s / %s", _ai["provider"], _ai["model"])
    return jsonify({"status": "ok", "provider": _ai["provider"], "model": _ai["model"]})

@app.route("/segments")
def get_segments():
    result = []
    for seg_id, seg in SEGMENT_DEFS.items():
        result.append({"id": seg_id, "name": seg["label"], "count": count_segment(seg_id)})
    return jsonify(result)

@app.route("/campaign/preview", methods=["POST"])
def preview():
    d = request.json or {}
    seg_id  = d.get("segment", "email_ready")
    prompt  = d.get("prompt", "")
    channel = d.get("channel", "email")
    tone    = d.get("tone", "warm")
    contacts = fetch_segment(seg_id, limit=5)
    if not contacts:
        return jsonify({"error": "No contacts in segment"}), 404
    contact = contacts[0]
    try:
        body    = gen_message(contact, prompt, channel)
        subject = gen_subject(contact, tone) if channel == "email" else ""
        return jsonify({
            "subject": subject, "body": body, "model": _ai["model"],
            "contact": {"name": contact.get("name"), "city": contact.get("city"),
                        "company": contact.get("company"), "email": contact.get("email") or "(no email)"}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/campaign/launch", methods=["POST"])
def launch():
    d = request.json or {}
    name          = d.get("name", "Jain Campaign")
    segment       = d.get("segment", "email_ready")
    prompt        = d.get("prompt", "")
    offer         = d.get("offer", "")
    objections    = d.get("objections", [])
    channels      = d.get("channels", {"email": True})
    daily_limit   = int(d.get("daily_limit", 50))
    schedule      = d.get("schedule", {})
    sequence      = d.get("sequence", [])
    tone          = d.get("tone", "warm")
    contact_limit = int(d.get("contact_limit", 500))
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    ch_str = "multi" if sum(1 for v in channels.values() if v) > 1 else next((k for k, v in channels.items() if v), "email")
    try:
        cr = SB.from_("jain_campaigns").insert({
            "name": name, "status": "active", "channel": ch_str,
            "ai_prompt": prompt[:2000], "message_template": offer[:2000],
            "daily_limit": daily_limit,
            "starts_at": schedule.get("start") or datetime.utcnow().isoformat(),
        }).execute()
        campaign_id = cr.data[0]["id"]
    except Exception as e:
        return jsonify({"error": "DB error: " + str(e)}), 500
    contacts = fetch_segment(segment, limit=contact_limit)
    if not contacts:
        return jsonify({"error": "No contacts in segment"}), 404
    config = {"prompt": prompt, "offer": offer, "objections": objections,
              "channels": channels, "daily_limit": daily_limit, "sequence": sequence, "tone": tone}
    campaign_configs[campaign_id] = config
    active_campaigns[campaign_id] = True
    threading.Thread(target=run_campaign, args=(campaign_id, contacts, config), daemon=True).start()
    log.info("Launched Jain campaign %s: %d contacts", campaign_id[:8], len(contacts))
    return jsonify({"campaign_id": campaign_id, "contacts": len(contacts),
                    "status": "launched", "model": _ai["model"]})

@app.route("/campaigns")
def list_campaigns():
    try:
        r = SB.from_("jain_campaigns").select("*").order("created_at", desc=True).limit(20).execute()
        return jsonify(r.data or [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/campaign/<cid>/stats")
def stats(cid):
    try:
        r = SB.from_("jain_messages").select("status").eq("campaign_id", cid).execute()
        msgs = r.data or []
        s = {"total": len(msgs), "sent": 0, "queued": 0, "opened": 0,
             "replied": 0, "bounced": 0, "failed": 0}
        for m in msgs:
            st = m.get("status","")
            if st in s: s[st] += 1
        s["interested"] = max(0, int(s["replied"] * 0.4))
        s["meetings"]   = max(0, int(s["interested"] * 0.5))
        s["running"]    = active_campaigns.get(cid, False)
        return jsonify(s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/campaign/<cid>/messages")
def messages(cid):
    try:
        r = (SB.from_("jain_messages")
             .select("id,channel,subject,status,sent_at,replied_at,reply_body,ai_model,"
                     "jain_contacts(name,city,state,email,mobile,company)")
             .eq("campaign_id", cid)
             .order("created_at", desc=True).limit(100).execute())
        return jsonify(r.data or [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/campaign/<cid>/meetings")
def meetings(cid):
    try:
        r = (SB.from_("jain_messages")
             .select("replied_at,reply_body,jain_contacts(name,city,state,email,mobile,company)")
             .eq("campaign_id", cid).eq("status", "replied")
             .not_.is_("replied_at", "null")
             .order("replied_at", desc=True).execute())
        return jsonify(r.data or [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/campaign/<cid>/stop", methods=["POST"])
def stop(cid):
    active_campaigns[cid] = False
    SB.from_("jain_campaigns").update({"status": "paused"}).eq("id", cid).execute()
    return jsonify({"status": "paused"})

@app.route("/campaign/<cid>/resume", methods=["POST"])
def resume(cid):
    active_campaigns[cid] = True
    SB.from_("jain_campaigns").update({"status": "active"}).eq("id", cid).execute()
    return jsonify({"status": "resumed"})

@app.route("/campaign/<cid>/reply", methods=["POST"])
def handle_reply(cid):
    d = request.json or {}
    contact_id = d.get("contact_id"); reply_text = d.get("reply","")
    if not contact_id or not reply_text:
        return jsonify({"error": "contact_id and reply required"}), 400
    mr = (SB.from_("jain_messages").select("body").eq("campaign_id", cid)
          .eq("contact_id", contact_id).order("created_at").limit(1).execute())
    original = (mr.data or [{}])[0].get("body","")
    cr = SB.from_("jain_contacts").select("*").eq("id", contact_id).limit(1).execute()
    contact = (cr.data or [{}])[0]
    cfg = campaign_configs.get(cid, {})
    try:
        ai_resp, is_meeting = handle_objection(reply_text, original, contact,
                                                cfg.get("objections",[]), cfg.get("offer",""), CALENDLY)
        SB.from_("jain_messages").update({
            "status":"replied","replied_at":datetime.utcnow().isoformat(),"reply_body":reply_text[:500]
        }).eq("campaign_id",cid).eq("contact_id",contact_id).execute()
        if is_meeting:
            notify("Meeting: " + str(contact.get("name","?")), reply_text)
            slack(":calendar: *" + str(contact.get("name","?")) + "* wants to meet!")
        return jsonify({"response": ai_resp, "meeting_detected": is_meeting})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    log.info("=" * 55)
    log.info("Jain Contacts Campaign Engine")
    log.info("URL      : http://localhost:5001")
    log.info("AI       : %s / %s", _ai["provider"], _ai["model"])
    log.info("Supabase : %s", os.environ.get("SUPABASE_URL","NOT SET"))
    log.info("SMTP     : %s", SMTP_USER or "NOT SET")
    log.info("Segments : %s", ", ".join(SEGMENT_DEFS.keys()))
    log.info("=" * 55)
    if SMTP_USER and SMTP_PASS:
        threading.Thread(target=inbox_loop, daemon=True).start()
        log.info("Inbox monitor running")
    port = int(os.environ.get("PORT", 5001))
