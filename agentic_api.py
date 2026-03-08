import json
import os
import time
from typing import TypedDict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Campus Safety API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a strict JSON classifier.

Return exactly ONE valid JSON object.
Return no markdown.
Return no explanation.
Return no text before or after JSON.
Do not use code fences.
Do not add extra keys.
Do not omit required keys.

You must classify the user message into this exact JSON format:

{
  "incident_type": "SAFETY_THREAT | MEDICAL_EMERGENCY | PANIC_DISTRESS | LOST_DISORIENTED | SUSPICIOUS_ACTIVITY | GENERAL_HELP",
  "severity": "LOW | MEDIUM | HIGH",
  "confidence": 0.0,
  "reason": "string",
  "guidance_title": "string",
  "guidance_steps": ["string", "string"],
  "recording_triggered": true,
  "escalation_recommended": true,
  "escalation_target": "Campus Security | Medical Services | None"
}

Allowed values only:

incident_type:
- SAFETY_THREAT
- MEDICAL_EMERGENCY
- PANIC_DISTRESS
- LOST_DISORIENTED
- SUSPICIOUS_ACTIVITY
- GENERAL_HELP

severity:
- LOW
- MEDIUM
- HIGH

escalation_target:
- Campus Security
- Medical Services
- None

Rules:
- confidence must be a number between 0 and 1
- guidance_steps must be an array of short strings
- reason must be short and direct
- recording_triggered must be true or false
- escalation_recommended must be true or false
- output valid JSON only
- no trailing commas
- no comments
- no extra fields

Decision rules:
- direct violence, threat, attack, stalking, harassment -> SAFETY_THREAT
- injury, collapse, fainting, chest pain, bleeding, breathing issue -> MEDICAL_EMERGENCY
- panic, crying, fear, overwhelmed, unsafe feeling -> PANIC_DISTRESS
- cannot find location, confused where they are, lost on campus -> LOST_DISORIENTED
- suspicious person, suspicious behavior, someone watching/following, unknown risk -> SUSPICIOUS_ACTIVITY
- general support request without danger -> GENERAL_HELP

Severity rules:
- immediate danger or urgent health risk -> HIGH
- concerning but not immediate life-threatening -> MEDIUM
- mild concern or request for help -> LOW

Recording rules:
- set recording_triggered to true for HIGH severity
- set recording_triggered to true for direct threat or suspicious activity with risk
- otherwise false

Escalation rules:
- choose Campus Security for threats, stalking, suspicious activity, violence
- choose Medical Services for health emergencies
- choose None if escalation is not needed

If uncertain, still return valid JSON with lower confidence."""

# ── Groq call ──────────────────────────────────────────────────────────────────
def analyze_query(user_query: str) -> dict:
    print("DEBUG USER QUERY:", user_query)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Classify this message:\n{user_query}"}
        ],
        temperature=0,
        top_p=0.1,
        max_tokens=400,
        response_format={"type": "json_object"}
    )

    result = response.choices[0].message.content

    try:
        return json.loads(result)
    except Exception:
        print("RAW MODEL OUTPUT:\n", result)
        return {"error": "Model did not return valid JSON"}

# ── LangGraph state ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_query:     str
    classification: dict
    actions:        List[str]
    display_text:   str          # ← what gets returned to Flutter

# ── Nodes ──────────────────────────────────────────────────────────────────────
def classify_node(state: AgentState) -> AgentState:
    result = analyze_query(state["user_query"])
    print("\nParsed JSON Output:")
    print(json.dumps(result, indent=2))
    state["classification"] = result
    return state

def decision_node(state: AgentState) -> AgentState:
    data       = state["classification"]
    severity   = data.get("severity",   "LOW")
    confidence = data.get("confidence", 0)
    recording  = data.get("recording_triggered",    False)
    escalation = data.get("escalation_recommended", False)

    actions = []

    if confidence < 0.6:
        actions.append("clarify")
    elif severity == "HIGH":
        actions.extend(["alert", "guidance"])
    elif severity == "MEDIUM":
        actions.extend(["notify", "guidance"])
    else:
        actions.append("guidance")

    if recording:  actions.append("record")
    if escalation: actions.append("escalate")

    state["actions"] = actions
    return state

def alert_node(state: AgentState) -> AgentState:
    d     = state["classification"]
    lines = [f"🚨 EMERGENCY — {d['guidance_title']}"]
    for i, step in enumerate(d.get("guidance_steps", []), 1):
        lines.append(f"{i}. {step}")
    if "record"   in state["actions"]: lines.append("\n🎙️ Audio recording started.")
    if "escalate" in state["actions"]: lines.append(f"📞 Escalating to {d['escalation_target']}.")
    state["display_text"] = "\n".join(lines)
    print("EMERGENCY ALERT ACTIVATED")
    return state

def notify_node(state: AgentState) -> AgentState:
    d     = state["classification"]
    lines = [f"⚠️  {d['guidance_title']}"]
    for i, step in enumerate(d.get("guidance_steps", []), 1):
        lines.append(f"{i}. {step}")
    if "escalate" in state["actions"]: lines.append(f"📞 Escalating to {d['escalation_target']}.")
    state["display_text"] = "\n".join(lines)
    print("Sending warning notification")
    return state

def guidance_node(state: AgentState) -> AgentState:
    d     = state["classification"]
    lines = [f"ℹ️  {d['guidance_title']}"]
    for i, step in enumerate(d.get("guidance_steps", []), 1):
        lines.append(f"{i}. {step}")
    state["display_text"] = "\n".join(lines)
    print("Showing safety guidance...")
    return state

def clarify_node(state: AgentState) -> AgentState:
    state["display_text"] = "Could you describe what's happening in more detail?"
    print("Asking user for clarification")
    return state

def route_actions(state: AgentState) -> str:
    actions = state["actions"]
    if "clarify" in actions: return "clarify"
    if "alert"   in actions: return "alert"
    if "notify"  in actions: return "notify"
    return "guidance"

# ── Build graph ────────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("classifier", classify_node)
builder.add_node("decision",   decision_node)
builder.add_node("guidance",   guidance_node)
builder.add_node("notify",     notify_node)
builder.add_node("alert",      alert_node)
builder.add_node("clarify",    clarify_node)

builder.set_entry_point("classifier")
builder.add_edge("classifier", "decision")
builder.add_conditional_edges(
    "decision",
    route_actions,
    {
        "guidance": "guidance",
        "notify":   "notify",
        "alert":    "alert",
        "clarify":  "clarify"
    }
)

# Every terminal node connects to END
builder.add_edge("guidance", END)
builder.add_edge("notify",   END)
builder.add_edge("alert",    END)
builder.add_edge("clarify",  END)

graph = builder.compile()

# ── API endpoints ──────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: QueryRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        start  = time.time()
        state  = {
            "user_query":     req.message,
            "classification": {},
            "actions":        [],
            "display_text":   ""
        }
        result = graph.invoke(state)
        return {
            "display_text":   result["display_text"],
            "actions":        result["actions"],
            "classification": result["classification"],
            "latency_ms":     round((time.time() - start) * 1000)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))