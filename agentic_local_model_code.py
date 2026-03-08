from llama_cpp import Llama, LlamaGrammar
import json
from typing import TypedDict, List

MODEL_PATH = "llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    verbose=False
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "incident_type": {
            "type": "string",
            "enum": ["SAFETY_THREAT", "MEDICAL_EMERGENCY", "PANIC_DISTRESS",
                     "LOST_DISORIENTED", "SUSPICIOUS_ACTIVITY", "GENERAL_HELP"]
        },
        "severity": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH"]
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason":         {"type": "string"},
        "guidance_title": {"type": "string"},
        "guidance_steps": {"type": "array", "items": {"type": "string"}},
        "recording_triggered":    {"type": "boolean"},
        "escalation_recommended": {"type": "boolean"},
        "escalation_target": {
            "type": "string",
            "enum": ["Campus Security", "Medical Services", "None"]
        }
    },
    "required": [
        "incident_type", "severity", "confidence", "reason",
        "guidance_title", "guidance_steps",
        "recording_triggered", "escalation_recommended", "escalation_target"
    ]
}

grammar = LlamaGrammar.from_json_schema(json.dumps(JSON_SCHEMA))

# ── Few-shot prompt — small models need EXAMPLES not just rules ────────────────
SYSTEM_PROMPT = """You are a campus safety classifier. Output JSON only.

EXAMPLES:

Input: "someone is following me and I am scared"
Output: {"incident_type":"SAFETY_THREAT","severity":"HIGH","confidence":0.95,"reason":"User is being followed and feels unsafe","guidance_title":"Move to safety immediately","guidance_steps":["Move to a crowded well-lit area","Call campus security","Stay on the phone with someone"],"recording_triggered":true,"escalation_recommended":true,"escalation_target":"Campus Security"}

Input: "I feel anxious about my exam"
Output: {"incident_type":"PANIC_DISTRESS","severity":"LOW","confidence":0.85,"reason":"User is stressed about academic pressure","guidance_title":"Calm down and breathe","guidance_steps":["Take slow deep breaths","Talk to a friend or counselor","Break the task into small steps"],"recording_triggered":false,"escalation_recommended":false,"escalation_target":"None"}

Input: "I fell down and my leg is bleeding badly"
Output: {"incident_type":"MEDICAL_EMERGENCY","severity":"HIGH","confidence":0.97,"reason":"User has a bleeding injury and needs medical help","guidance_title":"Get medical help now","guidance_steps":["Apply pressure to the wound","Call campus medical services","Do not move if in pain"],"recording_triggered":false,"escalation_recommended":true,"escalation_target":"Medical Services"}

Input: "I can't find the exam hall I am lost"
Output: {"incident_type":"LOST_DISORIENTED","severity":"LOW","confidence":0.9,"reason":"User cannot find their location on campus","guidance_title":"Find your way","guidance_steps":["Check the campus map","Ask a nearby student or staff","Call the campus helpline"],"recording_triggered":false,"escalation_recommended":false,"escalation_target":"None"}

Input: "there is a suspicious man standing outside the library staring at students"
Output: {"incident_type":"SUSPICIOUS_ACTIVITY","severity":"MEDIUM","confidence":0.88,"reason":"Suspicious individual reported near campus building","guidance_title":"Report suspicious activity","guidance_steps":["Do not approach the person","Move away from the area","Report to campus security immediately"],"recording_triggered":true,"escalation_recommended":true,"escalation_target":"Campus Security"}

Input: "I need help finding the counseling center"
Output: {"incident_type":"GENERAL_HELP","severity":"LOW","confidence":0.92,"reason":"User needs directions to campus support service","guidance_title":"Finding the counseling center","guidance_steps":["Check the campus app or website","Ask at the front desk of any building","Call the main campus number"],"recording_triggered":false,"escalation_recommended":false,"escalation_target":"None"}

RULES:
- anxious, stressed, overwhelmed, scared feeling = PANIC_DISTRESS, LOW or MEDIUM severity
- physical violence, attack, stalking = SAFETY_THREAT, HIGH severity
- injury, bleeding, chest pain, collapse = MEDICAL_EMERGENCY, HIGH severity
- lost, confused location = LOST_DISORIENTED, LOW severity
- suspicious person = SUSPICIOUS_ACTIVITY, MEDIUM severity
- general question = GENERAL_HELP, LOW severity
- recording_triggered = true ONLY for HIGH severity threats
- escalation_recommended = true ONLY for HIGH severity
- Output JSON only. No explanation. No markdown."""


def analyze_query(user_query: str) -> dict:
    print("DEBUG USER QUERY:", user_query)

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Input: \"{user_query}\"\nOutput:"}
        ],
        temperature=0,
        top_p=0.1,
        max_tokens=400,
        grammar=grammar
    )

    result = response["choices"][0]["message"]["content"]

    try:
        return json.loads(result)
    except Exception:
        print("RAW MODEL OUTPUT:\n", result)
        return {"error": "Model did not return valid JSON"}


# ── LangGraph ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_query:     str
    classification: dict
    actions:        List[str]
    display_text:   str

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
    actions    = []

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
    print(state["display_text"])
    return state

def notify_node(state: AgentState) -> AgentState:
    d     = state["classification"]
    lines = [f"⚠️  {d['guidance_title']}"]
    for i, step in enumerate(d.get("guidance_steps", []), 1):
        lines.append(f"{i}. {step}")
    if "escalate" in state["actions"]: lines.append(f"📞 Escalating to {d['escalation_target']}.")
    state["display_text"] = "\n".join(lines)
    print(state["display_text"])
    return state

def guidance_node(state: AgentState) -> AgentState:
    d     = state["classification"]
    lines = [f"ℹ️  {d['guidance_title']}"]
    for i, step in enumerate(d.get("guidance_steps", []), 1):
        lines.append(f"{i}. {step}")
    state["display_text"] = "\n".join(lines)
    print(state["display_text"])
    return state

def clarify_node(state: AgentState) -> AgentState:
    state["display_text"] = "Could you describe what's happening in more detail?"
    print(state["display_text"])
    return state

def route_actions(state: AgentState) -> str:
    actions = state["actions"]
    if "clarify" in actions: return "clarify"
    if "alert"   in actions: return "alert"
    if "notify"  in actions: return "notify"
    return "guidance"

from langgraph.graph import StateGraph, END

builder = StateGraph(AgentState)
builder.add_node("classifier", classify_node)
builder.add_node("decision",   decision_node)
builder.add_node("guidance",   guidance_node)
builder.add_node("notify",     notify_node)
builder.add_node("alert",      alert_node)
builder.add_node("clarify",    clarify_node)

builder.set_entry_point("classifier")
builder.add_edge("classifier", "decision")
builder.add_conditional_edges("decision", route_actions, {
    "guidance": "guidance",
    "notify":   "notify",
    "alert":    "alert",
    "clarify":  "clarify"
})
builder.add_edge("guidance", END)
builder.add_edge("notify",   END)
builder.add_edge("alert",    END)
builder.add_edge("clarify",  END)

graph = builder.compile()

# ── Run ────────────────────────────────────────────────────────────────────────
state = {
    "user_query":   input("User: "),
    "classification": {},
    "actions":        [],
    "display_text":   ""
}

graph.invoke(state)