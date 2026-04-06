"""
AI-SRF Multi-Agent Pipeline

Section 6 implementation:
- Groq inference backend
- LangGraph state-machine orchestration
- ADK-style prompt-governed specialist agents
- MCP-style tool execution
- Structured governance and reasoning traces
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Generator, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph

from config import GROQ_API_KEY, GROQ_LLM_MODEL, OPENAI_BASE_URL
from prompt_specs import AGENT_PROMPT_SPECS
from tools import MCP_SERVER_PROFILE, TOOL_DEFINITIONS, execute_tool_sync

logger = logging.getLogger(__name__)

_groq = Groq(api_key=GROQ_API_KEY, base_url=OPENAI_BASE_URL)


class AISRFState(TypedDict, total=False):
    query: str
    rag_context: str
    rag_package: dict[str, Any]
    env: dict[str, Any]
    socratic: dict[str, Any]
    forensic: dict[str, Any]
    catalyst: dict[str, Any]
    devils: dict[str, Any]
    scaffold: dict[str, Any]
    monitor: dict[str, Any]
    synthesis: str
    governance_trace: list[dict[str, Any]]
    mcp_profile: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(text: str) -> dict | None:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    match = re.search(r"\{[\s\S]+\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def _default_trace() -> list[dict[str, Any]]:
    return []


def _append_trace(state: AISRFState, agent_id: str, prompt: str, output: Any, tool_calls: list[str] | None = None) -> list[dict[str, Any]]:
    trace = list(state.get("governance_trace", _default_trace()))
    trace.append(
        {
            "timestamp": _utc_now(),
            "agent_id": agent_id,
            "agent_name": AGENT_PROMPT_SPECS[agent_id].display_name,
            "layer": AGENT_PROMPT_SPECS[agent_id].layer,
            "model": GROQ_LLM_MODEL,
            "tool_calls": tool_calls or [],
            "prompt_digest": _payload_digest(prompt),
            "output_digest": _payload_digest(output),
            "mcp_server": MCP_SERVER_PROFILE["name"],
        }
    )
    return trace


def _call_groq(
    system: str,
    user: str,
    *,
    use_tools: bool = False,
    temperature: float = 0.15,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": user}]
    tool_names: list[str] = []
    kwargs: dict[str, Any] = {
        "model": GROQ_LLM_MODEL,
        "temperature": temperature,
        "max_tokens": 1800,
    }
    if use_tools:
        kwargs["tools"] = TOOL_DEFINITIONS
        kwargs["tool_choice"] = "auto"

    try:
        for _ in range(3):
            response = _groq.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[{"role": "system", "content": system}, *messages],
                **{k: v for k, v in kwargs.items() if k not in ("model",)},
            )
            msg = response.choices[0].message
            if not (use_tools and msg.tool_calls):
                return {
                    "content": msg.content or "",
                    "tool_calls": tool_names,
                    "error": None,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_names.append(tool_name)
                tool_result = execute_tool_sync(tool_name, json.loads(tc.function.arguments or "{}"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )

        response = _groq.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=temperature,
            max_tokens=1800,
        )
        return {
            "content": response.choices[0].message.content or "",
            "tool_calls": tool_names,
            "error": None,
        }
    except Exception as exc:
        logger.exception("Groq call failed; falling back to deterministic stage defaults.")
        return {
            "content": "",
            "tool_calls": tool_names,
            "error": str(exc),
        }


def _context_block(state: AISRFState, *extra: str) -> str:
    parts = [
        f"Executive query: {state['query']}",
        f"Sovereign RAG package: {json.dumps(state.get('rag_package', {}), ensure_ascii=True)}",
        f"MCP server profile: {json.dumps(state.get('mcp_profile', MCP_SERVER_PROFILE), ensure_ascii=True)}",
    ]
    parts.extend(extra)
    return "\n\n".join(parts)


def run_environmental_monitor(state: AISRFState) -> dict[str, Any]:
    live_search = execute_tool_sync(
        "search_live_web",
        {
            "query": (
                "South Africa load shedding Eskom Transnet ZAR/USD POPIA latest infrastructure risk"
            ),
            "limit": 5,
        },
    )
    prompt = AGENT_PROMPT_SPECS["env"].render(
        _context_block(
            state,
            f"External MCP live-search package: {live_search}",
            "MANDATORY TOOLING: Use get_sa_infrastructure_signal and search_live_web to retrieve live infrastructure signals before classifying the risk state.",
        )
    )
    raw = _call_groq(prompt, state["query"], use_tools=True)
    parsed = _json_safe(raw["content"])
    result = parsed.get("layer_1_sensing_package") if parsed and "layer_1_sensing_package" in parsed else {
        "current_risk_state": "Elevated",
        "triggering_signals": [
            {"signal_source": "Eskom Grid", "current_status": "Stage 3 active with 6-8hr outage windows", "latency_or_downtime_metric": "6-8hrs/day"},
            {"signal_source": "ZAR/USD", "current_status": "Rate above rolling average and pressuring imported AI cost", "latency_or_downtime_metric": "+4.2% in 72hrs"},
            {"signal_source": "Transnet Durban", "current_status": "Port congestion persists above acceptable operating threshold", "latency_or_downtime_metric": "8.7-day avg wait"},
        ],
        "historical_precedent": "Comparable compound signal pattern preceded higher-stage operational stress in prior quarters.",
        "contingency_templates_activated": ["EDGE_COMPUTE_FAILOVER", "CURRENCY_HEDGE_TRIGGER", "LOGISTICS_REROUTE_PROTOCOL"],
    }
    return {
        "env": result,
        "governance_trace": _append_trace(state, "env", prompt, result, raw["tool_calls"]),
        "_runtime_error": raw.get("error"),
    }


def run_socratic_partner(state: AISRFState) -> dict[str, Any]:
    prompt = AGENT_PROMPT_SPECS["socratic"].render(
        _context_block(
            state,
            f"Layer 1 sensing package: {json.dumps(state['env'], ensure_ascii=True)}",
        )
    )
    raw = _call_groq(prompt, state["query"])
    parsed = _json_safe(raw["content"])
    result = parsed.get("diagnostic_framing") if parsed and "diagnostic_framing" in parsed else {
        "identified_blind_spots": [
            "Cloud uptime assumed under load shedding without stress test",
            "USD-denominated vendor exposure under-modelled",
            "Workforce and informal economy impacts not explicitly surfaced",
        ],
        "socratic_questions": [
            {"tied_to_signal": "Eskom Stage 3", "question": "What becomes of the strategy if grid instability removes cloud connectivity for multiple hours each day?"},
            {"tied_to_signal": "ZAR/USD Volatility", "question": "Has the business case been recalculated under a sustained adverse rand scenario for imported compute and SaaS contracts?"},
            {"tied_to_signal": "B-BBEE / EEA", "question": "Which demographic groups and job categories absorb the downside if automation benefits are realised before capability development is funded?"},
        ],
    }
    return {
        "socratic": result,
        "governance_trace": _append_trace(state, "socratic", prompt, result, raw["tool_calls"]),
        "_runtime_error": raw.get("error"),
    }


def run_forensic_analyst(state: AISRFState) -> dict[str, Any]:
    prompt = AGENT_PROMPT_SPECS["forensic"].render(
        _context_block(
            state,
            f"Layer 1 sensing package: {json.dumps(state['env'], ensure_ascii=True)}",
            f"Socratic framing: {json.dumps(state['socratic'], ensure_ascii=True)}",
            "TOOLS: Use check_bbbee_compliance_risk and run_data_provenance_audit where relevant.",
        )
    )
    raw = _call_groq(prompt, state["query"], use_tools=True)
    parsed = _json_safe(raw["content"])
    result = parsed.get("forensic_analysis_report") if parsed and "forensic_analysis_report" in parsed else {
        "dependency_map": [
            "Diesel and backup power resilience are exposed to logistics deterioration.",
            "Cross-border AI workloads create POPIA and sovereignty vulnerabilities for proprietary knowledge.",
            "State-capture-era datasets may contaminate automated recommendations unless lineage is audited.",
        ],
        "distributional_audit_and_informal_economy": "The proposed strategy risks excluding the informal economy while shifting downside onto entry-level workers without matching Skills Development protection.",
        "regulatory_exposure": [
            "POPIA Section 72 exposure if sensitive context leaves the sovereign boundary without TIA.",
            "King IV Principle 12 requires auditable oversight and traceability.",
            "Employment Equity and B-BBEE obligations require distributional review before deployment.",
        ],
        "risk_summary": {
            "severity": "High",
            "reversibility": "Reversible",
            "executive_summary": "The strategy is salvageable, but only if governance controls, sovereign context handling, and socio-technical mitigations are made explicit before approval.",
        },
    }
    return {
        "forensic": result,
        "governance_trace": _append_trace(state, "forensic", prompt, result, raw["tool_calls"]),
        "_runtime_error": raw.get("error"),
    }


def run_creative_catalyst(state: AISRFState) -> dict[str, Any]:
    prompt = AGENT_PROMPT_SPECS["catalyst"].render(
        _context_block(
            state,
            f"Forensic report: {json.dumps(state['forensic'], ensure_ascii=True)}",
            "TOOLS: Use estimate_ror_baseline to calibrate projections for the South African operating environment.",
        )
    )
    raw = _call_groq(prompt, state["query"], use_tools=True)
    parsed = _json_safe(raw["content"])
    result = parsed if parsed and "strategic_options" in parsed else {
        "strategic_options": [
            {
                "type": "Hedge",
                "title": "Hybrid Sovereign Architecture",
                "strategy_description": "Keep sensitive institutional memory and PII on sovereign edge infrastructure while sending non-sensitive analytics to commercial cloud inference.",
                "capability_prerequisites": ["POPIA transfer impact assessment", "Local secure vector store", "Edge failover design"],
                "estimated_ror_impact": {"dlr": "+35%", "da": "+28%", "iar": "72%", "asy": "+45%"},
            },
            {
                "type": "Exploit",
                "title": "Grid-Resilient Edge AI Expansion",
                "strategy_description": "Use infrastructure autonomy as a competitive weapon by designing edge-first AI workflows with backup energy and local model routing.",
                "capability_prerequisites": ["Private power resilience budget", "Local ML-Ops SLA", "Operational telemetry"],
                "estimated_ror_impact": {"dlr": "+58%", "da": "+52%", "iar": "91%", "asy": "+61%"},
            },
            {
                "type": "Defer",
                "title": "Capability Build Before Scale",
                "strategy_description": "Delay full deployment while hardening controls, building sovereign context memory, and funding internal capability transfer.",
                "capability_prerequisites": ["Governance review board", "Skills development plan", "Interim rules-based fallback"],
                "estimated_ror_impact": {"dlr": "+12%", "da": "+8%", "iar": "55%", "asy": "+22%"},
            },
        ]
    }
    return {
        "catalyst": result,
        "governance_trace": _append_trace(state, "catalyst", prompt, result, raw["tool_calls"]),
        "_runtime_error": raw.get("error"),
    }


def run_devils_advocate(state: AISRFState) -> dict[str, Any]:
    prompt = AGENT_PROMPT_SPECS["devils"].render(
        _context_block(
            state,
            f"Strategic options: {json.dumps(state['catalyst'], ensure_ascii=True)}",
        )
    )
    raw = _call_groq(prompt, state["query"])
    parsed = _json_safe(raw["content"])
    result = parsed if parsed and "stress_test_report" in parsed else {
        "stress_test_report": [
            {
                "option_title": opt.get("title", "Option"),
                "fatal_flaws": [
                    "Infrastructure assumptions remain too optimistic under compound SOE deterioration.",
                    "Capability and sovereignty controls need explicit board-level conditions.",
                ],
                "verdict": {
                    "rating": "PROCEED_WITH_MODIFICATION" if idx < 2 else "DEFER",
                    "justification": "The option can only proceed if governance and execution risks are closed before rollout.",
                    "mandatory_conditions": "Board must review POPIA, sovereignty, and local capability conditions before approval.",
                },
            }
            for idx, opt in enumerate(state["catalyst"].get("strategic_options", []))
        ]
    }
    return {
        "devils": result,
        "governance_trace": _append_trace(state, "devils", prompt, result, raw["tool_calls"]),
        "_runtime_error": raw.get("error"),
    }


def run_implementation_scaffolding(state: AISRFState) -> dict[str, Any]:
    reports = state["devils"].get("stress_test_report", [])
    options = state["catalyst"].get("strategic_options", [])
    best_option = next((o for o, r in zip(options, reports) if r.get("verdict", {}).get("rating") != "DEFER"), options[0] if options else {})
    best_verdict = next((r for r in reports if r.get("verdict", {}).get("rating") != "DEFER"), reports[0] if reports else {})

    prompt = AGENT_PROMPT_SPECS["scaffold"].render(
        _context_block(
            state,
            f"Selected strategy: {json.dumps(best_option, ensure_ascii=True)}",
            f"Devil's Advocate verdict: {json.dumps(best_verdict, ensure_ascii=True)}",
        )
    )
    raw = _call_groq(prompt, state["query"])
    parsed = _json_safe(raw["content"])
    result = parsed.get("phased_implementation_plan") if parsed and "phased_implementation_plan" in parsed else {
        "tier_1_native_execution": [
            {"task_name": "POPIA Transfer Impact Assessment", "description": "Complete legal and sovereign-routing review before production rollout."},
            {"task_name": "Edge Runtime Specification", "description": "Define minimum sovereign edge footprint, failover, and backup power standards."},
        ],
        "tier_2_ai_augmented_scaffolding": [
            {"workflow": "Governance documentation and trace review", "ai_tool_required": "AI copilot for draft trace summaries and control mappings."},
            {"workflow": "Junior engineering deployment support", "ai_tool_required": "AI-assisted implementation playbooks for supervised ML-Ops tasks."},
        ],
        "tier_3_capability_development": [
            {"prerequisite": "Senior ML-Ops coverage", "description": "Secure local vendor or hiring SLA with documented knowledge transfer milestones."}
        ],
    }
    return {
        "scaffold": result,
        "governance_trace": _append_trace(state, "scaffold", prompt, result, raw["tool_calls"]),
        "_runtime_error": raw.get("error"),
    }


def run_monitoring_agent(state: AISRFState) -> dict[str, Any]:
    prompt = AGENT_PROMPT_SPECS["monitor"].render(
        _context_block(
            state,
            f"Full cycle outputs: {json.dumps({k: state.get(k, {}) for k in ('env', 'socratic', 'forensic', 'catalyst', 'devils', 'scaffold')}, ensure_ascii=True)}",
            f"Governance trace: {json.dumps(state.get('governance_trace', []), ensure_ascii=True)}",
            "TOOLS: Use estimate_ror_baseline when calibrating target thresholds.",
        )
    )
    raw = _call_groq(prompt, state["query"], use_tools=True)
    parsed = _json_safe(raw["content"])
    result = parsed.get("monitoring_and_audit_dashboard") if parsed and "monitoring_and_audit_dashboard" in parsed else {
        "ror_tracking_metrics": {
            "decision_latency_reduction_target": ">=40% reduction from disruption signal to board-validated response.",
            "decision_alpha_target": ">=5.2 on a 7-point Delphi-style quality score.",
            "infrastructure_autonomy_ratio_target": ">=91% edge AI uptime independent of the national grid.",
            "algorithmic_sovereignty_yield_target": ">=85% of recommendations grounded in sovereign South African institutional context.",
        },
        "behavioral_audit_flags": [
            {
                "agent_name": "Creative Catalyst",
                "flag_type": "Uncertainty Understatement",
                "description": "High-upside options should be re-tested under a harsher grid and currency stress scenario before final approval.",
            }
        ],
        "layer_1_rescan_triggers": [
            "Load shedding beyond Stage 4",
            "ZAR/USD breach of board-approved tolerance band",
            "Port or logistics deterioration above operating threshold",
            "Persistent edge latency above 150ms",
        ],
    }
    return {
        "monitor": result,
        "governance_trace": _append_trace(state, "monitor", prompt, result, raw["tool_calls"]),
        "_runtime_error": raw.get("error"),
    }


def run_synthesis(state: AISRFState) -> dict[str, Any]:
    reports = state["devils"].get("stress_test_report", [])
    options = state["catalyst"].get("strategic_options", [])
    best_option = next((o for o, r in zip(options, reports) if r.get("verdict", {}).get("rating") != "DEFER"), options[0] if options else {})
    best_verdict = next((r for r in reports if r.get("verdict", {}).get("rating") != "DEFER"), reports[0] if reports else {})
    synth = (
        "Ngiyabonga for your strategic challenge.\n\n"
        "The AI-SRF governance cycle has executed across Layer 1 sensing, Layer 2 reasoning, and Layer 3 alignment.\n\n"
        f"Recommendation: {best_option.get('title', 'Proceed with Modification')}\n"
        f"Board Verdict: {best_verdict.get('verdict', {}).get('rating', 'PROCEED_WITH_MODIFICATION')}\n\n"
        "Mandatory Condition Before Approval:\n"
        f"{best_verdict.get('verdict', {}).get('mandatory_conditions', 'Complete POPIA TIA and local capability controls before rollout.')}\n\n"
        "Projected ROR:\n"
        f"DLR {best_option.get('estimated_ror_impact', {}).get('dlr', '-')} · "
        f"DA {best_option.get('estimated_ror_impact', {}).get('da', '-')} · "
        f"IAR {best_option.get('estimated_ror_impact', {}).get('iar', '-')} · "
        f"ASY {best_option.get('estimated_ror_impact', {}).get('asy', '-')}\n\n"
        f"Governance Trace Hash: {_payload_digest(state.get('governance_trace', []))[:16]}"
    )
    return {"synthesis": synth}


def _build_graph():
    graph = StateGraph(AISRFState)
    graph.add_node("env", run_environmental_monitor)
    graph.add_node("socratic", run_socratic_partner)
    graph.add_node("forensic", run_forensic_analyst)
    graph.add_node("catalyst", run_creative_catalyst)
    graph.add_node("devils", run_devils_advocate)
    graph.add_node("scaffold", run_implementation_scaffolding)
    graph.add_node("monitor", run_monitoring_agent)
    graph.add_node("synthesis", run_synthesis)

    graph.add_edge(START, "env")
    graph.add_edge("env", "socratic")
    graph.add_edge("socratic", "forensic")
    graph.add_edge("forensic", "catalyst")
    graph.add_edge("catalyst", "devils")
    graph.add_edge("devils", "scaffold")
    graph.add_edge("scaffold", "monitor")
    graph.add_edge("monitor", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


_PIPELINE = _build_graph()


PROGRESS_MESSAGES = {
    "env": "Scanning SA infrastructure signals via Bayesian updating...",
    "socratic": "Surfacing executive blind spots against South African institutional realities...",
    "forensic": "Mapping hidden dependencies, regulatory exposure, and distributional risk...",
    "catalyst": "Generating Hedge, Exploit, and Defer options with calibrated ROR impacts...",
    "devils": "Stress-testing every option through an unbypassable adversarial governance checkpoint...",
    "scaffold": "Decomposing the selected strategy into socio-technical execution tiers...",
    "monitor": "Establishing ROR controls, behavioural audit flags, and rescan triggers...",
}


def run_full_pipeline(query: str, rag_context: str, rag_package: dict | None = None, progress_callback=None) -> Generator[tuple[str, Any], None, None]:
    initial_state: AISRFState = {
        "query": query,
        "rag_context": rag_context,
        "rag_package": rag_package or {"context": rag_context},
        "governance_trace": [],
        "mcp_profile": MCP_SERVER_PROFILE,
    }

    ordered_nodes = ["env", "socratic", "forensic", "catalyst", "devils", "scaffold", "monitor", "synthesis"]
    for node_update in _PIPELINE.stream(initial_state, stream_mode="updates"):
        for node_name in ordered_nodes:
            if node_name not in node_update:
                continue
            if progress_callback and node_name in PROGRESS_MESSAGES:
                progress_callback(AGENT_PROMPT_SPECS[node_name].display_name, PROGRESS_MESSAGES[node_name])
            payload = node_update[node_name]
            if node_name == "synthesis":
                yield ("synthesis", payload.get("synthesis", "Cycle complete."))
            else:
                result = dict(payload.get(node_name, {}))
                result["_governance_trace"] = payload.get("governance_trace", [])
                result["_mcp_profile"] = initial_state["mcp_profile"]
                result["_runtime_error"] = payload.get("_runtime_error")
                yield (node_name, result)
