"""
AI-SRF Function-Calling Tools (MCP-style)
Implements: World Bank API, Stats SA, Eskom signal, ZAR/USD, data provenance audit
By: Bright Sikazwe, PhD Candidate
"""

import json
import httpx
import logging
from datetime import datetime
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

from mcp_bridge import get_configured_servers, search_live_web

MCP_SERVER_PROFILE = {
    "name": "ai-srf-governance-mcp",
    "transport": "in-process",
    "capabilities": [
        "infrastructure-signal retrieval",
        "regulatory anchor injection",
        "POPIA ingestion gate review",
        "ROR baseline estimation",
        "B-BBEE and EEA governance checks",
        "data provenance audit",
    ],
    "compliance_scope": ["POPIA", "King IV", "B-BBEE", "EEA", "Algorithmic Sovereignty"],
    "external_servers": sorted(get_configured_servers().keys()),
}

# ─── Tool definitions (Groq function-calling schema) ────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_live_web",
            "description": (
                "Uses configured MCP search or fetch servers to retrieve current live web results. "
                "Prefer for up-to-date infrastructure, regulation, or market signals when an MCP "
                "server such as Exa, Firecrawl, or Fetch is configured."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query or URL for fetch-style servers."},
                    "limit": {"type": "integer", "description": "Maximum result count to request."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_world_bank_sa_indicator",
            "description": (
                "Retrieves current South African macroeconomic indicators from the World Bank Open Data API. "
                "Use for GDP per capita, Gini coefficient, unemployment, digital economy statistics, "
                "broadband penetration, and structural inequality data. Only queries South Africa (ZAF)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator_code": {
                        "type": "string",
                        "description": (
                            "World Bank indicator code. Examples: "
                            "NY.GDP.PCAP.CD (GDP per capita), "
                            "SI.POV.GINI (Gini coefficient), "
                            "SL.UEM.TOTL.ZS (unemployment rate), "
                            "IT.NET.USER.ZS (internet users %), "
                            "IT.NET.BBND.P2 (fixed broadband subscriptions)"
                        ),
                    },
                    "years": {
                        "type": "integer",
                        "description": "Number of recent years to retrieve (default 3)",
                    },
                },
                "required": ["indicator_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sa_infrastructure_signal",
            "description": (
                "Returns current South African infrastructure risk signals covering Eskom load shedding "
                "stage, Transnet logistics status, ZAR/USD exchange rate, and rural broadband latency. "
                "Used by the Environmental Monitor Agent for Bayesian risk state classification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_type": {
                        "type": "string",
                        "enum": ["eskom", "transnet", "currency", "broadband", "all"],
                        "description": "Which infrastructure signal to retrieve",
                    }
                },
                "required": ["signal_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_regulatory_anchor_bundle",
            "description": (
                "Returns mandatory South African governance anchors to inject into a RAG context "
                "window before agent reasoning. Covers POPIA, King IV, B-BBEE, and EEA obligations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_laws": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of anchor domains to prioritise",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_popia_ingestion_gate",
            "description": (
                "Performs a lightweight POPIA gate review for a candidate RAG document or chunk. "
                "Flags likely residual personal information and returns an allow or review recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_name": {"type": "string"},
                    "text_excerpt": {"type": "string"},
                },
                "required": ["document_name", "text_excerpt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_data_provenance_audit",
            "description": (
                "Executes a data provenance audit on a proposed data source, checking for state-capture "
                "contamination risk, POPIA compliance status, Zondo Commission exposure, and data "
                "integrity score. Required before incorporating any public sector data into AI training."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_source": {
                        "type": "string",
                        "description": "Name or description of the data source to audit",
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["government", "soe", "private", "international", "social_media"],
                        "description": "Category of data source",
                    },
                },
                "required": ["data_source", "data_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_ror_baseline",
            "description": (
                "Estimates the Return on Resilience (ROR) baseline for an organisation given its "
                "sector, digital maturity stage, and current infrastructure exposure. Returns "
                "baseline values for DLR, DA, IAR, and ASY indicators."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "enum": ["banking", "mining", "retail", "telecom", "manufacturing", "healthcare", "government"],
                    },
                    "digital_maturity": {
                        "type": "string",
                        "enum": ["nascent", "emerging", "defined", "managed", "optimising"],
                        "description": "BUSA digital maturity taxonomy stage",
                    },
                    "load_shedding_stage": {
                        "type": "integer",
                        "description": "Current Eskom load shedding stage (0–8)",
                    },
                },
                "required": ["sector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_bbbee_compliance_risk",
            "description": (
                "Assesses B-BBEE and Employment Equity Act compliance risk for a proposed AI deployment. "
                "Checks for proxy variable discrimination, workforce displacement risk, and Skills "
                "Development scorecard impact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "deployment_type": {
                        "type": "string",
                        "description": "Description of the AI deployment (e.g., CV screening, credit scoring)",
                    },
                    "affected_roles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Job roles that will be automated or significantly changed",
                    },
                    "training_data_source": {
                        "type": "string",
                        "description": "Description of training data (e.g., historical HR data, formal bank records)",
                    },
                },
                "required": ["deployment_type"],
            },
        },
    },
]


# ─── Tool execution functions ─────────────────────────────────────────────────

async def execute_tool(tool_name: str, tool_args: dict) -> dict[str, Any]:
    """Dispatch to the appropriate tool function."""
    dispatch = {
        "search_live_web": _mcp_live_search,
        "get_world_bank_sa_indicator": _world_bank_indicator,
        "get_sa_infrastructure_signal": _infrastructure_signal,
        "get_regulatory_anchor_bundle": _regulatory_anchor_bundle,
        "verify_popia_ingestion_gate": _verify_popia_ingestion_gate,
        "run_data_provenance_audit": _data_provenance_audit,
        "estimate_ror_baseline": _ror_baseline,
        "check_bbbee_compliance_risk": _bbbee_compliance,
    }
    fn = dispatch.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return await fn(**tool_args)
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, e)
        return {"error": str(e), "tool": tool_name}


async def _mcp_live_search(query: str, limit: int = 5) -> dict:
    result = search_live_web(query=query, limit=limit)
    result["configured_servers"] = sorted(get_configured_servers().keys())
    result["config_source"] = next(
        (str(path) for path in [Path(__file__).resolve().parents[1] / "mcp_servers.json", Path(__file__).resolve().parents[1] / "mcp_servers.example.json"] if path.exists()),
        None,
    )
    return result


async def _world_bank_indicator(indicator_code: str, years: int = 3) -> dict:
    """Fetch live data from World Bank API for South Africa."""
    try:
        url = (
            f"https://api.worldbank.org/v2/country/ZAF/indicator/{indicator_code}"
            f"?format=json&mrv={years}&per_page=5"
        )
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                if len(data) > 1 and data[1]:
                    records = [
                        {
                            "year": rec.get("date"),
                            "value": rec.get("value"),
                            "indicator": rec.get("indicator", {}).get("value"),
                        }
                        for rec in data[1]
                        if rec.get("value") is not None
                    ]
                    return {
                        "source": "World Bank Open Data API",
                        "country": "South Africa",
                        "indicator": indicator_code,
                        "records": records,
                        "retrieved_at": datetime.utcnow().isoformat(),
                    }
    except Exception as e:
        logger.warning("World Bank API unavailable: %s — using cached values", e)

    # Fallback cached values for key indicators
    CACHED = {
        "NY.GDP.PCAP.CD": [{"year": "2023", "value": 6193, "indicator": "GDP per capita (USD)"}],
        "SI.POV.GINI":     [{"year": "2023", "value": 63.0, "indicator": "Gini index"}],
        "SL.UEM.TOTL.ZS":  [{"year": "2023", "value": 32.1, "indicator": "Unemployment, total (% of labour force)"}],
        "IT.NET.USER.ZS":  [{"year": "2023", "value": 72.4, "indicator": "Individuals using the Internet (% of population)"}],
        "IT.NET.BBND.P2":  [{"year": "2023", "value": 2.8,  "indicator": "Fixed broadband subscriptions (per 100 people)"}],
    }
    return {
        "source": "World Bank Open Data (cached — API unreachable)",
        "country": "South Africa",
        "indicator": indicator_code,
        "records": CACHED.get(indicator_code, [{"year": "2023", "value": "N/A", "indicator": indicator_code}]),
    }


async def _infrastructure_signal(signal_type: str = "all") -> dict:
    """
    Real-time SA infrastructure signals.
    Attempts EskomSePush API; falls back to contextually calibrated estimates.
    """
    signals = {}

    # --- Eskom (attempt EskomSePush) -----------------------------------------
    if signal_type in ("eskom", "all"):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get("https://loadshedding.eskom.co.za/LoadShedding/GetStatus")
                stage = int(r.text.strip()) - 1 if r.status_code == 200 else 3
        except Exception:
            stage = 3  # default contextual estimate
        signals["eskom"] = {
            "load_shedding_stage": stage,
            "daily_outage_hours": stage * 2,
            "diesel_cost_per_litre_zar": 21.0,
            "private_generation_capex_pct": "3–8%",
            "risk_level": "Critical" if stage >= 5 else "Compound" if stage >= 3 else "Elevated" if stage >= 1 else "Nominal",
        }

    # --- Transnet ---------------------------------------------------------------
    if signal_type in ("transnet", "all"):
        signals["transnet"] = {
            "durban_vessel_wait_days": 8.7,
            "locomotive_availability_pct": 47,
            "target_availability_pct": 70,
            "freight_rail_deficit_pct": 35,
            "logistics_data_integrity": "Structurally incomplete",
            "risk_level": "Compound",
        }

    # --- ZAR/USD ----------------------------------------------------------------
    if signal_type in ("currency", "all"):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    "https://api.exchangerate-api.com/v4/latest/USD",
                    timeout=4.0,
                )
                rate = r.json().get("rates", {}).get("ZAR", 18.9) if r.status_code == 200 else 18.9
        except Exception:
            rate = 18.9
        signals["currency"] = {
            "zar_usd_rate": round(rate, 2),
            "10yr_range": "R10.50–R19.70",
            "avg_annual_depreciation_pct": 4.8,
            "saas_cost_escalation_5yr_pct": "40–85%",
            "risk_level": "Elevated" if rate < 17 else "Compound" if rate < 20 else "Critical",
        }

    # --- Broadband --------------------------------------------------------------
    if signal_type in ("broadband", "all"):
        signals["broadband"] = {
            "urban_penetration_pct": 67,
            "rural_penetration_pct": 23,
            "avg_rural_latency_ms": 210,
            "mobile_internet_pct": 68.4,
            "fixed_broadband_households_pct": 14.1,
            "risk_level": "Elevated",
        }

    return {"signals": signals, "retrieved_at": datetime.utcnow().isoformat()}


async def _data_provenance_audit(data_source: str, data_type: str) -> dict:
    """Assess data provenance and state-capture contamination risk."""
    soe_sources = ["transnet", "eskom", "sanral", "denel", "prasa", "sabc", "saa", "post office"]
    gov_risk = any(s in data_source.lower() for s in soe_sources) or data_type in ("soe", "government")

    integrity_score = 45 if gov_risk else 78 if data_type == "private" else 88
    zondo_exposure = "High" if gov_risk else "Low"
    popia_risk = "High" if data_type in ("government", "soe") else "Medium"

    return {
        "data_source": data_source,
        "data_type": data_type,
        "integrity_score": integrity_score,
        "zondo_commission_exposure": zondo_exposure,
        "popia_compliance_risk": popia_risk,
        "state_capture_contamination": gov_risk,
        "recommendation": (
            "BLOCK: Conduct full data lineage audit before incorporating into AI training. "
            "State capture-era data produces AI outputs that reproduce historical distortions."
            if gov_risk else
            "PROCEED WITH MONITORING: Implement POPIA TIA and ongoing data quality validation."
        ),
        "mandatory_governance_actions": [
            "Complete Transfer Impact Assessment (TIA)" if popia_risk == "High" else None,
            "Commission independent data integrity audit" if gov_risk else None,
            "Document data lineage in AI System Card (King IV Principle 12)",
        ],
    }


async def _regulatory_anchor_bundle(include_laws: list | None = None) -> dict:
    requested = set(include_laws or ["POPIA", "King IV", "B-BBEE", "EEA"])
    anchors = {
        "POPIA": "POPIA Section 72 requires lawful basis and a Transfer Impact Assessment before cross-border transfer of personal information.",
        "King IV": "King IV Principle 12 requires auditable, non-delegable board oversight of technology and information governance.",
        "B-BBEE": "B-BBEE transformation and skills-development implications must be tested before workforce-affecting AI deployment.",
        "EEA": "The Employment Equity Act prohibits indirect discrimination caused by biased proxy variables and opaque automation.",
    }
    return {
        "anchors": [text for law, text in anchors.items() if law in requested],
        "retrieved_at": datetime.utcnow().isoformat(),
        "mcp_server": MCP_SERVER_PROFILE["name"],
    }


async def _verify_popia_ingestion_gate(document_name: str, text_excerpt: str) -> dict:
    excerpt = (text_excerpt or "").lower()
    pii_hits = [
        token for token in ["id number", "passport", "email", "phone", "cell", "address", "@", "dob", "birth"]
        if token in excerpt
    ]

    review_terms = ["cv", "medical", "bank account", "salary", "employee", "student number"]
    decision = "review" if pii_hits or any(term in excerpt for term in review_terms) else "allow"

    return {
        "document_name": document_name,
        "decision": decision,
        "pii_indicators": pii_hits,
        "reason": (
            "Residual personal-information indicators detected. Human review required before sovereign RAG ingestion."
            if decision == "review"
            else "No obvious personal-information indicators detected in supplied excerpt."
        ),
        "mcp_server": MCP_SERVER_PROFILE["name"],
    }


async def _ror_baseline(
    sector: str,
    digital_maturity: str = "emerging",
    load_shedding_stage: int = 3,
) -> dict:
    """Estimate ROR baselines by sector and maturity."""
    SECTOR_BASELINES = {
        "banking":       {"dlr_baseline_hrs": 68, "da_baseline": 3.8, "iar_baseline_pct": 58, "asy_baseline_pct": 42},
        "mining":        {"dlr_baseline_hrs": 94, "da_baseline": 3.2, "iar_baseline_pct": 44, "asy_baseline_pct": 31},
        "retail":        {"dlr_baseline_hrs": 52, "da_baseline": 4.1, "iar_baseline_pct": 61, "asy_baseline_pct": 48},
        "telecom":       {"dlr_baseline_hrs": 38, "da_baseline": 4.6, "iar_baseline_pct": 72, "asy_baseline_pct": 55},
        "manufacturing": {"dlr_baseline_hrs": 78, "da_baseline": 3.5, "iar_baseline_pct": 51, "asy_baseline_pct": 38},
        "healthcare":    {"dlr_baseline_hrs": 86, "da_baseline": 3.9, "iar_baseline_pct": 49, "asy_baseline_pct": 35},
        "government":    {"dlr_baseline_hrs": 120, "da_baseline": 2.8, "iar_baseline_pct": 33, "asy_baseline_pct": 22},
    }
    base = SECTOR_BASELINES.get(sector, SECTOR_BASELINES["retail"])

    # Penalise for load shedding stage
    iar_penalty = load_shedding_stage * 3
    dlr_penalty = load_shedding_stage * 4

    maturity_multiplier = {"nascent": 0.7, "emerging": 0.85, "defined": 1.0, "managed": 1.15, "optimising": 1.3}.get(digital_maturity, 1.0)

    return {
        "sector": sector,
        "digital_maturity": digital_maturity,
        "current_load_shedding_stage": load_shedding_stage,
        "baselines": {
            "decision_latency_hrs": int(base["dlr_baseline_hrs"] * (1 + dlr_penalty / 100)),
            "decision_alpha_likert": round(base["da_baseline"] * maturity_multiplier, 1),
            "infrastructure_autonomy_pct": max(0, base["iar_baseline_pct"] - iar_penalty),
            "algorithmic_sovereignty_pct": round(base["asy_baseline_pct"] * maturity_multiplier, 0),
        },
        "target_improvement": {
            "dlr_target_pct": "40% reduction",
            "da_target_score": "≥5.2 on 7-pt Delphi Likert",
            "iar_target_pct": "≥91% grid-independent",
            "asy_target_pct": "≥85% SA-local institutional grounding",
        },
    }


async def _bbbee_compliance(
    deployment_type: str,
    affected_roles: list = None,
    training_data_source: str = "",
) -> dict:
    """Assess B-BBEE and Employment Equity compliance risk."""
    affected_roles = affected_roles or []
    high_risk_roles = ["clerk", "admin", "data entry", "teller", "operator", "cashier", "driver", "general worker"]
    proxy_risk_sources = ["historical hr", "performance review", "zip code", "postal", "surname", "school", "address"]

    role_risk = any(r.lower() in " ".join(affected_roles).lower() for r in high_risk_roles)
    proxy_risk = any(p in training_data_source.lower() for p in proxy_risk_sources)

    return {
        "deployment_type": deployment_type,
        "eea_indirect_discrimination_risk": "High" if role_risk else "Medium",
        "proxy_variable_risk": "High" if proxy_risk else "Medium",
        "bbbee_skills_development_impact": "Negative" if role_risk else "Neutral — requires monitoring",
        "affected_demographic": "Entry-level Black workforce disproportionately at risk" if role_risk else "Monitor for disparate impact",
        "mandatory_actions": [
            "Commission independent distributional audit before deployment",
            "Establish AI ethics review committee with B-BBEE representation",
            "Document distributional audit results in AI System Card",
            "File B-BBEE impact assessment with Skills Development pillar manager",
            "Set 12-month post-deployment equity monitoring trigger",
        ],
        "king_iv_obligation": "Principle 12: Board must review distributional audit report",
        "popia_intersection": "Biographic proxy variables may constitute special personal information under POPIA Section 26",
    }


def execute_tool_sync(tool_name: str, tool_args: dict) -> str:
    """Synchronous wrapper for Streamlit compatibility."""
    import asyncio
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(execute_tool(tool_name, tool_args))
    loop.close()
    return json.dumps(result, indent=2)
