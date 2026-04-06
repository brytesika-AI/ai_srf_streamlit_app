# AI-SRF · AI-Driven Strategic Resilience Framework
### By: Bright Sikazwe, PhD Candidate
**University of Johannesburg — College of Business and Economics**  
Department of Information and Knowledge Management  
*"Umuntu ngumuntu ngabantu"*

---

## Overview
A production-grade Streamlit application implementing the full AI-SRF 7-agent governance cycle for South African corporate digitalisation strategy. Built on Groq's inference API with semantic RAG, ChromaDB vector storage, and function-calling tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  STREAMLIT UI  (app.py)                                          │
│  5 tabs: Dialogue · ROR Dashboard · Evidence Base · Peers · Card│
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1 — Environmental Monitor                                 │
│  Groq LLM + get_sa_infrastructure_signal (function calling)     │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2 — Context-Conditioned Reasoning                         │
│  Socratic Partner → Forensic Analyst → Creative Catalyst        │
│  → Devil's Advocate (UNBYPASSABLE)                              │
│  All agents: Groq llama-3.3-70b-versatile + function calling    │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3 — Socio-Technical Alignment                            │
│  Implementation Scaffolding + Monitoring Agent                   │
├─────────────────────────────────────────────────────────────────┤
│  RAG ENGINE  (rag_engine.py)                                     │
│  Groq nomic-embed-text-v1_5 → ChromaDB cosine search            │
│  PDF loader: RAW_DATA_RAG/ folder (LangChain PyPDFDirectoryLoader)│
├─────────────────────────────────────────────────────────────────┤
│  MCP-STYLE TOOLS  (tools.py)                                     │
│  World Bank API · SA infrastructure signals · Data provenance   │
│  ROR baseline estimator · B-BBEE compliance checker             │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add RAG documents (optional but recommended)
Place PDF files into the `RAW_DATA_RAG/` folder. These will be:
- Chunked with `RecursiveCharacterTextSplitter` (1000/200)
- Embedded with Groq `nomic-embed-text-v1_5`
- Stored in ChromaDB persistent store

Recommended PDFs: King IV report, POPIA guidelines, B-BBEE codes, Eskom annual reports, Zondo Commission summary, World Bank SA Digital Economy report.

### 3. Run
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Models Used (Groq)

| Purpose | Model | Notes |
|---|---|---|
| Reasoning (all 7 agents) | `llama-3.3-70b-versatile` | Function calling enabled |
| Embeddings (RAG) | `nomic-embed-text-v1_5` | 768-dim, 8192 token context |

## Tools (Function Calling / ADK-style)

| Tool | Description |
|---|---|
| `get_world_bank_sa_indicator` | World Bank Open Data API for SA macroeconomic indicators |
| `get_sa_infrastructure_signal` | Eskom/Transnet/ZAR/broadband real-time signals |
| `run_data_provenance_audit` | Zondo Commission contamination + POPIA compliance check |
| `estimate_ror_baseline` | ROR baseline by sector + digital maturity |
| `check_bbbee_compliance_risk` | B-BBEE / EEA proxy variable risk assessment |

## ROR Indicators

| Indicator | Measure |
|---|---|
| **DLR** Decision Latency Reduction | Time: disruption → board-validated response |
| **DA** Decision Alpha | Quality delta (7-pt Delphi Likert: viability + compliance + risk) |
| **IAR** Infrastructure Autonomy Ratio | Edge AI uptime % independent of national grid |
| **ASY** Algorithmic Sovereignty Yield | % recommendations grounded in SA-local data |

## Guardrail
All queries are gated against a scope keyword list. Non-SA-corporate-digitalisation queries are blocked at input with an explanatory message.

## Files

```
ai_srf_app/
├── app.py              — Streamlit UI (5 tabs)
├── agents.py           — 7-agent pipeline (ADK-style orchestration)
├── rag_engine.py       — Groq embeddings + ChromaDB RAG
├── tools.py            — MCP-style function-calling tools
├── config.py           — Models, constants, SA corpus, competitor cases
├── requirements.txt    — Python dependencies
├── RAW_DATA_RAG/       — Place PDF files here for RAG indexing
└── chroma_db/          — Auto-created ChromaDB persistent store
```

---
*AI-SRF v33 · Bright Sikazwe · PhD Candidate · University of Johannesburg*
