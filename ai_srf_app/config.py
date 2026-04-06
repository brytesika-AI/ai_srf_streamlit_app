"""
AI-SRF Configuration
By: Bright Sikazwe, PhD Candidate
University of Johannesburg — College of Business and Economics
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

# ─── Model config (Groq) ────────────────────────────────────────────────────
GROQ_LLM_MODEL       = "llama-3.3-70b-versatile"   # primary reasoning
GROQ_FAST_MODEL      = "llama3-8b-8192"             # fast routing/guardrail
GROQ_EMBED_MODEL     = "nomic-embed-text-v1_5"      # RAG embeddings
GROQ_EMBED_BASE_URL  = OPENAI_BASE_URL
GROQ_DEEPSEEK_MODEL  = "deepseek-r1-distill-qwen-32b"

# ─── Vector DB ──────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR   = "./chroma_db"
CHROMA_COLLECTION    = "ai_srf_sa_corpus"
RAG_CHUNK_SIZE       = 1000
RAG_CHUNK_OVERLAP    = 200
RAG_TOP_K            = 5

# ─── RAG source folder ──────────────────────────────────────────────────────
RAG_DATA_DIR         = "./RAW_DATA_RAG"

# ─── Risk state colours ─────────────────────────────────────────────────────
RISK_COLORS = {
    "Nominal":  {"icon": "🟢", "hex": "#16A34A", "bg": "#F0FDF4"},
    "Elevated": {"icon": "🟡", "hex": "#CA8A04", "bg": "#FEFCE8"},
    "Compound": {"icon": "🟠", "hex": "#EA580C", "bg": "#FFF7ED"},
    "Critical": {"icon": "🔴", "hex": "#DC2626", "bg": "#FEF2F2"},
}

LAYER_COLORS = {1: "#C2410C", 2: "#1E3A5F", 3: "#166534"}

# ─── Built-in SA corpus (fallback when RAW_DATA_RAG is empty) ───────────────
SA_CORPUS = [
    {
        "id": "eskom-001",
        "title": "Eskom Load Shedding & Digital Infrastructure",
        "content": (
            "Stage 1–8 load shedding costs the SA economy R4–5bn per stage per day. "
            "Corporate data centres require 4–6hr UPS plus diesel backup at R21/litre avg (2024). "
            "Cloud inference latency spikes 40–70% at Stage 4+. EskomSePush API provides real-time "
            "schedule data. Private electricity generation absorbs 3–8% capex in energy-intensive "
            "sectors. Stage 6 events average 12–14hr outage windows. AI systems designed without "
            "energy-independence assumptions carry compounding infrastructure risk in the SA context."
        ),
        "source": "Eskom SOC / PwC SA 2024",
        "keywords": ["eskom","power","load shedding","energy","grid","diesel","stage","electricity","uptime","infrastructure"],
    },
    {
        "id": "transnet-001",
        "title": "Transnet Port & Freight Rail Deterioration",
        "content": (
            "Durban container throughput fell 35% between 2019 and 2023. Average vessel wait exceeds "
            "8 days vs global average of 1.2 days. Freight rail locomotive availability stands at 47% "
            "vs a 70% target. Mining and agricultural AI supply chain tools are forced to operate on "
            "structurally corrupted logistics data. Port backlogs cascade into cold-chain failures "
            "nationwide. AI-powered supply chain algorithms require unbroken value-chain visibility — "
            "a precondition that corrupted Transnet data directly undermines."
        ),
        "source": "Havenga et al. 2023 / World Bank LPI 2023",
        "keywords": ["transnet","port","logistics","freight","rail","supply chain","shipping","containers","Durban"],
    },
    {
        "id": "popia-001",
        "title": "POPIA Data Residency & AI Governance",
        "content": (
            "POPIA (Act 4 of 2013) requires South African personal data to remain resident in SA. "
            "Cross-border AI processing requires a Transfer Impact Assessment (TIA). Cloud migrations "
            "offshore carry POPIA Section 72 violation risk. Fines reach up to R10m or 10 years "
            "imprisonment. Biometric, health, and financial data carry heightened obligations. "
            "The Information Regulator holds active enforcement powers and has issued enforcement "
            "notices to corporations deploying AI without adequate data governance documentation."
        ),
        "source": "Information Regulator SA 2023 / POPIA Act 4 of 2013",
        "keywords": ["POPIA","data","privacy","residency","cloud","personal information","compliance","cross-border","TIA"],
    },
    {
        "id": "king4-001",
        "title": "King IV Corporate Governance & AI Accountability",
        "content": (
            "King IV Principle 12 mandates that boards exercise non-delegable oversight of technology "
            "risk. JSE Listings Requirements align with King IV on technology risk disclosure. Boards "
            "must understand AI inputs, outputs, and failure modes. Devil's Advocate mechanisms align "
            "with Principle 15 on structured assurance. Integrated Reporting must include material "
            "AI governance risks. An AI System Card constitutes a formal accountability artifact "
            "satisfying King IV Principle 12 transparency requirements."
        ),
        "source": "IoDSA King IV 2016 / JSE Listings Requirements 2024",
        "keywords": ["King IV","governance","board","accountability","JSE","compliance","oversight","technology","directors","assurance"],
    },
    {
        "id": "bbbee-001",
        "title": "B-BBEE, Employment Equity & AI Workforce Impact",
        "content": (
            "AI automation disproportionately displaces entry-level Black workforce (40–60% at risk "
            "per sector analysis). The Employment Equity Act prohibits indirect discrimination via "
            "proxy variables. AI systems must undergo distributional audit before deployment. The "
            "Human Capital pillar of B-BBEE Skills Development must map against AI-generated "
            "capability gaps. Firms below B-BBEE Level 3 face procurement penalties. The AI-SRF "
            "navigates the irreconcilable tension between AI efficiency agenda and B-BBEE objectives."
        ),
        "source": "DTI B-BBEE Codes 2015 / PC4IR 2020",
        "keywords": ["B-BBEE","BEE","employment equity","transformation","EEA","skills","workforce","automation","empowerment","proxy"],
    },
    {
        "id": "informal-001",
        "title": "SA Informal Economy & AI Blind Spots",
        "content": (
            "60–70% of South African economic activity is informal (StatsSA 2024). Spaza shops: "
            "150,000+ outlets, predominantly cash-based. Stokvel savings: R50bn+ annually — entirely "
            "invisible to formal AI credit scoring. Rural broadband penetration: 23% vs urban 67%. "
            "USSD banking serves 30M+ unbanked citizens. AI models trained exclusively on formal "
            "banking data systematically exclude the informal economy. This represents simultaneously "
            "a governance failure and a commercial opportunity for firms that can bridge the gap."
        ),
        "source": "StatsSA 2024 / World Bank SA Financial Inclusion 2023",
        "keywords": ["informal economy","spaza","stokvel","rural","township","unbanked","USSD","digital divide","financial inclusion"],
    },
    {
        "id": "zar-001",
        "title": "ZAR Volatility & Technology Cost Exposure",
        "content": (
            "ZAR/USD 10-year range: R10.50–R19.70. SaaS licensing in USD exposes SA firms to 40–85% "
            "cost escalation on 5-year contracts. AWS, Azure, and GCP are USD-denominated. Average "
            "ZAR depreciation: 4.8% per year over 10 years. Dollar-denominated AI vendor lock-in "
            "creates compounding fiscal vulnerability. Currency hedging on IT contracts is limited "
            "for mid-cap JSE firms. Business cases modelled at current parity are routinely "
            "invalidated within 18 months by rand movements."
        ),
        "source": "SARB 2024 / PwC SA Digital Outlook 2024",
        "keywords": ["ZAR","currency","rand","forex","USD","exchange rate","cloud cost","SaaS","vendor","pricing","depreciation"],
    },
    {
        "id": "skills-001",
        "title": "SA AI Human Capital & Skills Gap",
        "content": (
            "SA produces approximately 800 ML/AI specialists per year against a demand of 15,000+ "
            "(BCG 2023). Senior ML-Ops engineers carry 6–9 month vacancy periods. The junior "
            "developer pipeline is being disrupted by GenAI commoditisation of entry-level coding "
            "roles. Brain drain: 35% of STEM graduates emigrate within 5 years. AI specialist "
            "salary premium: 2.8–4.2x market rate. AI-augmented scaffolding bridges the immediate "
            "gap as an explicit temporary mechanism — not a permanent substitute for human expertise."
        ),
        "source": "BCG 2023 / HSRC Skills Audit 2024",
        "keywords": ["skills","talent","human capital","ML","AI","engineers","capacity","training","brain drain","mentorship","upskilling"],
    },
    {
        "id": "statsza-001",
        "title": "StatsSA Digital Economy & Structural Inequality",
        "content": (
            "South Africa holds the world's highest Gini coefficient at 0.67. Formal unemployment: "
            "32.1% (Q3 2023). Youth unemployment (15–34): 44.6%. Digital economy constitutes 8.5% "
            "of GDP. Fixed broadband household penetration: 14.1%. Mobile internet: 68.4% of adults. "
            "ICT sector employs 284,000 workers formally. Digital skills gap affects 78% of SA "
            "firms. GDP per capita: USD 6,193 (World Bank 2023). Structural inequality is a "
            "primary non-negotiable design input for any SA corporate AI framework."
        ),
        "source": "StatsSA 2024 / World Bank SA 2023",
        "keywords": ["StatsSA","inequality","Gini","unemployment","GDP","digital economy","broadband","poverty","South Africa","structural"],
    },
    {
        "id": "capture-001",
        "title": "State Capture, Data Integrity & AI Governance",
        "content": (
            "The Zondo Commission (2022) documented systematic corruption across 13 SOEs that "
            "materially affected data integrity across the public sector. AI trained on "
            "state-capture-era government data reproduces historical distortions as outputs. "
            "Data provenance auditing is a Tier 1 governance requirement for any SA AI deployment "
            "that incorporates public sector data. National Treasury procurement data: 30–40% "
            "affected by irregular expenditure 2012–2020. POPIA enforcement now includes AI "
            "data pipeline audits as part of the Information Regulator's compliance framework."
        ),
        "source": "Zondo Commission 2022 / Information Regulator 2023",
        "keywords": ["state capture","Zondo","corruption","data integrity","provenance","public sector","SOE","governance failure"],
    },
]

# ─── Competitor cases ────────────────────────────────────────────────────────
COMPETITOR_CASES = [
    {
        "id": "mpesa",
        "market": "Kenya · East Africa",
        "company": "Safaricom M-Pesa",
        "challenge": "AI-driven financial services in infrastructure-constrained market with 70%+ informal economy",
        "approach": (
            "USSD-first AI architecture bypassing smartphone dependency entirely. "
            "Edge-processing enables offline transaction validation. "
            "Integrated informal savings (chama groups) into credit scoring via alternative data. "
            "Zero dependency on hyperscaler cloud for core transaction AI."
        ),
        "outcome": "47M users. 50% of Kenya GDP flows through platform. AI credit scoring reaches 84% previously unbanked.",
        "sa_relevance": "USSD-first design mirrors SA informal economy requirements. Alternative-data credit scoring addresses stokvel/spaza blind spots directly.",
        "ror": {"dlr": "+62%", "da": "+44%", "iar": "89%", "asy": "+71%"},
    },
    {
        "id": "interswitch",
        "market": "Nigeria · West Africa",
        "company": "Interswitch Group",
        "challenge": "AI payment infrastructure across unreliable grid with fragmented regulation and USD-denominated vendor costs",
        "approach": (
            "Distributed edge-node architecture for grid-independent transaction processing. "
            "Regulatory sandbox engagement with CBN. "
            "Refused hyperscaler lock-in on core AI — maintained sovereign training data control."
        ),
        "outcome": "18M daily transactions at 99.2% uptime despite 8+ hr daily outages. Listed at USD 1bn+.",
        "sa_relevance": "Vendor sovereignty strategy directly mirrors AI-SRF Algorithmic Sovereignty Yield. Regulatory sandbox model applicable to FSCA engagement in SA.",
        "ror": {"dlr": "+51%", "da": "+38%", "iar": "94%", "asy": "+58%"},
    },
    {
        "id": "irembo",
        "market": "Rwanda · Central Africa",
        "company": "Irembo / Rwanda Digital",
        "challenge": "Government-led AI digitalisation with severe talent shortage and 85% mobile-only citizen base",
        "approach": (
            "AI-augmented scaffolding for civil service — junior staff capacity extended by AI copilots. "
            "Upskilling SLAs embedded as hard conditions in ALL vendor contracts. "
            "Mobile-first by design with zero senior ML-Ops dependency at launch."
        ),
        "outcome": "100% government service digitisation. AI-augmented workforce 3x more productive. Digital literacy rose from 18% to 62% in 6 years.",
        "sa_relevance": "AI-augmented scaffolding maps directly to AI-SRF Tier 2 Implementation. SLA-embedded upskilling mirrors AI-SRF Tier 3 Capability Development.",
        "ror": {"dlr": "+73%", "da": "+55%", "iar": "81%", "asy": "+84%"},
    },
]

# ─── Silicon Sampling disruption scenarios ───────────────────────────────────
PRESET_SCENARIOS = [
    {
        "label": "☁️ CRM Cloud Migration · Banking",
        "query": (
            "Our JSE-listed bank is proposing to migrate its core CRM system to AWS. "
            "The business case assumes 30% cost savings but was modelled before recent ZAR depreciation. "
            "We need a full AI-SRF cycle covering POPIA data residency, ZAR/USD currency exposure, "
            "load shedding infrastructure risk, and B-BBEE compliance."
        ),
    },
    {
        "label": "⚡ Edge AI Deployment · Mining",
        "query": (
            "We operate platinum mines in a rural environment with average broadband latency of 400ms "
            "and Stage 4 load shedding. Our vendor proposes a cloud-based predictive maintenance AI "
            "with a 99.9% uptime SLA. Run the full strategic resilience cycle including vendor "
            "dependency analysis and ML-Ops skills gap assessment."
        ),
    },
    {
        "label": "⚖️ AI Hiring Tool · Employment Equity",
        "query": (
            "We are deploying an AI-based CV screening tool for a JSE-listed financial services firm. "
            "Our Chief People Officer has flagged Employment Equity Act and B-BBEE compliance concerns. "
            "Conduct a full distributional audit covering proxy variables, algorithmic injustice risk, "
            "and King IV governance obligations."
        ),
    },
    {
        "label": "📦 Supply Chain AI · Retail",
        "query": (
            "We are a JSE-listed retailer evaluating an AI-powered demand forecasting system. "
            "Transnet port delays are at record highs and our current logistics data is structurally "
            "corrupted. Run the AI-SRF cycle to build a resilient architecture that accounts for "
            "SOE infrastructure failure and informal economy consumer blind spots."
        ),
    },
]

# ─── Scope guardrail keywords ────────────────────────────────────────────────
SCOPE_KEYWORDS = [
    "digital", "ai ", "ai,", "artificial intelligence", "technolog", "strateg",
    "corporate", "jse", "south africa", " sa ", "eskom", "transnet", "popia",
    "b-bbee", "bee", "transform", "cloud", "data", "automati", "resilience",
    "govern", "board", "executi", "business", "enterprise", "system",
    "infrastructure", "deploy", "fintech", "banking", "mining", "retail",
    "telecom", "health", "manufactur", "supply chain", "innovati", "cyber",
    "platform", "api", "saas", "ml", "machine learn", "algorithm", "analytic",
    "startup", "venture", "digitalis", "ror", "return on resilience",
    "king iv", "employment equity", "zar", "rand", "ict", "it strategy",
]
