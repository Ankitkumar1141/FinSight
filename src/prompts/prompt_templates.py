from typing import Any, Dict, List, Tuple

SYSTEM_PROMPT = """You are a Financial Research Copilot — an expert analyst specialising in SEC filings, \
earnings calls, annual reports, investor presentations, and financial news.

Your responsibilities:
1. Provide accurate, data-driven analysis grounded strictly in the provided source documents.
2. Always cite sources using the format [Source N: <filename>, Page <X>].
3. Clearly separate document-backed facts from general financial knowledge.
4. Highlight key risks, opportunities, and trends with quantitative evidence where available.
5. If the documents lack sufficient information to answer, say so explicitly — do not speculate.

Formatting guidelines:
- Use headers (##) to organise long answers.
- Use bullet points for lists of risks, metrics, or highlights.
- Present financial data in tables where applicable."""


def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        year = meta.get("year", "")
        doc_type = meta.get("doc_type", "document")
        company = meta.get("company", "")

        header = f"[Source {i}: {source}"
        if company and company != "unknown":
            header += f" | {company}"
        if year and year != "unknown":
            header += f" | {year}"
        header += f" | Page {page} | {doc_type}]"

        parts.append(f"{header}\n{chunk['content']}")

    return "\n\n---\n\n".join(parts)


# ------------------------------------------------------------------ #
# Specialised prompt templates                                         #
# ------------------------------------------------------------------ #

_RISK_TEMPLATE = """\
CONTEXT FROM FINANCIAL DOCUMENTS:
{context}

## Task
Analyse the risks disclosed in the documents above and answer the question below.

Structure your response as:
1. **Business Risks** — competitive dynamics, customer concentration, market conditions
2. **Financial Risks** — liquidity, leverage, revenue volatility, foreign exchange
3. **Operational Risks** — supply chain, key personnel, technology, cybersecurity
4. **Regulatory / Legal Risks** — compliance, litigation, regulatory changes
5. **Macro Risks** — economic cycles, interest rates, geopolitical exposure

For each risk identified: describe it, rate severity (High / Medium / Low), and quote the source with citation.

Question: {query}"""


_REVENUE_TEMPLATE = """\
CONTEXT FROM FINANCIAL DOCUMENTS:
{context}

## Task
Extract and compare revenue data to answer the question below.

Structure your response as:
1. **Revenue Summary** — total revenue by reporting period with YoY / QoQ growth rates
2. **Segment / Geographic Breakdown** — if disclosed
3. **Drivers & Headwinds** — management's explanation for changes
4. **Forward Guidance** — any revenue outlook provided

Present financial figures in a table. Cite every figure precisely.

Question: {query}"""


_MGMT_TEMPLATE = """\
CONTEXT FROM FINANCIAL DOCUMENTS:
{context}

## Task
Extract and analyse management commentary to answer the question below.

Structure your response as:
1. **Strategic Priorities** — key initiatives and investments
2. **Performance Highlights** — what management emphasised
3. **Outlook & Guidance** — forward-looking statements
4. **Specific Topic** — focused commentary on what the question asks about
5. **Tone Analysis** — any notable shift vs. prior periods (if multiple docs available)

Include direct verbatim quotes with precise citations.

Question: {query}"""


_GENERAL_TEMPLATE = """\
CONTEXT FROM FINANCIAL DOCUMENTS:
{context}

Based on the financial documents above, answer the following question comprehensively. \
Cite every factual claim using [Source N: filename, Page X].

Question: {query}"""


def get_specialized_prompt(
    query: str, context_chunks: List[Dict[str, Any]]
) -> Tuple[str, str]:
    context = _build_context_block(context_chunks)
    q = query.lower()

    if any(w in q for w in ["risk", "risks", "threat", "concern", "uncertainty", "exposure"]):
        user_prompt = _RISK_TEMPLATE.format(context=context, query=query)
    elif any(w in q for w in ["revenue", "sales", "growth", "compare", "comparison", "income", "profit", "margin"]):
        user_prompt = _REVENUE_TEMPLATE.format(context=context, query=query)
    elif any(w in q for w in ["management", "ceo", "said", "mentioned", "stated", "outlook", "guidance", "strategy", "ai ", "invest"]):
        user_prompt = _MGMT_TEMPLATE.format(context=context, query=query)
    else:
        user_prompt = _GENERAL_TEMPLATE.format(context=context, query=query)

    return SYSTEM_PROMPT, user_prompt
