"""
Production-oriented Web Intelligence URL Discovery UI.

Purpose
- Input: vendor name, optional product name, optional approved domains.
- Uses Azure OpenAI Responses API with web_search enabled.
- Discovers candidate official public URLs for trust/security/privacy/compliance/status/advisory pages.
- Does not treat model-written URLs as trusted evidence.
- Prefers URLs returned from web_search tool source metadata.
- Shows token usage, latency, validation status, and raw response metadata.
- Optional LangSmith run logging if LANGSMITH_API_KEY is configured and langsmith is installed.

Run
  streamlit run web_intel_streamlit_websearch.py

Required .env
  AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
  AZURE_OPENAI_API_KEY=<key>
  AZURE_OPENAI_WEBSEARCH_DEPLOYMENT=gpt-5-4-mini-websearch-test

Optional .env
  AZURE_OPENAI_TIMEOUT_SECONDS=90
  WEBSEARCH_MAX_RESULTS=8
  LANGSMITH_API_KEY=<key>
  LANGSMITH_PROJECT=web-intelligence-rnd
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

load_dotenv(Path(__file__).with_name(".env"))

BLOCKED_DOMAINS = [
    "wikipedia.org",
    "reddit.com",
    "medium.com",
    "quora.com",
    "github.com",
    "stackoverflow.com",
]

DISCOVERY_CATEGORIES = [
    "trust center",
    "security",
    "privacy",
    "compliance",
    "service trust",
    "status",
    "security advisory",
    "vulnerability disclosure",
    "data protection",
    "data residency",
    "subprocessors",
]

SYSTEM_INSTRUCTIONS = """
You are a controlled vendor security URL discovery assistant for a Security Assessment Request workflow.
Your only job is to discover candidate public URLs that may contain vendor/product trust, security, privacy, compliance, status, advisory, or vulnerability disclosure information.
Do not make vendor risk decisions.
Do not infer that a vendor is secure or compliant.
Do not invent URLs.
Return only URLs that are supported by the web search results.
Prefer official vendor-owned domains when available.
If product-specific pages are not found, return the closest official vendor-level trust/security/compliance pages and explain the limitation in the relevance note.
""".strip()


@dataclass(frozen=True)
class CandidateUrl:
    url: str
    title: str
    snippet: str
    source_origin: str
    domain: str
    validation_status: str
    validation_reason: str
    score: int


@dataclass(frozen=True)
class UsageSummary:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    raw_usage: dict[str, Any]


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    raise RuntimeError(f"Missing required environment variable: {name}")


def approximate_token_count(text: str) -> int:
    # Fast local estimate for prompt-size debugging. Azure response.usage is the source of truth.
    # Rule of thumb: English/API prompts average around 4 characters per token.
    return max(1, (len(text or "") + 3) // 4)


def build_prompt_debug(vendor: str, product: str | None, approved_domains: list[str], max_results: int) -> dict[str, Any]:
    user_prompt = build_user_prompt(vendor, product, max_results)
    tool_payload = build_web_search_tool(approved_domains)
    request_payload = {
        "model": os.getenv("AZURE_OPENAI_WEBSEARCH_DEPLOYMENT", ""),
        "instructions": SYSTEM_INSTRUCTIONS,
        "tools": [tool_payload],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
        "input": user_prompt,
    }
    system_chars = len(SYSTEM_INSTRUCTIONS)
    user_chars = len(user_prompt)
    serialized_chars = len(json.dumps(request_payload, ensure_ascii=False))
    return {
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "user_prompt": user_prompt,
        "tool_payload": tool_payload,
        "request_payload": request_payload,
        "prompt_size_estimate": {
            "system_instruction_chars": system_chars,
            "user_prompt_chars": user_chars,
            "serialized_request_chars": serialized_chars,
            "estimated_system_instruction_tokens": approximate_token_count(SYSTEM_INSTRUCTIONS),
            "estimated_user_prompt_tokens": approximate_token_count(user_prompt),
            "estimated_serialized_request_tokens": approximate_token_count(json.dumps(request_payload, ensure_ascii=False)),
            "note": "This is a local estimate only. Azure response.usage is the billing/runtime source of truth after the call completes.",
        },
    }


def parse_csv_domains(value: str) -> list[str]:
    domains: list[str] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        item = item.removeprefix("https://").removeprefix("http://").split("/")[0]
        if item.startswith("www."):
            item = item[4:]
        domains.append(item)
    return sorted(set(domains))


def normalize_domain(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return host[4:] if host.startswith("www.") else host


def is_https_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.netloc)


def is_domain_match(domain: str, allowed_domains: list[str]) -> bool:
    if not allowed_domains:
        return True
    return any(domain == allowed or domain.endswith("." + allowed) for allowed in allowed_domains)


def derive_vendor_terms(vendor: str, product: str | None) -> list[str]:
    terms = [vendor.lower().strip()]
    compact_vendor = re.sub(r"[^a-z0-9]", "", vendor.lower())
    if compact_vendor and compact_vendor not in terms:
        terms.append(compact_vendor)
    if product:
        terms.extend(part for part in re.split(r"\s+", product.lower()) if len(part) >= 4)
    return sorted(set(t for t in terms if t))


def validate_candidate_url(url: str, vendor: str, product: str | None, approved_domains: list[str]) -> tuple[str, str]:
    if not is_https_url(url):
        return "rejected", "URL is missing HTTPS or valid host."

    domain = normalize_domain(url)
    if any(domain == blocked or domain.endswith("." + blocked) for blocked in BLOCKED_DOMAINS):
        return "rejected", "Domain is explicitly blocked."

    if approved_domains and not is_domain_match(domain, approved_domains):
        return "rejected", "Domain is outside the approved domain list."

    vendor_terms = derive_vendor_terms(vendor, product)
    joined = f"{domain} {url}".lower()
    if approved_domains:
        return "accepted", "Domain passed explicit approved-domain validation."

    if any(term in joined for term in vendor_terms):
        return "accepted_with_review", "No approved domains were supplied, but URL/domain appears vendor-related. Analyst/domain-owner review required."

    return "review_required", "No approved domains were supplied and vendor ownership is not deterministic."


def score_candidate(vendor: str, product: str | None, url: str, title: str, snippet: str, source_origin: str, validation_status: str) -> int:
    text = f"{url} {title} {snippet}".lower()
    score = 0

    if source_origin == "tool_source":
        score += 100
    else:
        score += 10

    if validation_status == "accepted":
        score += 80
    elif validation_status == "accepted_with_review":
        score += 40
    elif validation_status == "review_required":
        score += 10
    else:
        score -= 100

    vendor_l = vendor.lower().strip()
    if vendor_l and vendor_l in text:
        score += 25

    if product:
        product_terms = [p for p in re.split(r"\s+", product.lower()) if len(p) >= 4]
        score += sum(10 for term in product_terms if term in text)

    score += sum(8 for term in DISCOVERY_CATEGORIES if term in text)

    noisy_patterns = ["/search", "?q=", "/community", "/answers", "/blog/tag", "/author"]
    if any(pattern in url.lower() for pattern in noisy_patterns):
        score -= 50

    return max(score, 0)


def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text


def safe_json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(strip_json_fence(text))
    except Exception:
        return {"sources": []}


def build_client() -> OpenAI:
    endpoint = required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_key = required_env("AZURE_OPENAI_API_KEY")
    timeout_seconds = float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "90"))
    return OpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=api_key,
        timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        max_retries=0,
    )


def build_user_prompt(vendor: str, product: str | None, max_results: int) -> str:
    product_line = product if product else "not provided"
    categories = ", ".join(DISCOVERY_CATEGORIES)
    return f"""
Vendor: {vendor}
Product: {product_line}

Discover candidate public URLs for security assessment research.
Target URL categories: {categories}.

Return JSON only with this exact schema:
{{
  "sources": [
    {{
      "url": "https://example.com/path",
      "title": "string",
      "relevance_note": "why this URL is relevant to the vendor/product security assessment"
    }}
  ]
}}

Rules:
- Return at most {max_results} sources.
- Prefer official vendor-owned pages.
- Do not include forums, social media, Wikipedia, Reddit, GitHub issues, or third-party blogs.
- Do not summarize the vendor.
- Do not make a risk decision.
- Do not claim certification validity, breach absence, or compliance status.
- If product-specific trust pages are unavailable, return closest official vendor-level pages and state that limitation in relevance_note.
""".strip()


def get_usage_summary(response: Any) -> UsageSummary:
    usage = getattr(response, "usage", None)
    raw: dict[str, Any] = {}
    if usage is None:
        return UsageSummary(None, None, None, raw)

    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {k: getattr(usage, k) for k in dir(usage) if not k.startswith("_") and isinstance(getattr(usage, k), (int, str, float, dict, type(None)))}

    input_tokens = raw.get("input_tokens") or raw.get("prompt_tokens")
    output_tokens = raw.get("output_tokens") or raw.get("completion_tokens")
    total_tokens = raw.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)

    return UsageSummary(
        int(input_tokens) if input_tokens is not None else None,
        int(output_tokens) if output_tokens is not None else None,
        int(total_tokens) if total_tokens is not None else None,
        raw,
    )


def extract_tool_sources(response: Any) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for output_item in getattr(response, "output", []) or []:
        if getattr(output_item, "type", None) != "web_search_call":
            continue
        action = getattr(output_item, "action", None)
        if not action:
            continue
        for source in getattr(action, "sources", []) or []:
            url = (getattr(source, "url", None) or "").strip()
            if not url:
                continue
            sources.append(
                {
                    "url": url,
                    "title": (getattr(source, "title", "") or "").strip(),
                    "snippet": (getattr(source, "snippet", "") or "").strip(),
                    "source_origin": "tool_source",
                }
            )
    return sources


def extract_model_json_sources(response: Any) -> list[dict[str, str]]:
    parsed = safe_json_loads(getattr(response, "output_text", ""))
    sources: list[dict[str, str]] = []
    for item in parsed.get("sources", []) or []:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        sources.append(
            {
                "url": url,
                "title": str(item.get("title", "")).strip(),
                "snippet": str(item.get("relevance_note", "")).strip(),
                "source_origin": "model_json",
            }
        )
    return sources


def build_web_search_tool(approved_domains: list[str]) -> dict[str, Any]:
    filters: dict[str, Any] = {"blocked_domains": BLOCKED_DOMAINS}
    if approved_domains:
        filters["allowed_domains"] = approved_domains
    return {"type": "web_search", "filters": filters}


def log_langsmith_run(payload: dict[str, Any]) -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        return
    try:
        from langsmith import Client  # type: ignore

        client = Client()
        project_name = os.getenv("LANGSMITH_PROJECT", "web-intelligence-rnd")
        run = client.create_run(
            name="websearch_url_discovery",
            run_type="chain",
            project_name=project_name,
            inputs=payload.get("inputs", {}),
            outputs=payload.get("outputs", {}),
            extra=payload.get("extra", {}),
        )
        client.update_run(run.id, end_time=datetime.now(timezone.utc))
    except Exception:
        # LangSmith must never break the local R&D path.
        return


def discover_urls(vendor: str, product: str | None, approved_domains: list[str], max_results: int) -> dict[str, Any]:
    deployment = required_env("AZURE_OPENAI_WEBSEARCH_DEPLOYMENT")
    client = build_client()
    prompt = build_user_prompt(vendor, product, max_results)
    prompt_debug = build_prompt_debug(vendor, product, approved_domains, max_results)
    start = time.perf_counter()

    response = client.responses.create(
        model=deployment,
        instructions=SYSTEM_INSTRUCTIONS,
        tools=[build_web_search_tool(approved_domains)],
        tool_choice="auto",
        include=["web_search_call.action.sources"],
        input=prompt,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = get_usage_summary(response)

    raw_sources = extract_tool_sources(response) + extract_model_json_sources(response)
    candidates: list[CandidateUrl] = []
    seen: set[str] = set()

    for item in raw_sources:
        url = re.sub(r"[\s)]+$", "", item["url"].strip())
        if not url or url in seen:
            continue
        seen.add(url)
        domain = normalize_domain(url) if is_https_url(url) else ""
        status, reason = validate_candidate_url(url, vendor, product, approved_domains)
        score = score_candidate(vendor, product, url, item["title"], item["snippet"], item["source_origin"], status)
        candidates.append(
            CandidateUrl(
                url=url,
                title=item["title"],
                snippet=item["snippet"],
                source_origin=item["source_origin"],
                domain=domain,
                validation_status=status,
                validation_reason=reason,
                score=score,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    accepted = [c for c in candidates if c.validation_status in {"accepted", "accepted_with_review"}]

    result = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "deployment": deployment,
            "latency_ms": latency_ms,
            "max_results": max_results,
        },
        "inputs": {
            "vendor": vendor,
            "product": product,
            "approved_domains": approved_domains,
            "blocked_domains": BLOCKED_DOMAINS,
        },
        "usage": asdict(usage),
        "prompt_debug": prompt_debug,
        "policy": {
            "accepted_source_rule": "Prefer URLs returned by web_search tool source metadata. Model JSON URLs are never evidence and must pass local validation.",
            "production_evidence_rule": "A URL becomes evidence only after local validation and successful page scraping.",
        },
        "candidate_count": len(candidates),
        "accepted_candidate_count": len(accepted),
        "best_candidate": asdict(accepted[0]) if accepted else None,
        "candidates": [asdict(c) for c in candidates[:max_results]],
        "raw_model_output_text": getattr(response, "output_text", ""),
    }

    log_langsmith_run({"inputs": result["inputs"], "outputs": {"candidates": result["candidates"]}, "extra": {"usage": result["usage"], "run": result["run"]}})
    return result


def render_app() -> None:
    st.set_page_config(page_title="Web Intelligence URL Discovery", layout="wide")
    st.title("Web Intelligence URL Discovery")
    st.caption("Azure OpenAI web_search test for vendor/product trust and security URL discovery")

    with st.sidebar:
        st.header("Configuration")
        st.text_input("Azure deployment", value=os.getenv("AZURE_OPENAI_WEBSEARCH_DEPLOYMENT", ""), disabled=True)
        max_results = st.slider("Max results", min_value=1, max_value=20, value=int(os.getenv("WEBSEARCH_MAX_RESULTS", "8")))
        st.markdown("**Validation mode**")
        st.write("Supplying approved domains makes validation deterministic. Leaving it blank allows discovery but marks vendor ownership as review-required unless domain appears vendor-related.")

    col1, col2 = st.columns(2)
    with col1:
        vendor = st.text_input("Vendor name", value="Microsoft")
    with col2:
        product = st.text_input("Product name", value="Microsoft 365 Copilot")

    approved_domain_text = st.text_area(
        "Approved domains, optional comma-separated",
        value="",
        placeholder="Example: microsoft.com, learn.microsoft.com, servicetrust.microsoft.com",
        height=80,
    )
    approved_domains = parse_csv_domains(approved_domain_text)

    prompt_preview = build_prompt_debug(vendor.strip() or "<vendor>", product.strip() or None, approved_domains, max_results)

    with st.expander("Exact prompt and request payload that will be sent", expanded=False):
        st.markdown("**System instructions**")
        st.code(prompt_preview["system_instructions"], language="text")
        st.markdown("**User prompt**")
        st.code(prompt_preview["user_prompt"], language="text")
        st.markdown("**web_search tool payload**")
        st.json(prompt_preview["tool_payload"])
        st.markdown("**Prompt size estimate before API call**")
        st.json(prompt_preview["prompt_size_estimate"])

    run_clicked = st.button("Run web search discovery", type="primary")

    if not run_clicked:
        return

    if not vendor.strip():
        st.error("Vendor name is required.")
        return

    try:
        with st.spinner("Calling Azure OpenAI web_search..."):
            result = discover_urls(vendor.strip(), product.strip() or None, approved_domains, max_results)
    except BadRequestError as exc:
        st.error("BadRequestError. The deployment may not support Responses API web_search, or web_search is not enabled.")
        st.code(str(exc))
        return
    except RateLimitError as exc:
        st.error("RateLimitError. Deployment quota is currently unavailable.")
        st.code(str(exc))
        return
    except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
        st.error(f"Azure OpenAI call failed: {type(exc).__name__}")
        st.code(str(exc))
        return
    except Exception as exc:
        st.error(f"Unexpected error: {type(exc).__name__}")
        st.code(str(exc))
        return

    usage = result["usage"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Input tokens", usage.get("input_tokens") or "n/a")
    m2.metric("Output tokens", usage.get("output_tokens") or "n/a")
    m3.metric("Total tokens", usage.get("total_tokens") or "n/a")
    m4.metric("Latency ms", result["run"]["latency_ms"])

    with st.expander("Prompt debug after API call", expanded=True):
        st.markdown("**Exact system instructions sent**")
        st.code(result["prompt_debug"]["system_instructions"], language="text")
        st.markdown("**Exact user prompt sent**")
        st.code(result["prompt_debug"]["user_prompt"], language="text")
        st.markdown("**Exact request payload shape, key redacted automatically because it is not part of payload**")
        st.json(result["prompt_debug"]["request_payload"])
        st.markdown("**Local estimate vs Azure usage**")
        st.json({"local_estimate": result["prompt_debug"]["prompt_size_estimate"], "azure_usage": result["usage"]})

    st.subheader("Best candidate")
    if result["best_candidate"]:
        st.json(result["best_candidate"])
    else:
        st.warning("No accepted candidates. Review returned URLs or provide approved domains.")

    st.subheader("Candidates")
    if result["candidates"]:
        st.dataframe(result["candidates"], use_container_width=True)
    else:
        st.info("No candidate URLs returned.")

    with st.expander("Raw JSON result"):
        st.json(result)

    with st.expander("Raw model output text"):
        st.code(result.get("raw_model_output_text") or "")

    st.download_button(
        "Download result JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="websearch_url_discovery_result.json",
        mime="application/json",
    )


if __name__ == "__main__":
    render_app()
