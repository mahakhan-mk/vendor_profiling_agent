"""
Test Azure OpenAI Responses API web_search for vendor/product trust-center URL discovery.

Purpose:
- Input: vendor name and optional product name.
- Uses Azure OpenAI deployment with web_search enabled.
- Returns candidate trust/security/privacy/compliance/status URLs.
- Prefers URLs from web_search tool source metadata when available.
- Locally validates domains so model-written hallucinated URLs are rejected.

Run:
  python websearch_trust_center_test.py --vendor Microsoft --product "Microsoft 365 Copilot"

Required .env or shell variables:
  AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
  AZURE_OPENAI_API_KEY=<key>
  AZURE_OPENAI_WEBSEARCH_DEPLOYMENT=gpt-5-4-mini-websearch-test
  AZURE_OPENAI_TIMEOUT_SECONDS=60
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
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


DEFAULT_ALLOWED_DOMAINS = [
    "microsoft.com",
    "www.microsoft.com",
    "learn.microsoft.com",
    "trustcenter.microsoft.com",
    "servicetrust.microsoft.com",
    "privacy.microsoft.com",
    "status.cloud.microsoft",
    "status.office.com",
    "admin.microsoft.com",
    "msrc.microsoft.com",
]

BLOCKED_DOMAINS = [
    "wikipedia.org",
    "reddit.com",
    "medium.com",
    "quora.com",
    "github.com",
]

TRUST_TERMS = [
    "trust",
    "security",
    "privacy",
    "compliance",
    "service trust",
    "status",
    "advisory",
    "msrc",
    "data protection",
    "copilot",
    "microsoft 365",
]


@dataclass(frozen=True)
class SearchCandidate:
    url: str
    title: str
    snippet: str
    source_origin: str
    score: int


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    raise RuntimeError(f"Missing required environment variable: {name}")


def split_env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def normalize_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def is_allowed_url(url: str, allowed_domains: list[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False
    domain = normalize_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in allowed_domains)


def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text


def parse_model_json(text: str) -> dict[str, Any]:
    return json.loads(strip_json_fence(text))


def score_candidate(vendor: str, product: str | None, url: str, title: str, snippet: str, source_origin: str) -> int:
    text = f"{url} {title} {snippet}".lower()
    vendor_l = vendor.lower()
    product_l = (product or "").lower()

    score = 0
    if source_origin == "tool_source":
        score += 100
    else:
        score += 20

    if vendor_l and vendor_l in text:
        score += 30
    if product_l and product_l in text:
        score += 45

    for term in TRUST_TERMS:
        if term in text:
            score += 10

    url_l = url.lower()
    preferred_patterns = [
        "trustcenter.microsoft.com",
        "servicetrust.microsoft.com",
        "learn.microsoft.com",
        "privacy.microsoft.com",
        "msrc.microsoft.com",
        "status.cloud.microsoft",
    ]
    for pattern in preferred_patterns:
        if pattern in url_l:
            score += 25

    noisy_patterns = ["/search", "/answers/", "/community", "/blog/", "?q="]
    if any(pattern in url_l for pattern in noisy_patterns):
        score -= 40

    return max(score, 0)


def build_client() -> OpenAI:
    endpoint = required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_key = required_env("AZURE_OPENAI_API_KEY")
    timeout_seconds = float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "60"))

    return OpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=api_key,
        timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        max_retries=0,
    )


def build_prompt(vendor: str, product: str | None, max_results: int) -> str:
    product_line = product if product else "not provided"
    return f"""
Find official public Microsoft trust, security, privacy, compliance, service trust, status, or security documentation URLs for the given vendor/product.

Vendor: {vendor}
Product: {product_line}

Goal:
Return candidate URLs that would help a security analyst assess the vendor/product trust posture.
For Microsoft 365 Copilot, prefer official Microsoft pages about Microsoft 365 Copilot security, privacy, compliance, data protection, service trust, responsible AI, or Microsoft 365 trust/compliance documentation.

Return JSON only with this exact schema:
{{
  "sources": [
    {{
      "url": "https://example.com/path",
      "title": "string",
      "relevance_note": "short reason this URL is relevant"
    }}
  ]
}}

Rules:
- Return at most {max_results} sources.
- Use only official Microsoft-owned domains.
- Do not include third-party sites.
- Do not summarize the vendor.
- Do not make a risk decision.
- Do not invent URLs.
- If no product-specific trust page is found, return the closest official Microsoft 365 / Microsoft Trust Center pages and say that in relevance_note.
""".strip()


def extract_tool_sources(response: Any, allowed_domains: list[str]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []

    for output_item in getattr(response, "output", []) or []:
        if getattr(output_item, "type", None) != "web_search_call":
            continue

        action = getattr(output_item, "action", None)
        if not action:
            continue

        for source in getattr(action, "sources", []) or []:
            url = getattr(source, "url", None)
            if not url or not is_allowed_url(url, allowed_domains):
                continue

            sources.append(
                {
                    "url": url.strip(),
                    "title": (getattr(source, "title", "") or "").strip(),
                    "snippet": (getattr(source, "snippet", "") or "").strip(),
                    "source_origin": "tool_source",
                }
            )

    return sources


def extract_model_json_sources(response: Any, allowed_domains: list[str]) -> list[dict[str, str]]:
    try:
        parsed = parse_model_json(getattr(response, "output_text", ""))
    except (json.JSONDecodeError, TypeError):
        return []

    sources: list[dict[str, str]] = []
    for item in parsed.get("sources", []) or []:
        url = str(item.get("url", "")).strip()
        if not url or not is_allowed_url(url, allowed_domains):
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


def discover_trust_urls(vendor: str, product: str | None, max_results: int) -> dict[str, Any]:
    deployment = required_env("AZURE_OPENAI_WEBSEARCH_DEPLOYMENT")
    allowed_domains = split_env_list("APPROVED_DOMAINS", DEFAULT_ALLOWED_DOMAINS)
    client = build_client()

    response = client.responses.create(
        model=deployment,
        tools=[
            {
                "type": "web_search",
                "filters": {
                    "allowed_domains": allowed_domains,
                    "blocked_domains": BLOCKED_DOMAINS,
                },
            }
        ],
        tool_choice="auto",
        include=["web_search_call.action.sources"],
        input=build_prompt(vendor, product, max_results),
    )

    raw_candidates = extract_tool_sources(response, allowed_domains) + extract_model_json_sources(response, allowed_domains)

    candidates: list[SearchCandidate] = []
    seen_urls = set()
    for item in raw_candidates:
        url = re.sub(r"[\s)]+$", "", item["url"])
        if url in seen_urls:
            continue
        seen_urls.add(url)

        score = score_candidate(
            vendor=vendor,
            product=product,
            url=url,
            title=item["title"],
            snippet=item["snippet"],
            source_origin=item["source_origin"],
        )
        candidates.append(
            SearchCandidate(
                url=url,
                title=item["title"],
                snippet=item["snippet"],
                source_origin=item["source_origin"],
                score=score,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)

    return {
        "vendor": vendor,
        "product": product,
        "deployment": deployment,
        "accepted_rule": "Prefer web_search tool sources. Model JSON URLs are included only if they pass local domain validation.",
        "allowed_domains": allowed_domains,
        "candidate_count": len(candidates),
        "candidates": [candidate.__dict__ for candidate in candidates[:max_results]],
        "best_candidate": candidates[0].__dict__ if candidates else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Azure OpenAI web_search for vendor/product trust-center URL discovery.")
    parser.add_argument("--vendor", default="Microsoft", help="Vendor name, e.g. Microsoft")
    parser.add_argument("--product", default="Microsoft 365 Copilot", help="Product name, e.g. Microsoft 365 Copilot")
    parser.add_argument("--max-results", type=int, default=8, help="Maximum candidate URLs to return")
    parser.add_argument("--output", default="websearch_trust_center_output.json", help="Output JSON file path")
    args = parser.parse_args()

    try:
        result = discover_trust_urls(args.vendor.strip(), args.product.strip() if args.product else None, args.max_results)
    except BadRequestError as exc:
        raise SystemExit(
            "BadRequestError. The deployment may not support Responses API web_search, or web_search is not enabled.\n"
            f"{exc}"
        ) from exc
    except RateLimitError as exc:
        raise SystemExit(f"RateLimitError. Deployment quota unavailable.\n{exc}") from exc
    except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
        raise SystemExit(f"Azure OpenAI call failed: {type(exc).__name__}: {exc}") from exc

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
