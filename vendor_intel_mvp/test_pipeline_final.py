import json
import os
import re
import hashlib
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

FIRECRAWL_BASE_URL = os.getenv("FIRECRAWL_BASE_URL", "http://localhost:3002").rstrip("/")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = (
    os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_DEPLOYMENT")
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

TEST_VENDOR = os.getenv("TEST_VENDOR", "Microsoft")

MAX_SEARCH_RESULTS_PER_QUERY = 10
MAX_CANDIDATES_TO_SCRAPE = 8
MAX_SNIPPETS_PER_PAGE = 3
MAX_TOTAL_LLM_CHARS = 6000
MAX_SNIPPET_CHARS = 800
MIN_PAGE_TEXT_CHARS = 300
REQUEST_TIMEOUT_SECONDS = 120

APPROVED_DOMAINS = {
    "microsoft.com": 100,
    "msrc.microsoft.com": 100,
    "nvd.nist.gov": 100,
    "cisa.gov": 100,
    "securityweek.com": 95,
    "bleepingcomputer.com": 85,
    "thehackernews.com": 85,
    "techcrunch.com": 75,
    "arstechnica.com": 75,
    "cybersecuritynews.com": 70,
    "theregister.com": 70,
    "cloudsecurityalliance.org": 65,
}

SECURITY_KEYWORDS = [
    "breach", "data breach", "incident", "compromise", "vulnerability", "cve",
    "zero-day", "zeroday", "exploit", "ransomware", "malware", "credential",
    "phishing", "security advisory", "patch", "attack", "threat actor", "leak",
    "stolen", "supply-chain", "supply chain", "unauthorized access", "exfiltration",
]

GENERIC_PENALTY_TERMS = [
    "search/label", "/tag/", "/category/", "/topics/", "whitepaper", "report",
]


def normalize_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def approved_domain_score(url: str) -> int:
    domain = normalize_domain(url)
    for approved_domain, score in APPROVED_DOMAINS.items():
        if domain == approved_domain or domain.endswith("." + approved_domain):
            return score
    return 0


def is_approved_url(url: str) -> bool:
    return approved_domain_score(url) > 0


def build_queries(vendor: str) -> List[str]:
    return [
        f"{vendor} security incident",
        f"{vendor} data breach",
        f"{vendor} breach",
        f"{vendor} vulnerability",
        f"{vendor} cyber attack",
        f"{vendor} CVE",
        f"{vendor} security advisory",
    ]


def firecrawl_search(query: str) -> List[Dict[str, Any]]:
    response = requests.post(
        f"{FIRECRAWL_BASE_URL}/v2/search",
        json={"query": query, "limit": MAX_SEARCH_RESULTS_PER_QUERY},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Search HTTP {response.status_code}: {response.text[:500]}")

    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Search failed: {json.dumps(payload)[:500]}")

    web_results = payload.get("data", {}).get("web", [])
    results = []
    for item in web_results:
        url = item.get("url") or item.get("href") or ""
        if not url or not is_approved_url(url):
            continue
        results.append({
            "url": url,
            "title": item.get("title", ""),
            "snippet": item.get("description") or item.get("snippet") or item.get("body") or "",
        })
    return results


def candidate_score(vendor: str, result: Dict[str, Any]) -> int:
    url = result.get("url", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    text = f"{url} {title} {snippet}".lower()

    score = approved_domain_score(url)
    if vendor.lower() in text:
        score += 25
    for keyword in SECURITY_KEYWORDS:
        if keyword in text:
            score += 8
    for penalty in GENERIC_PENALTY_TERMS:
        if penalty in url.lower():
            score -= 25
    return max(score, 0) if is_approved_url(url) else 0


def discover_candidates(vendor: str) -> List[Dict[str, Any]]:
    seen = set()
    candidates = []

    for query in build_queries(vendor):
        print(f"\nSearching: {query}")
        try:
            results = firecrawl_search(query)
        except Exception as exc:
            print(f"Search failed: {exc}")
            continue

        print(f"Returned {len(results)} approved results")
        for result in results:
            url = result.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            score = candidate_score(vendor, result)
            print(f"{score:3} | {url}")
            if score > 0:
                candidates.append({
                    "url": url,
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "score": score,
                    "query": query,
                })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def firecrawl_scrape(url: str) -> Dict[str, Any]:
    response = requests.post(
        f"{FIRECRAWL_BASE_URL}/v2/scrape",
        json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Scrape HTTP {response.status_code}: {response.text[:500]}")

    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Scrape failed: {json.dumps(payload)[:500]}")

    data = payload.get("data", {})
    metadata = data.get("metadata", {})
    markdown = data.get("markdown") or ""
    if len(markdown) < MIN_PAGE_TEXT_CHARS:
        raise RuntimeError(f"Scrape returned too little markdown: {len(markdown)} chars")

    return {
        "requested_url": url,
        "resolved_url": metadata.get("url") or metadata.get("sourceURL") or url,
        "title": metadata.get("title") or "",
        "status_code": metadata.get("statusCode"),
        "markdown": markdown,
    }


def clean_text(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:16]


def dedupe_snippets(snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for item in snippets:
        key = hash_text(item["snippet"][:500])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def extract_relevant_snippets(vendor: str, page: Dict[str, Any]) -> List[Dict[str, Any]]:
    markdown = clean_text(page.get("markdown", ""))
    if len(markdown) < MIN_PAGE_TEXT_CHARS:
        raise RuntimeError("Low content returned")

    sentences = split_sentences(markdown)
    snippets = []
    vendor_l = vendor.lower()

    for i, sentence in enumerate(sentences):
        lower = sentence.lower()
        if vendor_l not in lower and not any(keyword in lower for keyword in SECURITY_KEYWORDS):
            continue

        start = max(0, i - 2)
        end = min(len(sentences), i + 3)
        snippet = " ".join(sentences[start:end])[:MAX_SNIPPET_CHARS].strip()
        if snippet:
            snippets.append({
                "source_url": page["resolved_url"] or page["requested_url"],
                "source_title": page.get("title", ""),
                "snippet": snippet,
            })

    return dedupe_snippets(snippets)


def estimate_chars(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False))


def cap_evidence_for_llm(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    capped = []
    total = 0
    for item in evidence:
        compact = {
            "source_url": item["source_url"],
            "source_title": item.get("source_title", "")[:180],
            "snippet": item["snippet"][:MAX_SNIPPET_CHARS],
        }
        size = estimate_chars(compact)
        if total + size > MAX_TOTAL_LLM_CHARS:
            break
        capped.append(compact)
        total += size
    return capped


def collect_evidence(vendor: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence = []
    retrieval_failures = []

    for candidate in candidates[:MAX_CANDIDATES_TO_SCRAPE]:
        url = candidate["url"]
        print(f"\nTrying URL: {url}")
        try:
            page = firecrawl_scrape(url)
            snippets = extract_relevant_snippets(vendor, page)
            if not snippets:
                raise RuntimeError("No relevant security snippets found")

            for snippet in snippets[:MAX_SNIPPETS_PER_PAGE]:
                snippet["candidate_score"] = candidate["score"]
                evidence.append(snippet)
            print(f"Accepted: {url} | snippets={min(len(snippets), MAX_SNIPPETS_PER_PAGE)}")
        except Exception as exc:
            error = str(exc)
            retrieval_failures.append({"url": url, "error": error[:300]})
            print(f"Skipped: {url} | {error[:160]}")

        if estimate_chars(cap_evidence_for_llm(evidence)) >= MAX_TOTAL_LLM_CHARS:
            break

    capped = cap_evidence_for_llm(evidence)
    return {
        "vendor": vendor,
        "evidence": capped,
        "retrieval_failures": retrieval_failures,
        "completed_with_limitations": bool(retrieval_failures) or not bool(capped),
    }


def parse_json_response(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return json.loads(text)


def build_messages(vendor: str, evidence_package: Dict[str, Any]) -> List[Dict[str, str]]:
    evidence_json = json.dumps(evidence_package, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "You are a vendor security reputation analyst. Return only valid JSON. "
                "Use only the provided evidence. Do not invent facts. Do not approve, reject, "
                "or assign final vendor risk. Every finding must cite a source_url from the evidence."
            ),
        },
        {
            "role": "user",
            "content": f"""
Vendor: {vendor}

Evidence package:
{evidence_json}

Return JSON with this schema:
{{
  "vendor": "{vendor}",
  "confirmed_public_security_incidents": [
    {{"finding": "string", "evidence": "string", "source_url": "string", "confidence": "high|medium|low"}}
  ],
  "reported_unconfirmed_security_incidents": [],
  "vulnerabilities_or_advisories": [],
  "certifications_or_compliance_claims": [],
  "trust_center_or_security_page_found": {{"found": false, "details": "string", "source_url": null}},
  "positive_signals": [],
  "negative_signals": [],
  "limitations": [],
  "requires_analyst_review": true
}}
""".strip(),
        },
    ]


def call_azure_openai(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_MODEL:
        raise RuntimeError(
            "Missing Azure OpenAI config. Required: AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_GPT_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT."
        )

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return parse_json_response(response.choices[0].message.content or "{}")


def validate_output(result: Dict[str, Any], evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    allowed_urls = {item["source_url"] for item in evidence_package.get("evidence", [])}
    result["requires_analyst_review"] = True

    serialized = json.dumps(result, ensure_ascii=False)
    cited_urls = set(re.findall(r"https?://[^\"'\s,}]+", serialized))
    invalid_urls = sorted(url for url in cited_urls if url not in allowed_urls)
    if invalid_urls:
        result.setdefault("limitations", []).append(
            "Model cited URLs outside retrieved evidence and they require rejection or manual review: "
            + json.dumps(invalid_urls)
        )

    if evidence_package.get("completed_with_limitations"):
        result.setdefault("limitations", []).append(
            "Collection completed with limitations because at least one approved source failed retrieval or no evidence was found."
        )
    return result


def empty_output(vendor: str, evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "vendor": vendor,
        "confirmed_public_security_incidents": [],
        "reported_unconfirmed_security_incidents": [],
        "vulnerabilities_or_advisories": [],
        "certifications_or_compliance_claims": [],
        "trust_center_or_security_page_found": {
            "found": False,
            "details": "No retrievable approved-source evidence found.",
            "source_url": None,
        },
        "positive_signals": [],
        "negative_signals": [],
        "limitations": [
            "No approved source could be scraped successfully.",
            "The run must be reviewed by an analyst.",
        ],
        "requires_analyst_review": True,
        "retrieval_failures": evidence_package.get("retrieval_failures", []),
    }


def main() -> None:
    vendor = TEST_VENDOR
    candidates = discover_candidates(vendor)

    print("\nTrusted candidates:")
    for candidate in candidates:
        print(f"{candidate['score']:3} | {candidate['url']}")

    evidence_package = collect_evidence(vendor, candidates) if candidates else {
        "vendor": vendor,
        "evidence": [],
        "retrieval_failures": [],
        "completed_with_limitations": True,
    }

    print("\nEvidence chars sent to LLM:", estimate_chars(evidence_package["evidence"]))
    print("Evidence snippets sent to LLM:", len(evidence_package["evidence"]))

    if not evidence_package["evidence"]:
        output = empty_output(vendor, evidence_package)
    else:
        messages = build_messages(vendor, evidence_package)
        print("Prompt chars sent to Azure:", sum(len(m["content"]) for m in messages))
        output = call_azure_openai(messages)
        output = validate_output(output, evidence_package)
        output["retrieval_failures"] = evidence_package["retrieval_failures"]

    with open("vendor_incident_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nFinal output:")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
