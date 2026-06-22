import os
import re
import json
import time
import hashlib
import requests
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional


FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_BASE_URL = os.getenv("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

MAX_CANDIDATES_TO_SCRAPE = 8
MAX_TOTAL_LLM_CHARS = 9000
MAX_SNIPPET_CHARS = 1200
MIN_PAGE_TEXT_CHARS = 500
REQUEST_TIMEOUT = 35

APPROVED_DOMAINS = {
    "microsoft.com": 100,
    "msrc.microsoft.com": 100,
    "nvd.nist.gov": 100,
    "cisa.gov": 100,
    "securityweek.com": 95,
    "thehackernews.com": 85,
    "techcrunch.com": 75,
    "cybersecuritynews.com": 70,
    "theregister.com": 70,
    "bleepingcomputer.com": 85,
    "arstechnica.com": 75,
}

SECURITY_KEYWORDS = [
    "breach", "data breach", "incident", "compromise", "vulnerability", "cve",
    "zero-day", "exploit", "ransomware", "malware", "credential", "phishing",
    "security advisory", "patch", "attack", "threat actor", "leak", "stolen",
]

GENERIC_PENALTY_TERMS = [
    "search/label", "/tag/", "/category/", "report", "whitepaper"
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


def search_web_placeholder(query: str) -> List[Dict[str, Any]]:
    """
    Replace this with your actual search provider.

    Acceptable implementations:
    - Bing Custom Search restricted to approved domains
    - Azure AI Foundry Bing grounding with domain allowlist
    - SerpAPI only if approved by business
    - Internal approved-source search service

    Return shape:
    [{"url": "...", "title": "...", "snippet": "..."}]
    """
    raise NotImplementedError("Wire this to your approved search provider.")


def build_queries(vendor: str) -> List[str]:
    return [
        f'{vendor} security incident',
        f'{vendor} data breach',
        f'{vendor} breach',
        f'{vendor} vulnerability',
        f'{vendor} cyber attack',
        f'{vendor} CVE',
        f'{vendor} security advisory',
    ]


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

    if not is_approved_url(url):
        score = 0

    return max(score, 0)


def discover_candidates(vendor: str) -> List[Dict[str, Any]]:
    seen = set()
    candidates = []

    for query in build_queries(vendor):
        print(f"\nSearching: {query}")
        results = search_web_placeholder(query)
        print(f"Returned {len(results)} results")

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
    if not FIRECRAWL_API_KEY:
        raise RuntimeError("Missing FIRECRAWL_API_KEY")

    endpoint = f"{FIRECRAWL_BASE_URL}/scrape"

    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "timeout": REQUEST_TIMEOUT * 1000,
    }

    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=REQUEST_TIMEOUT + 10)

    if response.status_code >= 400:
        raise RuntimeError(f"Scrape HTTP {response.status_code}: {response.text[:500]}")

    data = response.json()

    if not data.get("success", True):
        raise RuntimeError(f"Scrape failed: {json.dumps(data)[:500]}")

    page_data = data.get("data", data)
    markdown = page_data.get("markdown") or ""

    return {
        "url": page_data.get("metadata", {}).get("sourceURL") or url,
        "title": page_data.get("metadata", {}).get("title") or "",
        "markdown": markdown,
    }


def clean_text(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def extract_relevant_snippets(vendor: str, page: Dict[str, Any]) -> List[Dict[str, Any]]:
    markdown = clean_text(page.get("markdown", ""))

    if len(markdown) < MIN_PAGE_TEXT_CHARS:
        raise RuntimeError("Low content returned")

    sentences = split_sentences(markdown)
    snippets = []

    for i, sentence in enumerate(sentences):
        lower = sentence.lower()

        has_vendor = vendor.lower() in lower
        has_security_term = any(keyword in lower for keyword in SECURITY_KEYWORDS)

        if not (has_vendor or has_security_term):
            continue

        start = max(0, i - 2)
        end = min(len(sentences), i + 3)
        snippet = " ".join(sentences[start:end])
        snippet = snippet[:MAX_SNIPPET_CHARS].strip()

        if snippet:
            snippets.append({
                "source_url": page["url"],
                "source_title": page.get("title", ""),
                "snippet": snippet,
            })

    return dedupe_snippets(snippets)


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

            for snippet in snippets[:3]:
                snippet["candidate_score"] = candidate["score"]
                evidence.append(snippet)

            print(f"Accepted: {url} | snippets={len(snippets[:3])}")

        except Exception as e:
            error = str(e)
            retrieval_failures.append({
                "url": url,
                "error": error[:300],
            })
            print(f"Skipped: {url} | {error[:140]}")

        if estimate_llm_chars(evidence) >= MAX_TOTAL_LLM_CHARS:
            break

    return {
        "vendor": vendor,
        "evidence": cap_evidence_for_llm(evidence),
        "retrieval_failures": retrieval_failures,
        "completed_with_limitations": len(evidence) == 0 or len(retrieval_failures) > 0,
    }


def estimate_llm_chars(evidence: List[Dict[str, Any]]) -> int:
    return len(json.dumps(evidence, ensure_ascii=False))


def cap_evidence_for_llm(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    capped = []
    total = 0

    for item in evidence:
        compact = {
            "source_url": item["source_url"],
            "source_title": item.get("source_title", "")[:180],
            "snippet": item["snippet"][:MAX_SNIPPET_CHARS],
        }

        size = len(json.dumps(compact, ensure_ascii=False))

        if total + size > MAX_TOTAL_LLM_CHARS:
            break

        capped.append(compact)
        total += size

    return capped


def build_llm_prompt(vendor: str, evidence_package: Dict[str, Any]) -> List[Dict[str, str]]:
    evidence_json = json.dumps(evidence_package, ensure_ascii=False, indent=2)

    system = """
You are a vendor security reputation analyst.

Rules:
- Use only the evidence provided.
- Do not invent incidents, breaches, CVEs, certifications, or conclusions.
- Do not approve or reject the vendor.
- Do not assign final risk.
- Every finding must cite a source_url from the evidence.
- If evidence is weak, say so plainly.
- Output strict JSON only.
"""

    user = f"""
Vendor: {vendor}

Evidence package:
{evidence_json}

Return JSON with this schema:
{{
  "vendor": "...",
  "confirmed_public_security_incidents": [
    {{
      "finding": "...",
      "evidence": "...",
      "source_url": "...",
      "confidence": "high|medium|low"
    }}
  ],
  "reported_unconfirmed_security_incidents": [],
  "vulnerabilities_or_advisories": [],
  "certifications_or_compliance_claims": [],
  "trust_center_or_security_page_found": {{
    "found": true,
    "details": "...",
    "source_url": "..."
  }},
  "positive_signals": [],
  "negative_signals": [],
  "limitations": [],
  "requires_analyst_review": true
}}
"""

    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]


def call_azure_openai(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
        raise RuntimeError("Missing Azure OpenAI environment variables")

    url = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )

    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1600,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code >= 400:
        raise RuntimeError(f"Azure OpenAI HTTP {response.status_code}: {response.text[:1000]}")

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    return json.loads(content)


def validate_output(result: Dict[str, Any], evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    allowed_urls = {item["source_url"] for item in evidence_package["evidence"]}

    result["requires_analyst_review"] = True

    serialized = json.dumps(result, ensure_ascii=False)

    cited_urls = set(re.findall(r"https?://[^\"'\s,}]+", serialized))
    invalid_urls = [url for url in cited_urls if url not in allowed_urls]

    if invalid_urls:
        result.setdefault("limitations", []).append(
            f"Model cited URLs not present in retrieved evidence and they were rejected: {invalid_urls}"
        )

    if evidence_package.get("completed_with_limitations"):
        result.setdefault("limitations", []).append(
            "Collection completed with limitations because one or more approved sources could not be retrieved or no relevant evidence was found."
        )

    return result


def main() -> None:
    vendor = "Microsoft"

    candidates = discover_candidates(vendor)

    print("\nTrusted candidates:")
    for candidate in candidates:
        print(f"{candidate['score']:3} | {candidate['url']}")

    if not candidates:
        raise RuntimeError("No approved candidate sources found")

    evidence_package = collect_evidence(vendor, candidates)

    print("\nEvidence chars sent to LLM:", estimate_llm_chars(evidence_package["evidence"]))
    print("Evidence snippets sent to LLM:", len(evidence_package["evidence"]))

    if not evidence_package["evidence"]:
        output = {
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
            "retrieval_failures": evidence_package["retrieval_failures"],
        }
    else:
        messages = build_llm_prompt(vendor, evidence_package)
        output = call_azure_openai(messages)
        output = validate_output(output, evidence_package)
        output["retrieval_failures"] = evidence_package["retrieval_failures"]

    print("\nFinal output:")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()