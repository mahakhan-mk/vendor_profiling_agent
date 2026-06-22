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
MAX_CANDIDATES_TO_SCRAPE = 10
MAX_ACCEPTED_SOURCES = 5
MAX_SNIPPETS_PER_PAGE = 2
MAX_TOTAL_LLM_CHARS = 4000
MAX_SNIPPET_CHARS = 650
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

EXTERNAL_INCIDENT_SOURCE_BONUS = {
    "bleepingcomputer.com": 35,
    "securityweek.com": 30,
    "thehackernews.com": 25,
    "techcrunch.com": 25,
    "arstechnica.com": 20,
    "theregister.com": 20,
    "cybersecuritynews.com": 15,
    "cisa.gov": 20,
    "nvd.nist.gov": 15,
}

VENDOR_SELF_REPORT_DOMAINS = {
    "microsoft.com",
    "blogs.microsoft.com",
    "msrc.microsoft.com",
    "learn.microsoft.com",
    "portal.msrc.microsoft.com",
}

# Used only for vendor-owned trust/security/compliance/status page detection.
# Third-party sources remain valid for breach, vulnerability, and reputation evidence.
VENDOR_OWNED_DOMAINS = {
    "microsoft": [
        "microsoft.com",
        "blogs.microsoft.com",
        "msrc.microsoft.com",
        "learn.microsoft.com",
        "portal.msrc.microsoft.com",
    ],
    "okta": [
        "okta.com",
        "trust.okta.com",
        "status.okta.com",
    ],
    "snowflake": [
        "snowflake.com",
        "status.snowflake.com",
    ],
    "crowdstrike": [
        "crowdstrike.com",
    ],
}

TRUST_PAGE_TERMS = [
    "trust", "security", "msrc", "security-center", "compliance", "privacy", "status", "advisory"
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


def registered_source(url: str) -> str:
    domain = normalize_domain(url)
    for approved_domain in APPROVED_DOMAINS:
        if domain == approved_domain or domain.endswith("." + approved_domain):
            return approved_domain
    return domain


def is_vendor_self_report(url: str, vendor: str) -> bool:
    domain = normalize_domain(url)
    vendor_l = vendor.lower().replace(" ", "")
    if vendor_l and vendor_l in domain.replace(".", ""):
        return True
    return any(domain == d or domain.endswith("." + d) for d in VENDOR_SELF_REPORT_DOMAINS)


def deterministic_trust_signal(vendor: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect only vendor-owned trust/security/compliance/status/advisory pages.

    Third-party news/security sites are valid external evidence sources, but they must
    not be classified as the vendor's own trust center or security portal.
    """
    vendor_key = vendor.lower().replace(" ", "")
    vendor_domains = VENDOR_OWNED_DOMAINS.get(vendor_key, [])

    for item in evidence:
        url = item.get("source_url", "")
        title = item.get("source_title", "")
        haystack = f"{url} {title}".lower()
        domain = normalize_domain(url)

        is_vendor_owned = any(
            domain == vendor_domain or domain.endswith("." + vendor_domain)
            for vendor_domain in vendor_domains
        )
        if not is_vendor_owned:
            continue

        if any(term in haystack for term in TRUST_PAGE_TERMS):
            return {
                "found": True,
                "details": "Vendor-owned trust, security, compliance, status, or advisory page was found deterministically.",
                "source_url": url,
            }

    return {
        "found": False,
        "details": "No vendor-owned trust, security, compliance, status, or advisory page was detected in retrieved evidence.",
        "source_url": None,
    }


def classify_source_context(vendor: str, url: str, snippet: str) -> str:
    text = snippet.lower()
    if is_vendor_self_report(url, vendor):
        if any(term in text for term in ["customer", "customers", "client", "target organization", "victim organization"]):
            return "vendor_reported_customer_or_third_party_incident"
        return "vendor_self_reported_advisory_or_research"
    if any(term in text for term in ["affecting", "vulnerability", "cve", "patch", "advisory", "zero-day", "zero day"]):
        return "vulnerability_or_advisory"
    return "external_report_about_vendor_or_product"


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

    if not is_approved_url(url):
        return 0

    source = registered_source(url)
    score = approved_domain_score(url)

    if vendor.lower() in text:
        score += 25

    for keyword in SECURITY_KEYWORDS:
        if keyword in text:
            score += 8

    score += EXTERNAL_INCIDENT_SOURCE_BONUS.get(source, 0)

    if is_vendor_self_report(url, vendor):
        score -= 20

    if source in {"cisa.gov", "nvd.nist.gov", "msrc.microsoft.com"}:
        if any(term in text for term in ["cve-", "vulnerability", "advisory", "exploitation", "kev"]):
            score += 20

    for penalty in GENERIC_PENALTY_TERMS:
        if penalty in url.lower():
            score -= 35

    if re.search(r"/(search|tag|category|label)(/|$)", url.lower()):
        score -= 50

    return max(score, 0)


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
            source_url = page["resolved_url"] or page["requested_url"]
            snippets.append({
                "source_url": source_url,
                "source_title": page.get("title", ""),
                "source_type": classify_source_context(vendor, source_url, snippet),
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
            "source_title": item.get("source_title", "")[:160],
            "source_type": item.get("source_type", "unknown"),
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
    accepted_sources = 0

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
            accepted_sources += 1
            print(f"Accepted: {url} | snippets={min(len(snippets), MAX_SNIPPETS_PER_PAGE)}")
        except Exception as exc:
            error = str(exc)
            retrieval_failures.append({"url": url, "error": error[:300]})
            print(f"Skipped: {url} | {error[:160]}")

        if accepted_sources >= MAX_ACCEPTED_SOURCES:
            break
        if estimate_chars(cap_evidence_for_llm(evidence)) >= MAX_TOTAL_LLM_CHARS:
            break

    capped = cap_evidence_for_llm(evidence)
    return {
        "vendor": vendor,
        "evidence": capped,
        "deterministic_trust_signal": deterministic_trust_signal(vendor, capped),
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
    evidence_json = json.dumps(evidence_package, ensure_ascii=False, separators=(",", ":"))
    user_content = f"""
Vendor: {vendor}
Evidence package JSON:
{evidence_json}

Return strict JSON with exactly this top-level schema:
{{
  "vendor": "{vendor}",
  "confirmed_public_security_incidents": [
    {{
      "finding": "string",
      "incident_subject": "vendor|customer|third_party|vendor_product_or_service_abuse|unknown",
      "why_this_subject": "string",
      "evidence": "string",
      "source_url": "string",
      "confidence": "high|medium|low"
    }}
  ],
  "reported_unconfirmed_security_incidents": [],
  "vulnerabilities_or_advisories": [
    {{"finding":"string","affected_product_or_scope":"string|null","evidence":"string","source_url":"string","confidence":"high|medium|low"}}
  ],
  "certifications_or_compliance_claims": [],
  "trust_center_or_security_page_found": {{"found": false, "details": "string", "source_url": null}},
  "positive_signals": [],
  "negative_signals": [],
  "limitations": [],
  "requires_analyst_review": true
}}

Rules:
- Only put an item in confirmed_public_security_incidents when the evidence shows a security incident or compromise.
- If the vendor only reports on a customer attack, incident_subject must be customer, not vendor.
- If the evidence is mainly a CVE, patch, or advisory, put it under vulnerabilities_or_advisories, not confirmed_public_security_incidents.
- Use deterministic_trust_signal for trust_center_or_security_page_found when present.
- Third-party security news may support incidents or advisories, but it is not a vendor trust/security page.
- Keep each evidence field under 280 characters.
""".strip()
    return [
        {
            "role": "system",
            "content": (
                "You are a vendor security reputation analyst. Return only valid JSON. "
                "Use only the supplied evidence. Do not invent facts. Do not approve, reject, "
                "or assign final vendor risk. Distinguish vendor-owned incidents from customer incidents, "
                "third-party incidents, product abuse, and general advisories. Every finding must cite a source_url "
                "from evidence. Do not cite retrieval_failures as factual evidence."
            ),
        },
        {"role": "user", "content": user_content},
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

    trust_signal = evidence_package.get("deterministic_trust_signal")
    if trust_signal:
        result["trust_center_or_security_page_found"] = trust_signal

    serialized = json.dumps(result, ensure_ascii=False)
    cited_urls = set(re.findall(r"https?://[^\"'\s,}]+", serialized))
    invalid_urls = sorted(url for url in cited_urls if url not in allowed_urls)
    if invalid_urls:
        result.setdefault("limitations", []).append(
            "Model cited URLs outside retrieved evidence or cited failed-retrieval URLs. Treat those citations as rejected until manually reviewed: "
            + json.dumps(invalid_urls)
        )

    for item in result.get("confirmed_public_security_incidents", []) or []:
        finding = json.dumps(item, ensure_ascii=False).lower()
        if any(term in finding for term in ["customer", "target organization", "victim organization"]):
            item.setdefault("incident_subject", "customer")
            item.setdefault(
                "why_this_subject",
                "Evidence indicates the incident affected a customer or target organization, not necessarily the vendor itself.",
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
        "deterministic_trust_signal": {"found": False, "details": "No evidence retrieved.", "source_url": None},
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
