import argparse
import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


@dataclass(frozen=True)
class VendorIntelConfig:
    firecrawl_base_url: str = os.getenv("FIRECRAWL_BASE_URL", "http://localhost:3002").rstrip("/")
    azure_openai_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_model: Optional[str] = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    max_search_results_per_query: int = int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "20"))
    max_candidates_to_scrape: int = int(os.getenv("MAX_CANDIDATES_TO_SCRAPE", "12"))
    max_accepted_sources: int = int(os.getenv("MAX_ACCEPTED_SOURCES", "5"))
    max_snippets_per_page: int = int(os.getenv("MAX_SNIPPETS_PER_PAGE", "2"))
    max_total_llm_chars: int = int(os.getenv("MAX_TOTAL_LLM_CHARS", "4000"))
    max_snippet_chars: int = int(os.getenv("MAX_SNIPPET_CHARS", "650"))
    min_page_text_chars: int = int(os.getenv("MIN_PAGE_TEXT_CHARS", "300"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    max_search_queries: int = int(os.getenv("MAX_SEARCH_QUERIES", "10"))
    search_sources: List[str] = field(default_factory=lambda: ["web", "news"])
    firecrawl_enterprise_mode: Optional[str] = os.getenv("FIRECRAWL_ENTERPRISE_MODE")


APPROVED_DOMAINS: Dict[str, int] = {
    "bleepingcomputer.com": 85,
    "securityweek.com": 95,
    "thehackernews.com": 85,
    "techcrunch.com": 75,
    "arstechnica.com": 75,
    "theregister.com": 70,
    "cybersecuritynews.com": 70,
    "cisa.gov": 100,
    "nvd.nist.gov": 100,
}

EXTERNAL_INCIDENT_SOURCE_BONUS: Dict[str, int] = {
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

SECURITY_KEYWORDS = [
    "breach", "data breach", "incident", "compromise", "compromised", "vulnerability", "cve",
    "zero-day", "zero day", "zeroday", "exploit", "exploited", "ransomware", "malware",
    "credential", "phishing", "security advisory", "patch", "attack", "threat actor", "leak",
    "leaked", "stolen", "supply-chain", "supply chain", "unauthorized access", "exfiltration",
    "intrusion", "backdoor", "token theft", "account takeover", "privilege escalation",
]

INCIDENT_TERMS = [
    "breach", "data breach", "incident", "compromise", "compromised", "unauthorized access",
    "exfiltration", "stolen", "leaked", "ransomware", "malware", "supply chain", "supply-chain",
    "intrusion", "backdoor", "account takeover", "token theft",
]

VULNERABILITY_TERMS = [
    "cve-", "vulnerability", "zero-day", "zero day", "exploit", "exploited", "patch",
    "security update", "security advisory", "kev", "privilege escalation", "remote code execution",
]

GENERIC_PENALTY_TERMS = [
    "search/label", "/tag/", "/category/", "/topics/", "/author/", "whitepaper", "newsletter",
]


@dataclass
class Candidate:
    url: str
    title: str
    snippet: str
    score: int
    query: str
    source: str


@dataclass
class RetrievalFailure:
    url: str
    error: str


@dataclass
class EvidencePackage:
    vendor: str
    product: Optional[str]
    evidence: List[Dict[str, Any]]
    retrieval_failures: List[Dict[str, str]]
    completed_with_limitations: bool


def normalize_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def registered_source(url: str) -> str:
    domain = normalize_domain(url)
    for approved_domain in APPROVED_DOMAINS:
        if domain == approved_domain or domain.endswith("." + approved_domain):
            return approved_domain
    return domain


def approved_domain_score(url: str) -> int:
    return APPROVED_DOMAINS.get(registered_source(url), 0)


def is_approved_url(url: str) -> bool:
    return approved_domain_score(url) > 0


def compact_entity(vendor: str, product: Optional[str]) -> str:
    if product and product.strip().lower() != vendor.strip().lower():
        return f"{vendor} {product}".strip()
    return vendor.strip()


SEARCH_TERMS = [
    "breach",
    "data breach",
    "security incident",
    "compromised",
    "unauthorized access",
    "data theft",
    "token theft",
    "credential theft",
    "ransomware",
    "malware",
    "supply chain attack",
    "vulnerability",
    "CVE",
    "zero day",
    "exploit",
    "security advisory",
]


def build_queries(vendor: str, product: Optional[str] = None) -> List[str]:
    """
    Build broad vendor/product security queries.

    Domain restriction is NOT encoded with site:<domain>. Firecrawl documents
    includeDomains as the supported way to restrict the search corpus.
    This keeps the query natural while still enforcing the approved-source
    boundary in the API request and again locally after results return.
    """
    entity = compact_entity(vendor, product)
    vendor_clean = vendor.strip()

    roots: List[str] = []
    for root in [entity, vendor_clean]:
        root = root.strip()
        if root and root.lower() not in {r.lower() for r in roots}:
            roots.append(root)

    queries: List[str] = []
    seen = set()
    for root in roots:
        for term in SEARCH_TERMS:
            q = f"{root} {term}"
            key = q.lower()
            if key not in seen:
                seen.add(key)
                queries.append(q)

    return queries

def firecrawl_search(config: VendorIntelConfig, query: str) -> List[Dict[str, str]]:
    payload: Dict[str, Any] = {
        "query": query,
        "limit": config.max_search_results_per_query,
        "includeDomains": list(APPROVED_DOMAINS.keys()),
        "sources": config.search_sources,
    }

    # Optional enterprise/ZDR mode. Example: FIRECRAWL_ENTERPRISE_MODE=anon or zdr.
    if config.firecrawl_enterprise_mode:
        payload["enterprise"] = [config.firecrawl_enterprise_mode]

    response = requests.post(
        f"{config.firecrawl_base_url}/v2/search",
        json=payload,
        timeout=config.request_timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Search HTTP {response.status_code}: {response.text[:500]}")

    body = response.json()
    if not body.get("success"):
        raise RuntimeError(f"Search failed: {json.dumps(body)[:500]}")

    data = body.get("data", {})
    raw_results: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        for source_type in ["web", "news"]:
            for item in data.get(source_type, []) or []:
                item = dict(item)
                item["_firecrawl_source_type"] = source_type
                raw_results.append(item)
    elif isinstance(data, list):
        raw_results.extend(data)

    results: List[Dict[str, str]] = []
    for item in raw_results:
        url = item.get("url") or item.get("href") or ""
        if not url or not is_approved_url(url):
            continue
        results.append(
            {
                "url": url,
                "title": item.get("title", ""),
                "snippet": item.get("description") or item.get("snippet") or item.get("body") or "",
                "result_type": item.get("_firecrawl_source_type", "web"),
            }
        )
    return results


def candidate_score(vendor: str, product: Optional[str], result: Dict[str, str]) -> int:
    url = result.get("url", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    text = f"{url} {title} {snippet}".lower()

    if not is_approved_url(url):
        return 0

    source = registered_source(url)
    score = approved_domain_score(url) + EXTERNAL_INCIDENT_SOURCE_BONUS.get(source, 0)

    vendor_l = vendor.lower()
    product_l = (product or "").lower()
    entity_l = compact_entity(vendor, product).lower()

    if entity_l and entity_l in text:
        score += 35
    elif product_l and product_l in text:
        score += 25
    elif vendor_l and vendor_l in text:
        score += 20
    else:
        score -= 40

    for keyword in SECURITY_KEYWORDS:
        if keyword in text:
            score += 6

    if any(term in text for term in INCIDENT_TERMS):
        score += 20
    if any(term in text for term in VULNERABILITY_TERMS):
        score += 12

    for penalty in GENERIC_PENALTY_TERMS:
        if penalty in url.lower():
            score -= 35

    if re.search(r"/(search|tag|category|label|author)(/|$)", url.lower()):
        score -= 50

    return max(score, 0)


def discover_candidates(config: VendorIntelConfig, vendor: str, product: Optional[str]) -> List[Candidate]:
    seen_urls = set()
    candidates: List[Candidate] = []
    queries = build_queries(vendor, product)[: config.max_search_queries]

    print(f"Planned search queries: {len(queries)}")

    source_hit_counts: Dict[str, int] = {domain: 0 for domain in EXTERNAL_INCIDENT_SOURCE_BONUS}

    for query in queries:
        print(f"\nSearching: {query}")
        try:
            results = firecrawl_search(config, query)
        except Exception as exc:
            print(f"Search failed: {exc}")
            continue

        print(f"Returned {len(results)} approved results")
        for result in results:
            url = result.get("url", "")
            if not url or url in seen_urls:
                continue

            score = candidate_score(vendor, product, result)
            source = registered_source(url)
            print(f"{score:3} | {url}")

            if score > 0:
                seen_urls.add(url)
                source_hit_counts[source] = source_hit_counts.get(source, 0) + 1
                candidates.append(
                    Candidate(
                        url=url,
                        title=result.get("title", ""),
                        snippet=result.get("snippet", ""),
                        score=score,
                        query=query,
                        source=source,
                    )
                )

    candidates.sort(key=lambda c: (c.score, EXTERNAL_INCIDENT_SOURCE_BONUS.get(c.source, 0)), reverse=True)

    print("\nApproved source coverage:")
    for domain, count in source_hit_counts.items():
        print(f"{domain}: {count}")

    return candidates


def firecrawl_scrape(config: VendorIntelConfig, url: str) -> Dict[str, Any]:
    response = requests.post(
        f"{config.firecrawl_base_url}/v2/scrape",
        json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        timeout=config.request_timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Scrape HTTP {response.status_code}: {response.text[:500]}")

    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Scrape failed: {json.dumps(payload)[:500]}")

    data = payload.get("data", {})
    metadata = data.get("metadata", {})
    markdown = data.get("markdown") or ""
    if len(markdown) < config.min_page_text_chars:
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
    unique: List[Dict[str, Any]] = []
    for item in snippets:
        key = hash_text(item["snippet"][:500])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def classify_source_context(url: str, snippet: str) -> str:
    text = snippet.lower()
    if any(term in text for term in INCIDENT_TERMS):
        return "external_incident_or_breach_report"
    if any(term in text for term in VULNERABILITY_TERMS):
        return "external_vulnerability_or_advisory_report"
    return "external_security_reputation_report"


def extract_relevant_snippets(config: VendorIntelConfig, vendor: str, product: Optional[str], page: Dict[str, Any]) -> List[Dict[str, Any]]:
    markdown = clean_text(page.get("markdown", ""))
    if len(markdown) < config.min_page_text_chars:
        raise RuntimeError("Low content returned")

    sentences = split_sentences(markdown)
    snippets: List[Dict[str, Any]] = []
    vendor_l = vendor.lower()
    product_l = (product or "").lower()
    entity_l = compact_entity(vendor, product).lower()

    for i, sentence in enumerate(sentences):
        lower = sentence.lower()
        has_entity = entity_l in lower or vendor_l in lower or (product_l and product_l in lower)
        has_security = any(keyword in lower for keyword in SECURITY_KEYWORDS)
        if not (has_entity or has_security):
            continue

        start = max(0, i - 2)
        end = min(len(sentences), i + 3)
        snippet = " ".join(sentences[start:end])[: config.max_snippet_chars].strip()
        if not snippet:
            continue

        source_url = page["resolved_url"] or page["requested_url"]
        snippets.append(
            {
                "source_url": source_url,
                "source_title": page.get("title", ""),
                "source_domain": registered_source(source_url),
                "source_type": classify_source_context(source_url, snippet),
                "snippet": snippet,
            }
        )
    return dedupe_snippets(snippets)


def estimate_chars(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False))


def cap_evidence_for_llm(config: VendorIntelConfig, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    capped: List[Dict[str, Any]] = []
    total = 0
    for item in evidence:
        compact = {
            "source_url": item["source_url"],
            "source_title": item.get("source_title", "")[:160],
            "source_domain": item.get("source_domain", ""),
            "source_type": item.get("source_type", "unknown"),
            "snippet": item["snippet"][: config.max_snippet_chars],
        }
        size = estimate_chars(compact)
        if total + size > config.max_total_llm_chars:
            break
        capped.append(compact)
        total += size
    return capped


def collect_evidence(config: VendorIntelConfig, vendor: str, product: Optional[str], candidates: List[Candidate]) -> EvidencePackage:
    evidence: List[Dict[str, Any]] = []
    failures: List[RetrievalFailure] = []
    accepted_sources = 0

    for candidate in candidates[: config.max_candidates_to_scrape]:
        print(f"\nTrying URL: {candidate.url}")
        try:
            page = firecrawl_scrape(config, candidate.url)
            snippets = extract_relevant_snippets(config, vendor, product, page)
            if not snippets:
                raise RuntimeError("No relevant security snippets found")

            for snippet in snippets[: config.max_snippets_per_page]:
                snippet["candidate_score"] = candidate.score
                evidence.append(snippet)
            accepted_sources += 1
            print(f"Accepted: {candidate.url} | snippets={min(len(snippets), config.max_snippets_per_page)}")
        except Exception as exc:
            error = str(exc)
            failures.append(RetrievalFailure(candidate.url, error[:300]))
            print(f"Skipped: {candidate.url} | {error[:160]}")

        if accepted_sources >= config.max_accepted_sources:
            break
        if estimate_chars(cap_evidence_for_llm(config, evidence)) >= config.max_total_llm_chars:
            break

    capped = cap_evidence_for_llm(config, evidence)
    return EvidencePackage(
        vendor=vendor,
        product=product,
        evidence=capped,
        retrieval_failures=[failure.__dict__ for failure in failures],
        completed_with_limitations=bool(failures) or not bool(capped),
    )


def parse_json_response(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return json.loads(text)


def build_messages(vendor: str, product: Optional[str], evidence_package: EvidencePackage) -> List[Dict[str, str]]:
    evidence_json = json.dumps(evidence_package.__dict__, ensure_ascii=False, separators=(",", ":"))
    product_line = f"Product: {product}" if product else "Product: null"
    user_content = f"""
Vendor: {vendor}
{product_line}
Evidence package JSON:
{evidence_json}

Return strict JSON with exactly this top-level schema:
{{
  "vendor": "{vendor}",
  "product": {json.dumps(product)},
  "confirmed_public_security_incidents": [
    {{
      "finding": "string",
      "incident_subject": "vendor|product|customer|third_party|vendor_product_or_service_abuse|unknown",
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
  "positive_signals": [],
  "negative_signals": [],
  "limitations": [],
  "requires_analyst_review": true
}}

Rules:
- Use only evidence snippets. Do not use retrieval_failures as factual evidence.
- Third-party websites are valid sources for breaches, incidents, CVEs, advisories, and reputation findings.
- Put confirmed breaches, compromises, malware, ransomware, supply-chain attacks, token theft, data theft, or account compromise under confirmed_public_security_incidents only when evidence supports it.
- Put CVEs, exploited vulnerabilities, patches, and advisories under vulnerabilities_or_advisories.
- If the article says the vendor product was abused against customers, incident_subject must be vendor_product_or_service_abuse or customer, not vendor.
- Do not approve, reject, or assign final vendor risk.
- Every finding must cite one source_url from evidence.
- Keep each evidence field under 280 characters.
""".strip()
    return [
        {
            "role": "system",
            "content": (
                "You are a vendor security reputation analyst. Return only valid JSON. "
                "Use only the supplied evidence. Do not invent facts. Do not make final risk decisions. "
                "Separate vendor-owned incidents, product abuse, customer incidents, third-party incidents, and general advisories."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def call_azure_openai(config: VendorIntelConfig, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if not config.azure_openai_endpoint or not config.azure_openai_api_key or not config.azure_openai_model:
        raise RuntimeError(
            "Missing Azure OpenAI config. Required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "and AZURE_OPENAI_GPT_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT."
        )

    client = AzureOpenAI(
        azure_endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.azure_openai_api_version,
    )
    response = client.chat.completions.create(
        model=config.azure_openai_model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return parse_json_response(response.choices[0].message.content or "{}")


def validate_output(result: Dict[str, Any], evidence_package: EvidencePackage) -> Dict[str, Any]:
    allowed_urls = {item["source_url"] for item in evidence_package.evidence}
    result["requires_analyst_review"] = True

    serialized = json.dumps(result, ensure_ascii=False)
    cited_urls = set(re.findall(r"https?://[^\"'\s,}]+", serialized))
    invalid_urls = sorted(url for url in cited_urls if url not in allowed_urls)
    if invalid_urls:
        result.setdefault("limitations", []).append(
            "Model cited URLs outside retrieved evidence. Treat those citations as rejected until manually reviewed: "
            + json.dumps(invalid_urls)
        )

    if evidence_package.completed_with_limitations:
        result.setdefault("limitations", []).append(
            "Collection completed with limitations because at least one approved source failed retrieval or no evidence was found."
        )

    result["retrieval_failures"] = evidence_package.retrieval_failures
    return result


def empty_output(vendor: str, product: Optional[str], evidence_package: EvidencePackage) -> Dict[str, Any]:
    return {
        "vendor": vendor,
        "product": product,
        "confirmed_public_security_incidents": [],
        "reported_unconfirmed_security_incidents": [],
        "vulnerabilities_or_advisories": [],
        "certifications_or_compliance_claims": [],
        "positive_signals": [],
        "negative_signals": [],
        "limitations": [
            "No approved source could be scraped successfully.",
            "The run must be reviewed by an analyst.",
        ],
        "requires_analyst_review": True,
        "retrieval_failures": evidence_package.retrieval_failures,
    }


def run_vendor_intel(vendor: str, product: Optional[str], output_path: str) -> Dict[str, Any]:
    config = VendorIntelConfig()
    candidates = discover_candidates(config, vendor, product)

    print("\nTrusted candidates:")
    for candidate in candidates:
        print(f"{candidate.score:3} | {candidate.url}")

    evidence_package = collect_evidence(config, vendor, product, candidates) if candidates else EvidencePackage(
        vendor=vendor,
        product=product,
        evidence=[],
        retrieval_failures=[],
        completed_with_limitations=True,
    )

    print("\nEvidence chars sent to LLM:", estimate_chars(evidence_package.evidence))
    print("Evidence snippets sent to LLM:", len(evidence_package.evidence))

    if not evidence_package.evidence:
        output = empty_output(vendor, product, evidence_package)
    else:
        messages = build_messages(vendor, product, evidence_package)
        print("Prompt chars sent to Azure:", sum(len(m["content"]) for m in messages))
        output = call_azure_openai(config, messages)
        output = validate_output(output, evidence_package)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nFinal output:")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved: {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized approved-source vendor breach and vulnerability intelligence MVP.")
    parser.add_argument("--vendor", required=True, help="Vendor/company name, e.g. Okta, Snowflake, Microsoft")
    parser.add_argument("--product", default=None, help="Optional product name, e.g. Microsoft 365 Copilot")
    parser.add_argument("--output", default="vendor_incident_output.json", help="Output JSON path")
    args = parser.parse_args()

    run_vendor_intel(vendor=args.vendor.strip(), product=(args.product.strip() if args.product else None), output_path=args.output)


if __name__ == "__main__":
    main()
