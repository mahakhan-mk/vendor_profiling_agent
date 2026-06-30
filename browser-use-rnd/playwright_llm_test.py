import asyncio
import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AzureOpenAI
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, ValidationError


EvidenceType = Literal[
    "trust_center",
    "certification",
    "security_posture",
    "privacy",
    "data_residency",
    "subprocessor",
    "availability",
    "breach",
    "vulnerability",
    "insufficient_evidence",
]


class EvidenceItem(BaseModel):
    evidence_type: EvidenceType
    claim: str
    supporting_text: str
    source_url: str
    confidence: Literal["low", "medium", "high"]


class BrowserEvidenceResult(BaseModel):
    vendor_name: str
    source_url: str
    page_title: str | None = None
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    requires_analyst_review: bool = True


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def validate_https_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError(f"Only HTTPS URLs are allowed. Got: {url}")
    if not parsed.hostname:
        raise RuntimeError(f"Invalid URL. Got: {url}")


def compact_page_text(text: str, max_chars: int = 30000) -> str:
    evidence_keywords = [
        "trust",
        "security",
        "secure",
        "compliance",
        "certification",
        "certificate",
        "iso",
        "soc",
        "pci",
        "privacy",
        "gdpr",
        "data protection",
        "data residency",
        "data localization",
        "subprocessor",
        "availability",
        "uptime",
        "status",
        "incident",
        "breach",
        "vulnerability",
        "encryption",
        "access management",
        "identity",
        "audit",
        "risk",
    ]

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    selected: list[str] = []
    seen: set[str] = set()

    for line in lines:
        normalized = " ".join(line.lower().split())

        if normalized in seen:
            continue

        if any(keyword in normalized for keyword in evidence_keywords):
            selected.append(line)
            seen.add(normalized)

    if not selected:
        selected = lines[:300]

    compact = "\n".join(selected)

    return compact[:max_chars]


def build_failure_result(vendor_name: str, source_url: str, error: str) -> BrowserEvidenceResult:
    return BrowserEvidenceResult(
        vendor_name=vendor_name,
        source_url=source_url,
        page_title=None,
        evidence_items=[
            EvidenceItem(
                evidence_type="insufficient_evidence",
                claim="The local page extraction or evidence extraction run failed.",
                supporting_text=error[:1000],
                source_url=source_url,
                confidence="low",
            )
        ],
        limitations=[error[:1000]],
        requires_analyst_review=True,
    )


async def fetch_page_text(url: str) -> tuple[str | None, str]:
    validate_https_url(url)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        page = await browser.new_page(
            viewport={"width": 1400, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
        )

        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)

        title = await page.title()

        try:
            text = await page.locator("body").inner_text(timeout=30000)
        except Exception:
            text = await page.content()

        await browser.close()

    return title, text


def extract_evidence_with_llm(
    vendor_name: str,
    source_url: str,
    page_title: str | None,
    page_text: str,
) -> BrowserEvidenceResult:
    client = AzureOpenAI(
        azure_endpoint=required_env("AZURE_OPENAI_ENDPOINT"),
        api_key=required_env("AZURE_OPENAI_API_KEY"),
        api_version=required_env("AZURE_OPENAI_API_VERSION"),
    )

    deployment = required_env("AZURE_OPENAI_DEPLOYMENT")
    compact_text = compact_page_text(page_text, max_chars=30000)

    prompt = f"""
You are extracting vendor web evidence from one approved page.

Vendor:
{vendor_name}

Source URL:
{source_url}

Page title:
{page_title}

Rules:
- Use only the provided page text.
- Do not browse.
- Do not infer facts that are not supported by the provided text.
- Do not make a final vendor risk decision.
- Do not say the vendor is approved, rejected, safe, unsafe, compliant, or non-compliant.
- Extract only factual observations supported by the page text.
- Return evidence_items only for visible evidence.
- If a category is not visible, mention the gap in limitations, not as many fake evidence_items.
- supporting_text must be exact or near-exact text from the page.
- Return valid JSON only.

Allowed evidence_type values:
trust_center, certification, security_posture, privacy, data_residency, subprocessor, availability, breach, vulnerability, insufficient_evidence

Required JSON shape:
{{
  "vendor_name": "{vendor_name}",
  "source_url": "{source_url}",
  "page_title": "{page_title}",
  "evidence_items": [
    {{
      "evidence_type": "trust_center",
      "claim": "short factual observation",
      "supporting_text": "exact or near-exact source text",
      "source_url": "{source_url}",
      "confidence": "low"
    }}
  ],
  "limitations": [
    "short limitation"
  ],
  "requires_analyst_review": true
}}

Page text:
{compact_text}
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": "Return JSON only. No markdown. No prose.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    if not raw:
        raise RuntimeError("Azure OpenAI returned an empty response.")

    data = json.loads(raw)
    result = BrowserEvidenceResult.model_validate(data)
    result.requires_analyst_review = True

    if not result.evidence_items:
        result.evidence_items.append(
            EvidenceItem(
                evidence_type="insufficient_evidence",
                claim="No SAR-relevant evidence was extracted from the approved URL.",
                supporting_text="No relevant evidence item was returned by the model.",
                source_url=source_url,
                confidence="low",
            )
        )
        result.limitations.append("No evidence items were returned by the model.")

    return result


async def main() -> None:
    load_dotenv()

    source_url = os.getenv("TEST_URL", "https://www.cloudflare.com/trust-hub/").strip()
    vendor_name = os.getenv("VENDOR_NAME", "Cloudflare").strip()

    try:
        page_title, raw_page_text = await fetch_page_text(source_url)

        if not raw_page_text.strip():
            result = BrowserEvidenceResult(
                vendor_name=vendor_name,
                source_url=source_url,
                page_title=page_title,
                evidence_items=[
                    EvidenceItem(
                        evidence_type="insufficient_evidence",
                        claim="No page text was extracted from the approved URL.",
                        supporting_text="The extracted page body text was empty.",
                        source_url=source_url,
                        confidence="low",
                    )
                ],
                limitations=["No visible page body text was extracted."],
                requires_analyst_review=True,
            )
        else:
            result = extract_evidence_with_llm(
                vendor_name=vendor_name,
                source_url=source_url,
                page_title=page_title,
                page_text=raw_page_text,
            )

    except (ValidationError, json.JSONDecodeError, RuntimeError, Exception) as exc:
        result = build_failure_result(
            vendor_name=vendor_name,
            source_url=source_url,
            error=f"{type(exc).__name__}: {exc}",
        )

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "playwright_llm_result.json"
    payload = result.model_dump(mode="json")

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())