import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from browser_use import Agent, BrowserProfile, ChatAzureOpenAI


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


def get_allowed_domains(url: str) -> list[str]:
    host = urlparse(url).hostname
    if not host:
        return []

    host = host.lower()
    domains = {host}

    if host.startswith("www."):
        domains.add(host.removeprefix("www."))
    else:
        domains.add(f"www.{host}")

    return sorted(domains)


def extract_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Agent output did not contain a JSON object.")

    return json.loads(text[start : end + 1])


def build_failure_result(vendor_name: str, source_url: str, error: str) -> BrowserEvidenceResult:
    return BrowserEvidenceResult(
        vendor_name=vendor_name,
        source_url=source_url,
        page_title=None,
        evidence_items=[
            EvidenceItem(
                evidence_type="insufficient_evidence",
                claim="The browser agent failed to produce usable evidence from the approved URL.",
                supporting_text=error[:1000],
                source_url=source_url,
                confidence="low",
            )
        ],
        limitations=[
            "Local Browser Use R&D run failed or returned invalid JSON.",
            error[:1000],
        ],
        requires_analyst_review=True,
    )


async def run_agent() -> BrowserEvidenceResult:
    load_dotenv()

    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    os.environ.setdefault("TIMEOUT_BrowserStartEvent", "120")
    os.environ.setdefault("TIMEOUT_BrowserLaunchEvent", "120")
    os.environ.setdefault("TIMEOUT_BrowserConnectedEvent", "120")

    endpoint = required_env("AZURE_OPENAI_ENDPOINT")
    api_key = required_env("AZURE_OPENAI_API_KEY")
    deployment = required_env("AZURE_OPENAI_DEPLOYMENT")
    api_version = required_env("AZURE_OPENAI_API_VERSION")

    test_url = os.getenv("TEST_URL", "https://www.cloudflare.com/trust-hub/").strip()
    vendor_name = os.getenv("VENDOR_NAME", "Cloudflare").strip()
    allowed_domains = get_allowed_domains(test_url)

    if not allowed_domains:
        raise RuntimeError(f"Invalid TEST_URL: {test_url}")

    llm = ChatAzureOpenAI(
        model=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    task = f"""
Open this approved URL only:
{test_url}

Vendor name:
{vendor_name}

You are doing local R&D for evidence extraction only.

Rules:
- Do not browse outside the approved URL/domain.
- Do not make a final vendor risk decision.
- Do not say the vendor is approved, rejected, safe, unsafe, compliant, or non-compliant.
- Extract only observations that are directly supported by visible page text.
- If evidence is weak or missing, use evidence_type "insufficient_evidence".
- Return JSON only. No markdown. No prose outside JSON.

Allowed evidence_type values:
- trust_center
- certification
- security_posture
- privacy
- data_residency
- subprocessor
- availability
- breach
- vulnerability
- insufficient_evidence

Required JSON schema:
{{
  "vendor_name": "{vendor_name}",
  "source_url": "{test_url}",
  "page_title": "string or null",
  "evidence_items": [
    {{
      "evidence_type": "trust_center | certification | security_posture | privacy | data_residency | subprocessor | availability | breach | vulnerability | insufficient_evidence",
      "claim": "short factual observation only",
      "supporting_text": "exact or near-exact visible text from the page supporting the claim",
      "source_url": "{test_url}",
      "confidence": "low | medium | high"
    }}
  ],
  "limitations": [
    "limitations of this extraction"
  ],
  "requires_analyst_review": true
}}
"""

    browser_profile = BrowserProfile(
        allowed_domains=allowed_domains,
        enable_default_extensions=False,
        headless=False,
    )

    agent = Agent(
        task=task,
        llm=llm,
        browser_profile=browser_profile,
    )

    try:
        history = await agent.run(max_steps=25)

        if hasattr(history, "final_result"):
            raw = history.final_result()
        else:
            raw = str(history)

        data = extract_json(raw)
        result = BrowserEvidenceResult.model_validate(data)

        if not result.evidence_items:
            result.evidence_items.append(
                EvidenceItem(
                    evidence_type="insufficient_evidence",
                    claim="No SAR-relevant evidence was extracted from the approved URL.",
                    supporting_text="No relevant supporting text was returned by the browser agent.",
                    source_url=test_url,
                    confidence="low",
                )
            )
            result.limitations.append("No evidence items were returned by the browser agent.")

        result.requires_analyst_review = True
        return result

    except (json.JSONDecodeError, ValidationError, ValueError, RuntimeError, Exception) as exc:
        return build_failure_result(vendor_name, test_url, f"{type(exc).__name__}: {exc}")


def save_result(result: BrowserEvidenceResult) -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "browser_agent_result.json"
    payload = result.model_dump(mode="json")

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    final_result = asyncio.run(run_agent())
    save_result(final_result)