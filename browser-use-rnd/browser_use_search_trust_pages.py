import asyncio
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from browser_use import Agent, BrowserProfile, ChatAzureOpenAI


SourceType = Literal[
    "trust_center",
    "security",
    "privacy",
    "compliance",
    "certification",
    "status",
    "data_residency",
    "subprocessor",
    "vulnerability_disclosure",
    "insufficient_evidence",
]


class CandidateSource(BaseModel):
    source_type: SourceType
    page_title: str | None = None
    source_url: str
    reason_selected: str
    confidence: Literal["low", "medium", "high"]


class TrustPageSearchResult(BaseModel):
    vendor_name: str
    search_query: str
    candidate_sources: list[CandidateSource] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    requires_analyst_review: bool = True
    llm_hit_count: int = 0


class LlmHitCounter:
    def __init__(self) -> None:
        self.count = 0

    def increment(self, method_name: str) -> None:
        self.count += 1
        print(f"[LLM HIT {self.count}] {method_name}")


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def patch_llm_hit_counter(llm: Any, counter: LlmHitCounter) -> None:
    """
    Monkey-patches common LLM call methods so this R&D script can count Azure LLM hits.

    Browser Use internals may call different methods across versions, so this patches
    every common sync/async method if it exists on the ChatAzureOpenAI object.
    """
    method_names = [
        "invoke",
        "ainvoke",
        "complete",
        "acomplete",
        "chat",
        "achat",
        "call",
        "acall",
        "__call__",
    ]

    for method_name in method_names:
        if not hasattr(llm, method_name):
            continue

        original = getattr(llm, method_name)

        if not callable(original):
            continue

        if inspect.iscoroutinefunction(original):
            async def async_wrapper(*args: Any, __original=original, __name=method_name, **kwargs: Any) -> Any:
                counter.increment(__name)
                return await __original(*args, **kwargs)

            setattr(llm, method_name, async_wrapper)
        else:
            def sync_wrapper(*args: Any, __original=original, __name=method_name, **kwargs: Any) -> Any:
                counter.increment(__name)
                return __original(*args, **kwargs)

            setattr(llm, method_name, sync_wrapper)


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


def build_failure_result(
    vendor_name: str,
    search_query: str,
    error: str,
    llm_hit_count: int,
) -> TrustPageSearchResult:
    return TrustPageSearchResult(
        vendor_name=vendor_name,
        search_query=search_query,
        candidate_sources=[
            CandidateSource(
                source_type="insufficient_evidence",
                page_title=None,
                source_url="",
                reason_selected=f"Search failed: {error[:500]}",
                confidence="low",
            )
        ],
        limitations=[
            "Browser Use search failed or returned invalid JSON.",
            error[:1000],
        ],
        requires_analyst_review=True,
        llm_hit_count=llm_hit_count,
    )


def parse_allowed_domains() -> list[str] | None:
    """
    Optional env var:
    ALLOWED_DOMAINS=bing.com,www.bing.com,microsoft.com,www.microsoft.com,learn.microsoft.com,trust.microsoft.com

    For dynamic vendor discovery, leave it empty during R&D.
    For controlled testing, set it explicitly.
    """
    raw = os.getenv("ALLOWED_DOMAINS", "").strip()
    if not raw:
        return None

    domains = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return domains or None


async def run_search_agent() -> TrustPageSearchResult:
    load_dotenv()

    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    os.environ.setdefault("TIMEOUT_BrowserStartEvent", "120")
    os.environ.setdefault("TIMEOUT_BrowserLaunchEvent", "120")
    os.environ.setdefault("TIMEOUT_BrowserConnectedEvent", "120")
    os.environ.setdefault("TIMEOUT_BrowserStateRequestEvent", "120")
    os.environ.setdefault("TIMEOUT_ScreenshotEvent", "60")

    endpoint = required_env("AZURE_OPENAI_ENDPOINT")
    api_key = required_env("AZURE_OPENAI_API_KEY")
    deployment = required_env("AZURE_OPENAI_DEPLOYMENT")
    api_version = required_env("AZURE_OPENAI_API_VERSION")

    vendor_name = (
        sys.argv[1].strip()
        if len(sys.argv) > 1
        else os.getenv("VENDOR_NAME", "Microsoft 365 Copilot").strip()
    )

    max_steps = int(os.getenv("BROWSER_USE_MAX_STEPS", "8"))

    search_query = (
        f"{vendor_name} official trust center security privacy compliance certifications"
    )
    search_url = f"https://www.bing.com/search?q={quote_plus(search_query)}"

    counter = LlmHitCounter()

    llm = ChatAzureOpenAI(
        model=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    patch_llm_hit_counter(llm, counter)

    allowed_domains = parse_allowed_domains()

    browser_profile_kwargs: dict[str, Any] = {
        "enable_default_extensions": False,
        "headless": False,
    }

    if allowed_domains:
        browser_profile_kwargs["allowed_domains"] = allowed_domains

    browser_profile = BrowserProfile(**browser_profile_kwargs)

    task = f"""
Open this Bing search URL:
{search_url}

Vendor:
{vendor_name}

Goal:
Find official public trust/security/privacy/compliance/certification URLs for the vendor.

Rules:
- This is URL discovery only.
- Do not extract final evidence from pages.
- Do not make a vendor risk decision.
- Do not return ads.
- Do not return third-party review sites, Reddit, forums, resellers, marketplaces, or news articles.
- Prefer official vendor-owned domains.
- Prefer pages that look like trust center, security, privacy, compliance, certification, status, data residency, subprocessor, or vulnerability disclosure pages.
- If official ownership is unclear, include it only with low confidence and explain the uncertainty.
- Return JSON only. No markdown. No prose outside JSON.

Required JSON schema:
{{
  "vendor_name": "{vendor_name}",
  "search_query": "{search_query}",
  "candidate_sources": [
    {{
      "source_type": "trust_center",
      "page_title": "string or null",
      "source_url": "https://...",
      "reason_selected": "short reason",
      "confidence": "low"
    }}
  ],
  "limitations": [
    "short limitation"
  ],
  "requires_analyst_review": true
}}

Finish as soon as you have enough official candidate URLs.
"""

    agent = Agent(
        task=task,
        llm=llm,
        browser_profile=browser_profile,
    )

    try:
        history = await agent.run(max_steps=max_steps)

        if hasattr(history, "final_result"):
            raw = history.final_result()
        else:
            raw = str(history)

        data = extract_json(raw)
        result = TrustPageSearchResult.model_validate(data)
        result.requires_analyst_review = True
        result.llm_hit_count = counter.count

        if not result.candidate_sources:
            result.candidate_sources.append(
                CandidateSource(
                    source_type="insufficient_evidence",
                    page_title=None,
                    source_url="",
                    reason_selected="No candidate trust/security/compliance URLs were found.",
                    confidence="low",
                )
            )
            result.limitations.append("No candidate URLs were returned by the agent.")

        return result

    except (json.JSONDecodeError, ValidationError, ValueError, RuntimeError, Exception) as exc:
        return build_failure_result(
            vendor_name=vendor_name,
            search_query=search_query,
            error=f"{type(exc).__name__}: {exc}",
            llm_hit_count=counter.count,
        )


def save_result(result: TrustPageSearchResult) -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    safe_vendor = re.sub(r"[^a-zA-Z0-9_-]+", "_", result.vendor_name).strip("_").lower()
    output_path = output_dir / f"browser_use_search_{safe_vendor}.json"

    payload = result.model_dump(mode="json")
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nLLM hits counted: {result.llm_hit_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    final_result = asyncio.run(run_search_agent())
    save_result(final_result)