import os
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


load_dotenv(Path(__file__).with_name(".env"))


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value.strip()
    raise RuntimeError(
        f"Missing required environment variable: {name}. "
        f"Set it in the shell or in {Path(__file__).with_name('.env')}"
    )


AZURE_OPENAI_ENDPOINT = get_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_OPENAI_API_KEY = get_required_env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = get_required_env("AZURE_OPENAI_GPT_DEPLOYMENT")

AZURE_OPENAI_TIMEOUT_SECONDS = float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "45"))
AZURE_OPENAI_MAX_RETRIES = int(os.getenv("AZURE_OPENAI_MAX_RETRIES", "2"))

client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY,
    timeout=httpx.Timeout(AZURE_OPENAI_TIMEOUT_SECONDS, connect=10.0),
    max_retries=0,
)

APPROVED_DOMAINS = [
    "bleepingcomputer.com",
    "securityweek.com",
    "msrc.microsoft.com",
    "microsoft.com",
    "okta.com",
]

BLOCKED_DOMAINS = [
    "wikipedia.org",
    "reddit.com",
    "medium.com",
    "quora.com",
]


def normalize_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def is_allowed_url(url: str) -> bool:
    domain = normalize_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in APPROVED_DOMAINS)


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return json.loads(text)


def discover_urls_with_azure_web_search(vendor: str, product: str | None = None) -> list[dict[str, str]]:
    entity = f"{vendor} {product}".strip() if product else vendor

    prompt = f"""
Find at most 5 candidate public source URLs for {entity} security incidents or vulnerabilities.

Return JSON only:
{{
  "sources": [
    {{
      "url": "https://example.com/path",
      "title": "string",
      "relevance_note": "short reason"
    }}
  ]
}}

Rules:
- Return at most 5 sources.
- Do not summarize.
- Do not make a risk decision.
""".strip()
    last_error: Exception | None = None

    for attempt in range(1, AZURE_OPENAI_MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=AZURE_OPENAI_MODEL,
                reasoning={"effort": "low"},
                tools=[
                    {
                        "type": "web_search",
                        "filters": {
                            "allowed_domains": APPROVED_DOMAINS,
                            "blocked_domains": BLOCKED_DOMAINS,
                        },
                    }
                ],
                tool_choice="auto",
                input=prompt,
            )

            parsed = extract_json_object(response.output_text)
            raw_sources = parsed.get("sources", [])

            sources: list[dict[str, str]] = []
            seen_urls = set()

            for item in raw_sources:
                url = str(item.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue

                if not is_allowed_url(url):
                    continue

                seen_urls.add(url)
                sources.append(
                    {
                        "url": url,
                        "title": str(item.get("title", "")).strip(),
                        "snippet": str(item.get("relevance_note", "")).strip(),
                    }
                )

            return sources

        except (APITimeoutError, APIConnectionError, APIStatusError, RateLimitError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == AZURE_OPENAI_MAX_RETRIES:
                break
            time.sleep(min(3 * attempt, 10))

    raise RuntimeError(
        f"Azure OpenAI web_search failed after {AZURE_OPENAI_MAX_RETRIES} attempt(s). "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )


if __name__ == "__main__":
    try:
        results = discover_urls_with_azure_web_search("Okta")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as exc:
        raise SystemExit(str(exc)) from exc