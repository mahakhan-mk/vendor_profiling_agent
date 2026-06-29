import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    RateLimitError,
)


load_dotenv(Path(__file__).with_name(".env"))


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    raise RuntimeError(
        f"Missing required environment variable: {name}. "
        f"Set it in {Path(__file__).with_name('.env')}"
    )


AZURE_OPENAI_ENDPOINT = get_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_OPENAI_API_KEY = get_required_env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = get_required_env("AZURE_OPENAI_GPT_DEPLOYMENT")
AZURE_OPENAI_TIMEOUT_SECONDS = float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "60"))
AZURE_OPENAI_MAX_RETRIES = int(os.getenv("AZURE_OPENAI_MAX_RETRIES", "1"))

client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY,
    timeout=httpx.Timeout(AZURE_OPENAI_TIMEOUT_SECONDS, connect=10.0),
    max_retries=0,
)

APPROVED_DOMAINS = [
    "bleepingcomputer.com",
    "securityweek.com",
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


def test_web_search() -> list[dict[str, str]]:
    prompt = """
Search the web for one public source about Okta security incidents.

Return JSON only with this exact schema:
{
  "sources": [
    {
      "url": "https://example.com/path",
      "title": "string",
      "relevance_note": "short reason"
    }
  ]
}

Rules:
- Return at most 1 source.
- Prefer bleepingcomputer.com, securityweek.com, or okta.com.
- Do not summarize the article.
- Do not make a risk decision.
""".strip()

    try:
        response = client.responses.create(
            model=AZURE_OPENAI_MODEL,
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

    except BadRequestError as exc:
        raise RuntimeError(
            "Bad request. Most likely the deployment does not support Responses API web_search, "
            "the deployment name is wrong, or the web_search tool is not enabled for this resource. "
            f"Details: {exc}"
        ) from exc

    except RateLimitError as exc:
        raise RuntimeError(
            "Rate limit exceeded. The deployment is valid, but quota/capacity is not available. "
            "Use a smaller deployment, another region, or request quota increase. "
            f"Details: {exc}"
        ) from exc

    except (APITimeoutError, APIConnectionError, APIStatusError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Web search test failed: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    results = test_web_search()
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if not results:
        raise SystemExit(
            "The call succeeded, but no allowed-domain URLs were returned. "
            "This means web_search is reachable, but the domain filter/query returned no usable source."
        )