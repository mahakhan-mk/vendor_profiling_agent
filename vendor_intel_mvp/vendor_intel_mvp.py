import json
import os
from typing import Any

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()

FIRECRAWL_BASE_URL = os.environ["FIRECRAWL_BASE_URL"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_MODEL = os.environ["AZURE_OPENAI_GPT_DEPLOYMENT"]
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

TEST_VENDOR = "Microsoft"
TEST_URL = "https://www.bleepingcomputer.com/news/security/github-disables-microsoft-repos-pushing-password-stealing-malware/"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


def firecrawl_scrape(url: str) -> dict[str, Any]:
    response = requests.post(
        f"{FIRECRAWL_BASE_URL}/v2/scrape",
        json={"url": url, "formats": ["markdown"]},
        timeout=120,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Scrape HTTP {response.status_code}: {response.text[:500]}")

    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Scrape failed: {payload}")

    data = payload.get("data", {})
    metadata = data.get("metadata", {})
    markdown = data.get("markdown") or ""

    if len(markdown) < 300:
        raise RuntimeError("Scrape returned too little markdown.")

    return {
        "requested_url": url,
        "resolved_url": metadata.get("url") or metadata.get("sourceURL"),
        "title": metadata.get("title"),
        "status_code": metadata.get("statusCode"),
        "markdown": markdown,
    }


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return json.loads(text)


def extract_incident(vendor: str, page: dict[str, Any]) -> dict[str, Any]:
    compact_page = {
        "url": page["resolved_url"] or page["requested_url"],
        "title": page["title"],
        "markdown": page["markdown"][:3000],
    }

    prompt = f"""
Extract cybersecurity incident evidence for {vendor} from this single page.

Return only valid JSON:
{{
  "vendor_name": "string",
  "source_url": "string",
  "is_relevant": true,
  "security_incidents": [
    {{
      "title": "string",
      "date": "string or null",
      "incident_type": "breach | vulnerability | advisory | outage | unknown",
      "evidence": "string"
    }}
  ],
  "limitations": ["string"]
}}

Rules:
- Only include incidents explicitly involving {vendor}.
- Do not infer.
- If the page is not actually about {vendor}, set is_relevant=false and security_incidents=[].
- No final risk decision.

Page:
{json.dumps(compact_page, indent=2)}
"""

    print(f"Prompt chars sent to Azure: {len(prompt):,}")

    response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. No markdown fences.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    text = response.choices[0].message.content or ""
    return parse_json_response(text)


def main() -> None:
    print(f"Scraping: {TEST_URL}")
    page = firecrawl_scrape(TEST_URL)

    print(f"Scraped title: {page['title']}")
    print(f"Scraped markdown chars: {len(page['markdown']):,}")

    extracted = extract_incident(TEST_VENDOR, page)

    output = {
        "vendor": TEST_VENDOR,
        "scraped_page": {
            "requested_url": page["requested_url"],
            "resolved_url": page["resolved_url"],
            "title": page["title"],
            "markdown_length": len(page["markdown"]),
        },
        "extracted": extracted,
    }

    print(json.dumps(output, indent=2))

    with open("single_url_incident_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()