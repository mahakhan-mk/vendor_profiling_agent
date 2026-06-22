import json
import os
from dotenv import load_dotenv
import requests

load_dotenv()

FIRECRAWL_BASE_URL = os.getenv("FIRECRAWL_BASE_URL", "http://localhost:3002")

queries = [
    "Microsoft breach",
    "Microsoft security incident",
    "Microsoft data breach",
    "Microsoft vulnerability",
    "site:bleepingcomputer.com Microsoft breach",
    'site:bleepingcomputer.com "Microsoft" breach',
    "Okta breach",
    "Okta security incident",
    "Snowflake breach",
    "CrowdStrike security incident",
]

for query in queries:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")

    try:
        response = requests.post(
            f"{FIRECRAWL_BASE_URL}/v2/search",
            json={"query": query, "limit": 10},
            timeout=60,
        )
        print(f"HTTP: {response.status_code}")

        payload = response.json()
        print(f"SUCCESS: {payload.get('success')}")
        print(f"CREDITS: {payload.get('creditsUsed')}")

        results = payload.get("data", {}).get("web", [])
        print(f"RESULT COUNT: {len(results)}")

        for i, result in enumerate(results, start=1):
            print(f"\n[{i}] {result.get('title')}")
            print(result.get("url"))
            print(result.get("description"))

    except Exception as exc:
        print(f"FAILED: {exc}")