# Generalized Vendor Intelligence MVP

This code searches only approved external cybersecurity sources, retrieves candidate pages through Firecrawl, extracts compact evidence snippets, and asks Azure OpenAI to produce analyst-review JSON.

## Environment

Create `.env` in the project root:

```env
FIRECRAWL_BASE_URL=http://localhost:3002
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_GPT_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python vendor_intel.py --vendor Okta --output okta_intel.json
python vendor_intel.py --vendor Snowflake --output snowflake_intel.json
python vendor_intel.py --vendor Microsoft --product "Microsoft 365 Copilot" --output copilot_intel.json
```

## Approved external sources

- bleepingcomputer.com
- securityweek.com
- thehackernews.com
- techcrunch.com
- arstechnica.com
- theregister.com
- cybersecuritynews.com
- cisa.gov
- nvd.nist.gov

The script does not make final approval/rejection decisions. It returns evidence, citations, limitations, retrieval failures, and `requires_analyst_review=true`.
