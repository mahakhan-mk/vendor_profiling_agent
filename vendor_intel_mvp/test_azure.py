import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
)

r = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_GPT_DEPLOYMENT"],
    messages=[{"role": "user", "content": "Return only: OK"}],
)

print(r.choices[0].message.content)