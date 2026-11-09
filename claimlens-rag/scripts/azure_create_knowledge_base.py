"""Create a Foundry IQ knowledge source + knowledge base over the pipeline's search index.

This puts Azure's managed agentic retrieval on top of the SAME index the own
pipeline builds, so RETRIEVAL_MODE=pipeline and RETRIEVAL_MODE=managed can be
compared on one eval set.

Uses the stable Azure AI Search REST API (2026-04-01). Auth: Entra ID token
(DefaultAzureCredential) or AZURE_SEARCH_API_KEY.

    python scripts/azure_create_knowledge_base.py \
        --endpoint https://<search>.search.windows.net \
        --index claimlens-chunks --source claimlens-ks --kb claimlens-kb
"""

from __future__ import annotations

import argparse
import json
import os

import httpx

API_VERSION = "2026-04-01"


def headers() -> dict[str, str]:
    key = os.environ.get("AZURE_SEARCH_API_KEY")
    if key:
        return {"api-key": key, "Content-Type": "application/json"}
    from azure.identity import DefaultAzureCredential

    token = DefaultAzureCredential().get_token("https://search.azure.com/.default").token
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("AZURE_SEARCH_ENDPOINT"), required=not os.environ.get("AZURE_SEARCH_ENDPOINT"))
    ap.add_argument("--index", default=os.environ.get("AZURE_SEARCH_INDEX", "claimlens-chunks"))
    ap.add_argument("--source", default=os.environ.get("AZURE_KB_SOURCE_NAME", "claimlens-ks"))
    ap.add_argument("--kb", default=os.environ.get("AZURE_KB_NAME", "claimlens-kb"))
    args = ap.parse_args()
    base = args.endpoint.rstrip("/")
    h = headers()

    source = {
        "name": args.source,
        "kind": "searchIndex",
        "description": "ClaimLens chunks (medical records and payer policies) built by the own pipeline.",
        "searchIndexParameters": {
            "searchIndexName": args.index,
            "sourceDataFields": [{"name": f} for f in ["chunk_id", "doc_id", "source", "section", "page", "content"]],
        },
    }
    r = httpx.put(f"{base}/knowledgesources('{args.source}')", params={"api-version": API_VERSION}, headers=h, json=source, timeout=60)
    print("knowledge source:", r.status_code, r.text[:300])
    r.raise_for_status()

    kb = {
        "name": args.kb,
        "description": "ClaimLens claims review knowledge base.",
        "knowledgeSources": [{"name": args.source}],
    }
    r = httpx.put(f"{base}/knowledgebases('{args.kb}')", params={"api-version": API_VERSION}, headers=h, json=kb, timeout=60)
    print("knowledge base:", r.status_code, r.text[:300])
    r.raise_for_status()
    print(json.dumps({"AZURE_KB_NAME": args.kb, "AZURE_KB_SOURCE_NAME": args.source}, indent=2))


if __name__ == "__main__":
    main()
