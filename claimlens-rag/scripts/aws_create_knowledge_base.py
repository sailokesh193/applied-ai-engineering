"""Create a Bedrock Knowledge Base + S3 data source for RETRIEVAL_MODE=managed.

Two flavours:
  --type managed    Bedrock Managed Knowledge Base (AWS runs the vector store,
                    hybrid search and reranking by default)
  --type s3vectors  Customer-managed knowledge base on Amazon S3 Vectors
                    (cheapest storage, semantic search only)

The service role comes from the Terraform output `kb_role_arn`.

    python scripts/aws_create_knowledge_base.py --type managed \
        --role-arn arn:aws:iam::123456789012:role/claimlens-kb-role \
        --bucket my-claimlens-docs --prefix claimlens/raw/
"""

from __future__ import annotations

import argparse
import json
import time

import boto3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["managed", "s3vectors"], default="managed")
    ap.add_argument("--name", default="claimlens-kb")
    ap.add_argument("--role-arn", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="claimlens/raw/")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--embed-model", default="amazon.titan-embed-text-v2:0")
    ap.add_argument("--dimensions", type=int, default=1024)
    args = ap.parse_args()

    agent = boto3.client("bedrock-agent", region_name=args.region)
    embed_arn = f"arn:aws:bedrock:{args.region}::foundation-model/{args.embed_model}"

    if args.type == "managed":
        kb = agent.create_knowledge_base(
            name=args.name,
            roleArn=args.role_arn,
            knowledgeBaseConfiguration={
                "type": "MANAGED",
                "managedKnowledgeBaseConfiguration": {"embeddingModelType": "MANAGED"},
            },
        )["knowledgeBase"]
    else:
        s3v = boto3.client("s3vectors", region_name=args.region)
        bucket_name = f"{args.name}-vectors"
        bucket_arn = s3v.create_vector_bucket(vectorBucketName=bucket_name)["vectorBucketArn"]
        s3v.create_index(
            vectorBucketName=bucket_name,
            indexName=args.name,
            dataType="float32",
            dimension=args.dimensions,
            distanceMetric="cosine",
            # keep Bedrock's large text/metadata fields out of the 2 KB filterable budget
            metadataConfiguration={"nonFilterableMetadataKeys": ["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]},
        )
        index_arn = s3v.get_index(vectorBucketName=bucket_name, indexName=args.name)["index"]["indexArn"]
        kb = agent.create_knowledge_base(
            name=args.name,
            roleArn=args.role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": embed_arn,
                    "embeddingModelConfiguration": {"bedrockEmbeddingModelConfiguration": {"dimensions": args.dimensions}},
                },
            },
            storageConfiguration={
                "type": "S3_VECTORS",
                "s3VectorsConfiguration": {"vectorBucketArn": bucket_arn, "indexArn": index_arn},
            },
        )["knowledgeBase"]

    kb_id = kb["knowledgeBaseId"]
    while agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]["status"] == "CREATING":
        time.sleep(5)

    ds = agent.create_data_source(
        knowledgeBaseId=kb_id,
        name=f"{args.name}-s3",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {"bucketArn": f"arn:aws:s3:::{args.bucket}", "inclusionPrefixes": [args.prefix]},
        },
    )["dataSource"]

    print(
        json.dumps(
            {
                "BEDROCK_KB_ID": kb_id,
                "BEDROCK_KB_DATA_SOURCE_ID": ds["dataSourceId"],
                "BEDROCK_KB_MANAGED": str(args.type == "managed").lower(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
