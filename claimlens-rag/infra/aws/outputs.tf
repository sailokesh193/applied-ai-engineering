output "api_url" {
  value = "http://${aws_lb.api.dns_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "docs_bucket" {
  value = aws_s3_bucket.docs.bucket
}

output "aoss_endpoint" {
  value = aws_opensearchserverless_collection.vectors.collection_endpoint
}

output "kb_role_arn" {
  value = aws_iam_role.kb.arn
}
