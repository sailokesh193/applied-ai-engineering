variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  type    = string
  default = "claimlens"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "retrieval_mode" {
  type        = string
  default     = "pipeline"
  description = "pipeline (own LangGraph pipeline) or managed (Bedrock Knowledge Base)"
}

variable "bedrock_llm_model" {
  type        = string
  description = "Bedrock model ID or inference profile ID used for generation (check the Bedrock console for what is enabled in your Region)"
}

variable "bedrock_rerank_model_arn" {
  type        = string
  description = "Rerank model ARN, e.g. arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0"
}

variable "bedrock_kb_id" {
  type    = string
  default = ""
}

variable "bedrock_kb_managed" {
  type    = string
  default = "false"
}

variable "admin_principal_arns" {
  type        = list(string)
  default     = []
  description = "Extra IAM user/role ARNs allowed to read and write the OpenSearch collection (for ingestion from CI or a laptop)"
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "allowed_cidrs" {
  type        = list(string)
  default     = ["0.0.0.0/0"]
  description = "Who can reach the load balancer. Restrict this; add HTTPS + auth before real data."
}
