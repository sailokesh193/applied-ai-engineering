variable "subscription_id" {
  type = string
}

variable "name" {
  type    = string
  default = "claimlens"
}

variable "location" {
  type    = string
  default = "eastus2"
}

variable "openai_location" {
  type        = string
  default     = "eastus2"
  description = "Region for Azure OpenAI; must offer the chosen models"
}

variable "search_sku" {
  type        = string
  default     = "basic"
  description = "basic or higher is required for the semantic ranker"
}

variable "embed_model" {
  type    = string
  default = "text-embedding-3-large"
}

variable "embed_model_version" {
  type    = string
  default = "1"
}

variable "embed_capacity" {
  type    = number
  default = 50
}

variable "chat_model" {
  type    = string
  default = "gpt-4.1"
}

variable "chat_model_version" {
  type        = string
  default     = "2025-04-14"
  description = "Check model availability and versions in the Foundry portal for your Region"
}

variable "chat_capacity" {
  type    = number
  default = 50
}

variable "retrieval_mode" {
  type    = string
  default = "pipeline"
}

variable "container_image" {
  type        = string
  default     = ""
  description = "Full image reference in the ACR (e.g. acrxyz.azurecr.io/claimlens:latest). Empty = placeholder image for the first apply."
}
