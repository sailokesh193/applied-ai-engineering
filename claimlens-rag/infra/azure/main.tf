terraform {
  required_version = ">= 1.6"
  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 4.0" }
    random  = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

provider "azurerm" {
  features {}
  subscription_id     = var.subscription_id
  storage_use_azuread = true # storage account has shared keys disabled
}

resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

locals {
  name   = "${var.name}${random_string.suffix.result}"
  tags   = { app = var.name }
  search = "https://${azurerm_search_service.search.name}.search.windows.net"
}

resource "azurerm_resource_group" "rg" {
  name     = "rg-${var.name}"
  location = var.location
  tags     = local.tags
}

resource "azurerm_user_assigned_identity" "app" {
  name                = "id-${var.name}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

# ------------------------------------------------------------------ Azure AI Search (+ semantic ranker)
resource "azurerm_search_service" "search" {
  name                          = "srch-${local.name}"
  resource_group_name           = azurerm_resource_group.rg.name
  location                      = azurerm_resource_group.rg.location
  sku                           = var.search_sku
  semantic_search_sku           = "free"
  local_authentication_enabled  = true
  authentication_failure_mode   = "http403" # accept both Entra ID (RBAC) and keys
  public_network_access_enabled = true
  tags                          = local.tags
}

# ------------------------------------------------------------------ Document Intelligence
resource "azurerm_cognitive_account" "docintel" {
  name                  = "di-${local.name}"
  resource_group_name   = azurerm_resource_group.rg.name
  location              = azurerm_resource_group.rg.location
  kind                  = "FormRecognizer"
  sku_name              = "S0"
  custom_subdomain_name = "di-${local.name}" # required for Entra ID auth
  tags                  = local.tags
}

# ------------------------------------------------------------------ Azure OpenAI (Microsoft Foundry models)
resource "azurerm_cognitive_account" "openai" {
  name                  = "aoai-${local.name}"
  resource_group_name   = azurerm_resource_group.rg.name
  location              = var.openai_location
  kind                  = "OpenAI"
  sku_name              = "S0"
  custom_subdomain_name = "aoai-${local.name}"
  tags                  = local.tags
}

resource "azurerm_cognitive_deployment" "embed" {
  name                 = var.embed_model
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = var.embed_model
    version = var.embed_model_version
  }
  sku {
    name     = "Standard"
    capacity = var.embed_capacity
  }
}

resource "azurerm_cognitive_deployment" "chat" {
  name                 = var.chat_model
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = var.chat_model
    version = var.chat_model_version
  }
  sku {
    name     = "GlobalStandard"
    capacity = var.chat_capacity
  }
  depends_on = [azurerm_cognitive_deployment.embed]
}

# ------------------------------------------------------------------ Blob storage (for a managed, Blob-backed knowledge source)
resource "azurerm_storage_account" "docs" {
  name                            = "st${local.name}"
  resource_group_name             = azurerm_resource_group.rg.name
  location                        = azurerm_resource_group.rg.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false
  tags                            = local.tags
}

resource "azurerm_storage_container" "raw" {
  name                  = "claimlens-raw"
  storage_account_id    = azurerm_storage_account.docs.id
  container_access_type = "private"
}

# ------------------------------------------------------------------ container registry + app
resource "azurerm_container_registry" "acr" {
  name                = "acr${local.name}"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = false
}

resource "azurerm_log_analytics_workspace" "logs" {
  name                = "log-${local.name}"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_app_environment" "env" {
  name                       = "cae-${var.name}"
  resource_group_name        = azurerm_resource_group.rg.name
  location                   = azurerm_resource_group.rg.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
}

resource "azurerm_container_app" "api" {
  name                         = "ca-${var.name}"
  resource_group_name          = azurerm_resource_group.rg.name
  container_app_environment_id = azurerm_container_app_environment.env.id
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.app.id]
  }

  dynamic "registry" {
    for_each = var.container_image == "" ? [] : [1]
    content {
      server   = azurerm_container_registry.acr.login_server
      identity = azurerm_user_assigned_identity.app.id
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 1
    max_replicas = 3
    container {
      name   = "api"
      image  = var.container_image == "" ? "mcr.microsoft.com/k8se/quickstart:latest" : var.container_image
      cpu    = 1.0
      memory = "2Gi"

      dynamic "env" {
        for_each = {
          CLOUD                         = "azure"
          RETRIEVAL_MODE                = var.retrieval_mode
          AZURE_CLIENT_ID               = azurerm_user_assigned_identity.app.client_id # lets DefaultAzureCredential pick this identity
          AZURE_DOCINTEL_ENDPOINT       = azurerm_cognitive_account.docintel.endpoint
          AZURE_OPENAI_ENDPOINT         = azurerm_cognitive_account.openai.endpoint
          AZURE_OPENAI_EMBED_DEPLOYMENT = azurerm_cognitive_deployment.embed.name
          AZURE_OPENAI_CHAT_DEPLOYMENT  = azurerm_cognitive_deployment.chat.name
          AZURE_SEARCH_ENDPOINT         = local.search
          AZURE_SEARCH_INDEX            = "claimlens-chunks"
          AZURE_KB_NAME                 = "claimlens-kb"
          AZURE_KB_SOURCE_NAME          = "claimlens-ks"
          AZURE_STORAGE_ACCOUNT_URL     = azurerm_storage_account.docs.primary_blob_endpoint
        }
        content {
          name  = env.key
          value = env.value
        }
      }
    }
  }

  depends_on = [azurerm_role_assignment.acr_pull]
}

# ------------------------------------------------------------------ RBAC for the app identity (no keys in the app)
locals {
  app_roles = {
    search_data    = { scope = azurerm_search_service.search.id, role = "Search Index Data Contributor" }
    search_service = { scope = azurerm_search_service.search.id, role = "Search Service Contributor" }
    openai         = { scope = azurerm_cognitive_account.openai.id, role = "Cognitive Services OpenAI User" }
    docintel       = { scope = azurerm_cognitive_account.docintel.id, role = "Cognitive Services User" }
    blob           = { scope = azurerm_storage_account.docs.id, role = "Storage Blob Data Contributor" }
  }
}

resource "azurerm_role_assignment" "app" {
  for_each             = local.app_roles
  scope                = each.value.scope
  role_definition_name = each.value.role
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.app.principal_id
}

# Same data-plane roles for whoever runs Terraform (to ingest from a laptop or CI)
data "azurerm_client_config" "me" {}

resource "azurerm_role_assignment" "operator" {
  for_each             = local.app_roles
  scope                = each.value.scope
  role_definition_name = each.value.role
  principal_id         = data.azurerm_client_config.me.object_id
}
