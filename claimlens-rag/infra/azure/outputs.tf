output "api_url" {
  value = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}

output "search_endpoint" {
  value = local.search
}

output "openai_endpoint" {
  value = azurerm_cognitive_account.openai.endpoint
}

output "docintel_endpoint" {
  value = azurerm_cognitive_account.docintel.endpoint
}

output "storage_account_url" {
  value = azurerm_storage_account.docs.primary_blob_endpoint
}

output "resource_group" {
  value = azurerm_resource_group.rg.name
}
