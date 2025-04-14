variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
}

variable "node_count" {
  description = "Number of nodes in the GKE cluster"
  type        = number
  default     = 3
}

variable "node_machine_type" {
  description = "Machine type for nodes"
  type        = string
  default     = "n1-standard-4"
}

resource "google_container_cluster" "primary" {
  name               = var.cluster_name
  location           = var.region
  min_master_version = var.kubernetes_version
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Enable network policy 
  network_policy {
    enabled = true
  }
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "${var.cluster_name}-node-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = var.node_count
  
  node_config {
    machine_type = var.node_machine_type
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
    ]
    
    # Configure Kubernetes labels
    labels = {
      env = "prod"
    }
    
    # Enable workload identity on the node pool
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Add GPU if needed
    # guest_accelerator {
    #   type  = "nvidia-tesla-t4"
    #   count = 1
    # }
  }
}

# Get cluster credentials
data "google_client_config" "default" {}

output "kubeconfig" {
  value = {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  }
  sensitive = true
}

output "endpoint" {
  value     = "https://${google_container_cluster.primary.endpoint}"
  sensitive = true
}

output "ca_certificate" {
  value     = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive = true
}

output "token" {
  value     = data.google_client_config.default.access_token
  sensitive = true
}

output "cluster_name" {
  value = google_container_cluster.primary.name
} 