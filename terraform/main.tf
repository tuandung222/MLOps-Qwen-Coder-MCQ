terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.11"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.5"
    }
  }
  backend "s3" {
    # Optional: Uncomment and configure for AWS backend
    # bucket = "my-terraform-state"
    # key    = "qwen-coder-mcq/terraform.tfstate"
    # region = "us-east-1"
  }
  # Alternative GCP backend
  # backend "gcs" {
  #   bucket = "my-terraform-state"
  #   prefix = "qwen-coder-mcq"
  # }
}

# Define variables for provider selection
variable "cloud_provider" {
  description = "Cloud provider to use (gcp or aws)"
  type        = string
  default     = "gcp"
  validation {
    condition     = contains(["gcp", "aws"], var.cloud_provider)
    error_message = "Valid values for cloud_provider are (gcp, aws)."
  }
}

variable "project_name" {
  description = "Project name for resources"
  type        = string
  default     = "qwen-coder-mcq"
}

variable "region" {
  description = "Region to deploy resources"
  type        = string
  default     = "us-central1" # GCP default
}

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = ""
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.25.0"
}

variable "node_count" {
  description = "Number of nodes in the Kubernetes cluster"
  type        = number
  default     = 3
}

variable "node_machine_type" {
  description = "Machine type for GCP nodes"
  type        = string
  default     = "n1-standard-4"
}

variable "aws_instance_type" {
  description = "Instance type for AWS nodes"
  type        = string
  default     = "t3.xlarge"
}

# Provider configurations
provider "google" {
  project = var.gcp_project_id
  region  = var.region
}

provider "aws" {
  region = var.aws_region
}

# Conditionally select the cloud provider
module "gcp_kubernetes" {
  source             = "./modules/gcp"
  count              = var.cloud_provider == "gcp" ? 1 : 0
  project_id         = var.gcp_project_id
  region             = var.region
  cluster_name       = "${var.project_name}-cluster"
  kubernetes_version = var.kubernetes_version
  node_count         = var.node_count
  node_machine_type  = var.node_machine_type
}

module "aws_kubernetes" {
  source             = "./modules/aws"
  count              = var.cloud_provider == "aws" ? 1 : 0
  region             = var.aws_region
  cluster_name       = "${var.project_name}-cluster"
  kubernetes_version = var.kubernetes_version
  node_count         = var.node_count
  instance_type      = var.aws_instance_type
}

# Configure Kubernetes provider based on selected cloud
locals {
  kubeconfig = var.cloud_provider == "gcp" ? module.gcp_kubernetes[0].kubeconfig : module.aws_kubernetes[0].kubeconfig
  cluster_endpoint = var.cloud_provider == "gcp" ? module.gcp_kubernetes[0].endpoint : module.aws_kubernetes[0].endpoint
  cluster_ca_certificate = var.cloud_provider == "gcp" ? module.gcp_kubernetes[0].ca_certificate : module.aws_kubernetes[0].ca_certificate
  cluster_token = var.cloud_provider == "gcp" ? module.gcp_kubernetes[0].token : module.aws_kubernetes[0].token
}

provider "kubernetes" {
  host                   = local.cluster_endpoint
  cluster_ca_certificate = base64decode(local.cluster_ca_certificate)
  token                  = local.cluster_token
}

provider "helm" {
  kubernetes {
    host                   = local.cluster_endpoint
    cluster_ca_certificate = base64decode(local.cluster_ca_certificate)
    token                  = local.cluster_token
  }
}

# Deploy Kubernetes resources
module "kubernetes_resources" {
  source = "./modules/k8s"
  depends_on = [
    module.gcp_kubernetes,
    module.aws_kubernetes,
  ]
  namespace          = "qwen-api"
  app_name           = "qwen-coder-api"
  app_image          = "your-docker-registry.com/qwen-coder-api:latest"
  replicas           = 3
  model_path         = "tuandunghcmut/Qwen25_Coder_MultipleChoice_v4"
  enable_monitoring  = true
  enable_tracing     = true
}

# Outputs
output "kubernetes_cluster_name" {
  value = var.cloud_provider == "gcp" ? module.gcp_kubernetes[0].cluster_name : module.aws_kubernetes[0].cluster_name
}

output "kubernetes_endpoint" {
  value = local.cluster_endpoint
  sensitive = true
}

output "api_endpoint" {
  value = module.kubernetes_resources.api_endpoint
}

output "grafana_endpoint" {
  value = module.kubernetes_resources.grafana_endpoint
}

output "prometheus_endpoint" {
  value = module.kubernetes_resources.prometheus_endpoint
}

output "jaeger_endpoint" {
  value = module.kubernetes_resources.jaeger_endpoint
} 