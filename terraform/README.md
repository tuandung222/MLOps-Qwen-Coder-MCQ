# Qwen-Coder-MCQ Terraform Deployment

This directory contains Terraform configurations to deploy the Qwen-Coder-MCQ platform on Kubernetes, supporting both Google Cloud Platform (GCP) and Amazon Web Services (AWS).

## Architecture

The Terraform configurations set up:

- A Kubernetes cluster (GKE on GCP or EKS on AWS)
- Kubernetes resources for the Qwen-Coder-MCQ application
- Monitoring stack (Prometheus and Grafana)
- Distributed tracing (Jaeger)

## Prerequisites

1. [Terraform](https://developer.hashicorp.com/terraform/install) (1.0.0+)
2. Cloud provider CLI:
   - For GCP: [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
   - For AWS: [AWS CLI](https://aws.amazon.com/cli/)
3. [kubectl](https://kubernetes.io/docs/tasks/tools/)
4. Proper cloud provider credentials:
   - For GCP: `gcloud auth login` and `gcloud auth application-default login`
   - For AWS: Configure via `aws configure`

## Configuration

Edit the `terraform.tfvars` file to customize your deployment:

```hcl
# Choose "gcp" or "aws" as your cloud provider
cloud_provider = "gcp"

# GCP settings (if using GCP)
gcp_project_id = "your-gcp-project-id"
region = "us-central1"

# AWS settings (if using AWS)
aws_region = "us-east-1"

# Kubernetes configuration
kubernetes_version = "1.25.0"
node_count = 3

# Application configuration
project_name = "qwen-coder-mcq"
```

## Deployment Steps

1. Initialize Terraform:
   ```bash
   terraform init
   ```

2. Validate your configuration:
   ```bash
   terraform validate
   ```

3. Plan your deployment:
   ```bash
   terraform plan
   ```

4. Apply the configuration:
   ```bash
   terraform apply
   ```

5. Access your deployment:
   ```bash
   # Get Kubernetes context
   terraform output -raw kubernetes_cluster_name
   
   # For GCP
   gcloud container clusters get-credentials $(terraform output -raw kubernetes_cluster_name) --region $(terraform output -raw region)
   
   # For AWS
   aws eks update-kubeconfig --name $(terraform output -raw kubernetes_cluster_name) --region $(terraform output -raw aws_region)
   
   # Get service endpoints
   terraform output api_endpoint
   terraform output grafana_endpoint
   terraform output prometheus_endpoint
   terraform output jaeger_endpoint
   ```

## Enabling GPU Support

To enable GPU support:

1. Uncomment the GPU-related settings in either:
   - `modules/gcp/main.tf` for GCP
   - `modules/aws/main.tf` for AWS

2. Uncomment the GPU resource limits in `modules/k8s/main.tf`

## Cleaning Up

To destroy all created resources:

```bash
terraform destroy
```

## Module Structure

- `main.tf` - Main Terraform configuration
- `terraform.tfvars` - Variable values
- `modules/` - Reusable modules
  - `gcp/` - Google Cloud Platform module
  - `aws/` - Amazon Web Services module
  - `k8s/` - Kubernetes resources module

## Customization

- To change the model, update the `model_path` variable
- To modify resource allocations, edit the resource limits in `modules/k8s/main.tf`
- To adjust monitoring, edit Prometheus and Grafana configurations in `modules/k8s/main.tf` 