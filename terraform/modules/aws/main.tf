variable "region" {
  description = "AWS region"
  type        = string
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
}

variable "node_count" {
  description = "Number of nodes in the EKS cluster"
  type        = number
  default     = 3
}

variable "instance_type" {
  description = "Instance type for nodes"
  type        = string
  default     = "t3.xlarge"
}

data "aws_availability_zones" "available" {}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 3)
}

# Create VPC for EKS
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"

  name                 = "${var.cluster_name}-vpc"
  cidr                 = "10.0.0.0/16"
  azs                  = local.azs
  private_subnets      = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets       = ["10.0.4.0/24", "10.0.5.0/24", "10.0.6.0/24"]
  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  # Tags required for EKS
  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"           = "1"
  }
}

# Create EKS cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 18.0"

  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Self-managed node groups
  self_managed_node_group_defaults = {
    instance_type                          = var.instance_type
    update_launch_template_default_version = true
  }

  self_managed_node_groups = {
    worker_group = {
      min_size     = var.node_count
      max_size     = var.node_count
      desired_size = var.node_count

      instance_type = var.instance_type

      # Add GPU if needed
      # ami_id        = "ami-0123456789abcdef0" # Deep Learning AMI with GPU support
      # instance_type = "g4dn.xlarge"           # GPU instance
    }
  }

  # Ensure IAM Role for Service Accounts (IRSA)
  enable_irsa = true

  # Extend node-to-node security group rules
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
    egress_all = {
      description      = "Node all egress"
      protocol         = "-1"
      from_port        = 0
      to_port          = 0
      type             = "egress"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = ["::/0"]
    }
  }
}

# Get authentication data for providing to Kubernetes provider
data "aws_eks_cluster" "cluster" {
  name = module.eks.cluster_id
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_id
}

output "kubeconfig" {
  value = {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = data.aws_eks_cluster.cluster.certificate_authority[0].data
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
  sensitive = true
}

output "endpoint" {
  value     = data.aws_eks_cluster.cluster.endpoint
  sensitive = true
}

output "ca_certificate" {
  value     = data.aws_eks_cluster.cluster.certificate_authority[0].data
  sensitive = true
}

output "token" {
  value     = data.aws_eks_cluster_auth.cluster.token
  sensitive = true
}

output "cluster_name" {
  value = module.eks.cluster_id
} 
