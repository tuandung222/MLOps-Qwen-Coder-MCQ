variable "namespace" {
  description = "Kubernetes namespace"
  type        = string
  default     = "qwen-api"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "qwen-coder-api"
}

variable "app_image" {
  description = "Docker image for the application"
  type        = string
}

variable "replicas" {
  description = "Number of replicas"
  type        = number
  default     = 3
}

variable "model_path" {
  description = "Path to the model"
  type        = string
  default     = "tuandunghcmut/Qwen25_Coder_MultipleChoice_v4"
}

variable "enable_monitoring" {
  description = "Enable Prometheus and Grafana"
  type        = bool
  default     = true
}

variable "enable_tracing" {
  description = "Enable Jaeger tracing"
  type        = bool
  default     = true
}

# Create namespace
resource "kubernetes_namespace" "qwen_api" {
  metadata {
    name = var.namespace
    labels = {
      name = var.namespace
    }
  }
}

# Create Kubernetes deployment
resource "kubernetes_deployment" "qwen_api" {
  metadata {
    name      = var.app_name
    namespace = kubernetes_namespace.qwen_api.metadata[0].name
    labels = {
      app = var.app_name
    }
  }
  
  spec {
    replicas = var.replicas
    
    selector {
      match_labels = {
        app = var.app_name
      }
    }
    
    template {
      metadata {
        labels = {
          app = var.app_name
        }
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/path"   = "/metrics"
          "prometheus.io/port"   = "8000"
        }
      }
      
      spec {
        container {
          name  = var.app_name
          image = var.app_image
          
          resources {
            limits = {
              cpu    = "2"
              memory = "8Gi"
              # Uncomment for GPU
              # "nvidia.com/gpu" = "1"
            }
            requests = {
              cpu    = "1"
              memory = "4Gi"
            }
          }
          
          port {
            container_port = 8000
          }
          
          env {
            name  = "MODEL_PATH"
            value = var.model_path
          }
          
          env {
            name  = "DEVICE"
            value = "cuda:0"
          }
          
          env {
            name  = "MAX_LENGTH"
            value = "2048"
          }
          
          env {
            name  = "JAEGER_HOST"
            value = "jaeger"
          }
          
          env {
            name  = "JAEGER_PORT"
            value = "6831"
          }
          
          readiness_probe {
            http_get {
              path = "/api/v1/health"
              port = 8000
            }
            initial_delay_seconds = 60
            period_seconds        = 10
          }
          
          liveness_probe {
            http_get {
              path = "/api/v1/health"
              port = 8000
            }
            initial_delay_seconds = 60
            period_seconds        = 30
          }
        }
      }
    }
  }
}

# Create Kubernetes service
resource "kubernetes_service" "qwen_api" {
  metadata {
    name      = var.app_name
    namespace = kubernetes_namespace.qwen_api.metadata[0].name
  }
  
  spec {
    selector = {
      app = var.app_name
    }
    
    port {
      port        = 80
      target_port = 8000
    }
    
    type = "ClusterIP"
  }
}

# Create Kubernetes ingress
resource "kubernetes_ingress_v1" "qwen_api" {
  metadata {
    name      = "${var.app_name}-ingress"
    namespace = kubernetes_namespace.qwen_api.metadata[0].name
    annotations = {
      "kubernetes.io/ingress.class" = "nginx"
      "nginx.ingress.kubernetes.io/ssl-redirect" = "false"
    }
  }
  
  spec {
    rule {
      http {
        path {
          path = "/api"
          path_type = "Prefix"
          backend {
            service {
              name = kubernetes_service.qwen_api.metadata[0].name
              port {
                number = 80
              }
            }
          }
        }
      }
    }
  }
}

# Install Prometheus and Grafana using Helm
resource "helm_release" "prometheus" {
  count      = var.enable_monitoring ? 1 : 0
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "prometheus"
  namespace  = kubernetes_namespace.qwen_api.metadata[0].name
  
  set {
    name  = "server.persistentVolume.size"
    value = "10Gi"
  }
  
  set {
    name  = "alertmanager.persistentVolume.size"
    value = "2Gi"
  }
}

resource "helm_release" "grafana" {
  count      = var.enable_monitoring ? 1 : 0
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  namespace  = kubernetes_namespace.qwen_api.metadata[0].name
  
  set {
    name  = "persistence.enabled"
    value = "true"
  }
  
  set {
    name  = "persistence.size"
    value = "5Gi"
  }
  
  set {
    name  = "adminPassword"
    value = "admin"
  }
  
  # Configure Prometheus data source
  set {
    name  = "datasources.datasources\\.yaml.apiVersion"
    value = "1"
  }
  
  set {
    name  = "datasources.datasources\\.yaml.datasources[0].name"
    value = "Prometheus"
  }
  
  set {
    name  = "datasources.datasources\\.yaml.datasources[0].type"
    value = "prometheus"
  }
  
  set {
    name  = "datasources.datasources\\.yaml.datasources[0].url"
    value = "http://prometheus-server"
  }
  
  set {
    name  = "datasources.datasources\\.yaml.datasources[0].isDefault"
    value = "true"
  }
}

# Install Jaeger using Helm
resource "helm_release" "jaeger" {
  count      = var.enable_tracing ? 1 : 0
  name       = "jaeger"
  repository = "https://jaegertracing.github.io/helm-charts"
  chart      = "jaeger"
  namespace  = kubernetes_namespace.qwen_api.metadata[0].name
}

# Outputs for accessing services
output "api_endpoint" {
  value = "${kubernetes_service.qwen_api.metadata[0].name}.${kubernetes_namespace.qwen_api.metadata[0].name}.svc.cluster.local"
}

output "prometheus_endpoint" {
  value = var.enable_monitoring ? "prometheus-server.${kubernetes_namespace.qwen_api.metadata[0].name}.svc.cluster.local" : "monitoring-disabled"
}

output "grafana_endpoint" {
  value = var.enable_monitoring ? "grafana.${kubernetes_namespace.qwen_api.metadata[0].name}.svc.cluster.local" : "monitoring-disabled"
}

output "jaeger_endpoint" {
  value = var.enable_tracing ? "jaeger-query.${kubernetes_namespace.qwen_api.metadata[0].name}.svc.cluster.local" : "tracing-disabled"
} 