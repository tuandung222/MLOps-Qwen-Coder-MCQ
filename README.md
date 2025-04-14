# System Architecture Diagrams

## Overall System Architecture

```mermaid
graph TD
    subgraph Users
        Client["Web Client"]
    end

    subgraph K8S["Kubernetes Cluster (3 replicas)"]
        subgraph API["API Layer"]
            FastAPI["FastAPI Application"] --> Model["Qwen2.5-Coder-1.5B Model"]
        end
        subgraph Monitoring["Monitoring Stack"]
            Prometheus["Prometheus"]
            Grafana["Grafana Dashboards"]
            Jaeger["Jaeger Tracing"]
        end
    end

    subgraph CI_CD["CI/CD Pipeline"]
        Jenkins["Jenkins"]
    end

    subgraph Cloud["Cloud Services (Optional)"]
        CloudStorage["Cloud Storage"]
        CloudCompute["Compute Services"]
    end

    Client -- "API Requests" --> FastAPI
    FastAPI -- "Metrics/Traces" --> Prometheus
    Prometheus --> Grafana
    FastAPI -- "Distributed Tracing" --> Jaeger
    Jenkins -- "Deploy to" --> K8S
    CloudStorage -- "Backup/Assets" --> K8S
    CloudCompute -- "Additional Resources" --> K8S
```

## API Service Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Model as "Qwen2.5-Coder-1.5B Model"
    participant Jaeger
    participant Prometheus

    Client->>FastAPI: POST /api/v1/mcq/answer
    FastAPI->>Jaeger: Start tracing span
    FastAPI->>Model: Format prompt & generate answer
    Model-->>FastAPI: YAML-formatted response
    FastAPI->>Prometheus: Report metrics
    FastAPI->>Jaeger: End tracing span
    FastAPI-->>Client: Return MCQ answer with reasoning
```

## Monitoring Stack

```mermaid
graph LR
    subgraph API["API Application"]
        Metrics["API Metrics Endpoint"]
        OpenTelemetry["OpenTelemetry SDK"]
    end

    subgraph Monitoring["Monitoring Stack"]
        Prometheus["Prometheus Server"]
        Alertmanager["Alertmanager"]
        Grafana["Grafana"]
        Jaeger["Jaeger Tracing"]
    end

    Metrics --> Prometheus
    Prometheus --> Alertmanager
    Prometheus --> Grafana
    OpenTelemetry --> Jaeger
    Jaeger --> Grafana
```

## CI/CD Pipeline

```mermaid
graph TD
    subgraph GitOps
        GitHub["GitHub Repository"]
    end

    subgraph CI["Jenkins CI Pipeline"]
        Build["Build Docker Image"]
        Test["Run Tests"]
        DockerRegistry["Push to Docker Registry"]
    end

    subgraph CD["Deployment"]
        Deploy["Deploy to Kubernetes"]
        Monitor["Deploy Monitoring Stack"]
        Verify["Verify Deployment"]
    end

    GitHub --> |Webhook| Build
    Build --> Test
    Test --> DockerRegistry
    DockerRegistry --> Deploy
    Deploy --> Monitor
    Monitor --> Verify
```

## Kubernetes Deployment

```mermaid
graph TD
    subgraph Kubernetes["Kubernetes Cluster"]
        subgraph API["API Deployment"]
            Replica1["API Pod 1"]
            Replica2["API Pod 2"]
            Replica3["API Pod 3"]
            APIService["API Service"]
            Ingress["Ingress Controller"]
        end

        subgraph MonitoringStack["Monitoring Deployment"]
            PromPod["Prometheus Pod"]
            GrafanaPod["Grafana Pod"]
            JaegerPod["Jaeger Pod"]
            PromService["Prometheus Service"]
            GrafanaService["Grafana Service"]
            JaegerService["Jaeger Service"]
        end
    end

    Replica1 --> APIService
    Replica2 --> APIService
    Replica3 --> APIService
    APIService --> Ingress
    
    PromPod --> PromService
    GrafanaPod --> GrafanaService
    JaegerPod --> JaegerService
    
    PromService --> Ingress
    GrafanaService --> Ingress
    JaegerService --> Ingress
```

## Docker Compose Development Environment

```mermaid
graph TD
    subgraph DockerCompose["Docker Compose Environment"]
        APIContainer["API Container"]
        PrometheusContainer["Prometheus Container"]
        GrafanaContainer["Grafana Container"]
        JaegerContainer["Jaeger Container"]
    end

    APIContainer -- "Metrics" --> PrometheusContainer
    APIContainer -- "Traces" --> JaegerContainer
    PrometheusContainer -- "Data Source" --> GrafanaContainer
    JaegerContainer -- "Data Source" --> GrafanaContainer
```

## Model Inference Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as "FastAPI"
    participant Tokenizer as "Tokenizer"
    participant Model as "Qwen2.5-Coder-1.5B Model"
    participant Parser as "YAML Parser"

    Client->>API: POST with MCQ question & choices
    API->>API: Format prompt
    API->>Tokenizer: Tokenize prompt
    Tokenizer->>Model: Send tokens
    Model->>Model: Generate response
    Model->>Parser: YAML output
    Parser->>API: Parsed structured response
    API->>Client: Return formatted answer
```