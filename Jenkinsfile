pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: qwen-coder-api-builder
spec:
  containers:
  - name: docker
    image: docker:20.10.16
    command:
    - cat
    tty: true
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
  - name: kubectl
    image: bitnami/kubectl:1.25
    command:
    - cat
    tty: true
  - name: helm
    image: alpine/helm:3.11.1
    command:
    - cat
    tty: true
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
      type: Socket
"""
        }
    }
    
    environment {
        DOCKER_REGISTRY = "your-registry.example.com"
        DOCKER_IMAGE = "${DOCKER_REGISTRY}/qwen-coder-api"
        DOCKER_TAG = "${BUILD_NUMBER}"
        KUBERNETES_NAMESPACE = "qwen-api"
        KUBECONFIG_CREDENTIALS_ID = "kubeconfig"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Run Tests') {
            steps {
                sh 'echo "Running tests..."'
                // Add your test commands here
                // sh 'python -m pytest tests/'
            }
        }
        
        stage('Build and Push Docker Image') {
            steps {
                container('docker') {
                    withCredentials([string(credentialsId: 'docker-registry-credentials', variable: 'DOCKER_AUTH')]) {
                        sh """
                            echo \${DOCKER_AUTH} | docker login ${DOCKER_REGISTRY} --username user --password-stdin
                            docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
                            docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:latest
                            docker push ${DOCKER_IMAGE}:${DOCKER_TAG}
                            docker push ${DOCKER_IMAGE}:latest
                        """
                    }
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                container('kubectl') {
                    withCredentials([file(credentialsId: "${KUBECONFIG_CREDENTIALS_ID}", variable: 'KUBECONFIG')]) {
                        sh """
                            export KUBECONFIG=\${KUBECONFIG}
                            
                            # Create namespace if it doesn't exist
                            kubectl get namespace ${KUBERNETES_NAMESPACE} || kubectl create namespace ${KUBERNETES_NAMESPACE}
                            
                            # Update deployment image
                            kubectl set image deployment/qwen-coder-api qwen-coder-api=${DOCKER_IMAGE}:${DOCKER_TAG} -n ${KUBERNETES_NAMESPACE}
                            
                            # Wait for rollout
                            kubectl rollout status deployment/qwen-coder-api -n ${KUBERNETES_NAMESPACE} --timeout=300s
                        """
                    }
                }
            }
        }
        
        stage('Deploy Monitoring Stack') {
            when {
                expression { return params.DEPLOY_MONITORING }
            }
            steps {
                container('helm') {
                    withCredentials([file(credentialsId: "${KUBECONFIG_CREDENTIALS_ID}", variable: 'KUBECONFIG')]) {
                        sh """
                            export KUBECONFIG=\${KUBECONFIG}
                            
                            # Add Helm repositories
                            helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
                            helm repo add grafana https://grafana.github.io/helm-charts
                            helm repo update
                            
                            # Deploy Prometheus
                            helm upgrade --install prometheus prometheus-community/prometheus \
                                --namespace ${KUBERNETES_NAMESPACE} \
                                -f kubernetes/prometheus-values.yaml
                            
                            # Deploy Grafana
                            helm upgrade --install grafana grafana/grafana \
                                --namespace ${KUBERNETES_NAMESPACE} \
                                -f kubernetes/grafana-values.yaml
                            
                            # Deploy Jaeger
                            helm upgrade --install jaeger jaegertracing/jaeger \
                                --namespace ${KUBERNETES_NAMESPACE} \
                                -f kubernetes/jaeger-values.yaml
                        """
                    }
                }
            }
        }
    }
    
    post {
        success {
            echo 'Deployment successful!'
        }
        failure {
            echo 'Deployment failed!'
        }
    }
}