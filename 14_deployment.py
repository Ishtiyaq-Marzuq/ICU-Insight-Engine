"""
14_deployment.py

Deployment scripts and configuration for ICU patient monitoring system.
Provides Docker containerization, deployment automation, and system configuration.

Features:
- Docker containerization for all components
- Kubernetes deployment configurations
- Environment-specific configurations
- Health monitoring and logging
- Backup and recovery procedures
- Security configurations

Usage:
    python 14_deployment.py --build_docker
    python 14_deployment.py --deploy --environment production
    python 14_deployment.py --health_check
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import logging
import subprocess
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Deployment management system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.deployment_config = self._load_deployment_config()
    
    def _load_deployment_config(self) -> Dict:
        """Load deployment configuration"""
        return {
            'environments': {
                'development': {
                    'replicas': 1,
                    'resources': {'cpu': '500m', 'memory': '1Gi'},
                    'database_url': 'sqlite:///dev.db',
                    'log_level': 'DEBUG'
                },
                'staging': {
                    'replicas': 2,
                    'resources': {'cpu': '1000m', 'memory': '2Gi'},
                    'database_url': 'postgresql://staging:password@staging-db:5432/icu_monitor',
                    'log_level': 'INFO'
                },
                'production': {
                    'replicas': 3,
                    'resources': {'cpu': '2000m', 'memory': '4Gi'},
                    'database_url': 'postgresql://prod:password@prod-db:5432/icu_monitor',
                    'log_level': 'WARNING'
                }
            },
            'services': {
                'api': {
                    'port': 8000,
                    'health_check': '/health',
                    'dependencies': ['database', 'redis']
                },
                'monitoring': {
                    'port': 8001,
                    'health_check': '/monitoring/health',
                    'dependencies': ['api', 'database']
                },
                'dashboard': {
                    'port': 8501,
                    'health_check': '/_stcore/health',
                    'dependencies': ['api']
                }
            }
        }
    
    def create_dockerfile(self, service: str) -> str:
        """Create Dockerfile for a specific service"""
        
        if service == 'api':
            dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        elif service == 'monitoring':
            dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8001/monitoring/health || exit 1

# Run monitoring service
CMD ["python", "08_realtime_pipeline.py"]
"""
        
        elif service == 'dashboard':
            dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
        
        else:
            raise ValueError(f"Unknown service: {service}")
        
        return dockerfile_content
    
    def create_docker_compose(self, environment: str = 'development') -> str:
        """Create docker-compose.yml for the specified environment"""
        
        env_config = self.deployment_config['environments'][environment]
        
        compose_content = f"""
version: '3.8'

services:
  database:
    image: postgres:13
    environment:
      POSTGRES_DB: icu_monitor
      POSTGRES_USER: icu_user
      POSTGRES_PASSWORD: icu_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U icu_user -d icu_monitor"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://icu_user:icu_password@database:5432/icu_monitor
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL={env_config['log_level']}
    depends_on:
      database:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  monitoring:
    build:
      context: .
      dockerfile: Dockerfile.monitoring
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://icu_user:icu_password@database:5432/icu_monitor
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL={env_config['log_level']}
    depends_on:
      api:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/monitoring/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
      - LOG_LEVEL={env_config['log_level']}
    depends_on:
      api:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - dashboard
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
"""
        
        return compose_content
    
    def create_kubernetes_manifests(self, environment: str = 'production') -> Dict[str, str]:
        """Create Kubernetes deployment manifests"""
        
        env_config = self.deployment_config['environments'][environment]
        
        manifests = {}
        
        # Namespace
        manifests['namespace.yaml'] = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: icu-monitor
  labels:
    name: icu-monitor
"""
        
        # ConfigMap
        manifests['configmap.yaml'] = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: icu-monitor-config
  namespace: icu-monitor
data:
  DATABASE_URL: "postgresql://icu_user:icu_password@postgres-service:5432/icu_monitor"
  REDIS_URL: "redis://redis-service:6379"
  LOG_LEVEL: "{env_config['log_level']}"
  ENVIRONMENT: "{environment}"
"""
        
        # Secret
        manifests['secret.yaml'] = f"""
apiVersion: v1
kind: Secret
metadata:
  name: icu-monitor-secret
  namespace: icu-monitor
type: Opaque
data:
  database-password: aWN1X3Bhc3N3b3Jk  # base64 encoded
  redis-password: cmVkaXNfcGFzc3dvcmQ=  # base64 encoded
"""
        
        # API Deployment
        manifests['api-deployment.yaml'] = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-deployment
  namespace: icu-monitor
spec:
  replicas: {env_config['replicas']}
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: icu-monitor-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: icu-monitor-config
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: icu-monitor-config
              key: REDIS_URL
        resources:
          requests:
            cpu: {env_config['resources']['cpu']}
            memory: {env_config['resources']['memory']}
          limits:
            cpu: {env_config['resources']['cpu']}
            memory: {env_config['resources']['memory']}
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
"""
        
        # API Service
        manifests['api-service.yaml'] = f"""
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: icu-monitor
spec:
  selector:
    app: api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
"""
        
        # Dashboard Deployment
        manifests['dashboard-deployment.yaml'] = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dashboard-deployment
  namespace: icu-monitor
spec:
  replicas: {env_config['replicas']}
  selector:
    matchLabels:
      app: dashboard
  template:
    metadata:
      labels:
        app: dashboard
    spec:
      containers:
      - name: dashboard
        image: icu-monitor-dashboard:latest
        ports:
        - containerPort: 8501
        env:
        - name: API_URL
          value: "http://api-service:8000"
        resources:
          requests:
            cpu: {env_config['resources']['cpu']}
            memory: {env_config['resources']['memory']}
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
"""
        
        # Dashboard Service
        manifests['dashboard-service.yaml'] = f"""
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
  namespace: icu-monitor
spec:
  selector:
    app: dashboard
  ports:
  - port: 8501
    targetPort: 8501
  type: LoadBalancer
"""
        
        # Ingress
        manifests['ingress.yaml'] = f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: icu-monitor-ingress
  namespace: icu-monitor
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - icu-monitor.example.com
    secretName: icu-monitor-tls
  rules:
  - host: icu-monitor.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dashboard-service
            port:
              number: 8501
"""
        
        return manifests
    
    def create_nginx_config(self) -> str:
        """Create nginx configuration"""
        
        nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }
    
    upstream dashboard_backend {
        server dashboard:8501;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # API routes
        location /api/ {
            proxy_pass http://api_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Dashboard routes
        location / {
            proxy_pass http://dashboard_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        # Health check
        location /health {
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }
    }
}
"""
        
        return nginx_config
    
    def create_health_check_script(self) -> str:
        """Create health check script"""
        
        script_content = """#!/bin/bash

# Health check script for ICU monitoring system
set -e

echo "Starting health check..."

# Check API health
echo "Checking API health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy"
else
    echo "❌ API is unhealthy"
    exit 1
fi

# Check Dashboard health
echo "Checking Dashboard health..."
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Dashboard is healthy"
else
    echo "❌ Dashboard is unhealthy"
    exit 1
fi

# Check Database connectivity
echo "Checking Database connectivity..."
if python -c "import psycopg2; psycopg2.connect('postgresql://icu_user:icu_password@localhost:5432/icu_monitor')" > /dev/null 2>&1; then
    echo "✅ Database is accessible"
else
    echo "❌ Database is not accessible"
    exit 1
fi

# Check Redis connectivity
echo "Checking Redis connectivity..."
if python -c "import redis; redis.Redis(host='localhost', port=6379).ping()" > /dev/null 2>&1; then
    echo "✅ Redis is accessible"
else
    echo "❌ Redis is not accessible"
    exit 1
fi

echo "🎉 All services are healthy!"
"""
        
        return script_content
    
    def create_backup_script(self) -> str:
        """Create backup script"""
        
        script_content = """#!/bin/bash

# Backup script for ICU monitoring system
set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="icu_monitor_backup_${DATE}.tar.gz"

echo "Starting backup process..."

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
echo "Backing up database..."
pg_dump -h localhost -U icu_user -d icu_monitor > $BACKUP_DIR/database_${DATE}.sql

# Backup application data
echo "Backing up application data..."
tar -czf $BACKUP_DIR/$BACKUP_FILE \\
    --exclude='.git' \\
    --exclude='__pycache__' \\
    --exclude='*.pyc' \\
    --exclude='.venv' \\
    --exclude='data/raw' \\
    /app

# Backup configuration
echo "Backing up configuration..."
cp -r /app/config.py $BACKUP_DIR/
cp -r /app/requirements.txt $BACKUP_DIR/

# Cleanup old backups (keep last 7 days)
echo "Cleaning up old backups..."
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
"""
        
        return script_content
    
    def build_docker_images(self):
        """Build Docker images for all services"""
        
        services = ['api', 'monitoring', 'dashboard']
        
        for service in services:
            logger.info(f"Building Docker image for {service}...")
            
            # Create Dockerfile
            dockerfile_content = self.create_dockerfile(service)
            dockerfile_path = self.project_root / f"Dockerfile.{service}"
            
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build image
            image_name = f"icu-monitor-{service}:latest"
            build_cmd = [
                'docker', 'build',
                '-f', str(dockerfile_path),
                '-t', image_name,
                '.'
            ]
            
            try:
                subprocess.run(build_cmd, check=True, cwd=self.project_root)
                logger.info(f"Successfully built {image_name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to build {image_name}: {e}")
                raise
    
    def deploy_environment(self, environment: str):
        """Deploy to specified environment"""
        
        logger.info(f"Deploying to {environment} environment...")
        
        # Create docker-compose file
        compose_content = self.create_docker_compose(environment)
        compose_path = self.project_root / f"docker-compose.{environment}.yml"
        
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        # Deploy with docker-compose
        deploy_cmd = ['docker-compose', '-f', str(compose_path), 'up', '-d']
        
        try:
            subprocess.run(deploy_cmd, check=True, cwd=self.project_root)
            logger.info(f"Successfully deployed to {environment}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy to {environment}: {e}")
            raise
    
    def run_health_check(self):
        """Run health check on deployed services"""
        
        logger.info("Running health check...")
        
        # Create health check script
        script_content = self.create_health_check_script()
        script_path = self.project_root / "health_check.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        # Run health check
        try:
            subprocess.run([str(script_path)], check=True, cwd=self.project_root)
            logger.info("Health check passed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Health check failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Deployment Management")
    parser.add_argument("--build_docker", action="store_true", help="Build Docker images")
    parser.add_argument("--deploy", action="store_true", help="Deploy to environment")
    parser.add_argument("--environment", type=str, default="development", 
                       choices=["development", "staging", "production"],
                       help="Target environment")
    parser.add_argument("--health_check", action="store_true", help="Run health check")
    parser.add_argument("--create_manifests", action="store_true", help="Create Kubernetes manifests")
    parser.add_argument("--output_dir", type=str, help="Output directory for manifests")
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    manager = DeploymentManager()
    
    if args.build_docker:
        logger.info("Building Docker images...")
        manager.build_docker_images()
        print("✅ Docker images built successfully")
    
    if args.deploy:
        logger.info(f"Deploying to {args.environment} environment...")
        manager.deploy_environment(args.environment)
        print(f"✅ Successfully deployed to {args.environment}")
    
    if args.health_check:
        logger.info("Running health check...")
        manager.run_health_check()
        print("✅ Health check passed")
    
    if args.create_manifests:
        logger.info("Creating Kubernetes manifests...")
        manifests = manager.create_kubernetes_manifests(args.environment)
        
        output_dir = Path(args.output_dir) if args.output_dir else config.RESULTS_DIR / "k8s"
        output_dir.mkdir(exist_ok=True)
        
        for filename, content in manifests.items():
            file_path = output_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created {file_path}")
        
        print("✅ Kubernetes manifests created successfully")

if __name__ == "__main__":
    main()
