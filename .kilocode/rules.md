# Project Rules

## Environment Configuration

- **Do not use `.env` files** for configuration
- Environment variables should be passed directly via:
  - Shell environment (`export VAR=value`)
  - Docker Compose command line (`MILVUS_PORT=19530 docker compose up`)
  - CI/CD pipeline configuration
  - Container orchestration secrets (Kubernetes, etc.)
