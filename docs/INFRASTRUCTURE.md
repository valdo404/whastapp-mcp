# Infrastructure Setup

This document describes how to set up and run the Milvus infrastructure for the WhatsApp MCP Server.

## Prerequisites

- **Docker Desktop** (or Docker Engine + Docker Compose)
  - macOS: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Windows: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: Install Docker Engine and Docker Compose plugin

- **Resource Requirements**:
  | Resource | Minimum | Recommended |
  |----------|---------|-------------|
  | CPU      | 2 cores | 4+ cores    |
  | Memory   | 8 GB    | 16 GB       |
  | Disk     | 20 GB   | 50+ GB SSD  |

> **Note for Apple Silicon (M1/M2/M3)**: Ensure Docker Desktop has at least 8 GB RAM allocated in Settings → Resources.

## Quick Start

### 1. Start the Infrastructure

```bash
# Navigate to the project directory
cd /path/to/whatsapp-mcp

# Start all services in detached mode
docker compose up -d
```

### 2. Verify Services are Running

```bash
# Check service status
docker compose ps

# Expected output:
# NAME               STATUS                   PORTS
# milvus-etcd        running (healthy)        2379/tcp
# milvus-minio       running (healthy)        0.0.0.0:9000-9001->9000-9001/tcp
# milvus-standalone  running (healthy)        0.0.0.0:9091->9091/tcp, 0.0.0.0:19530->19530/tcp
```

### 3. Wait for Milvus to be Ready

Milvus takes approximately 60-90 seconds to fully initialize. You can monitor the startup:

```bash
# Watch Milvus logs
docker compose logs -f milvus

# Wait until you see: "Milvus Proxy successfully initialized"
```

## Services Overview

| Service | Container Name | Ports | Purpose |
|---------|---------------|-------|---------|
| **etcd** | milvus-etcd | 2379 (internal) | Metadata storage for Milvus |
| **MinIO** | milvus-minio | 9000 (API), 9001 (Console) | S3-compatible object storage |
| **Milvus** | milvus-standalone | 19530 (gRPC), 9091 (metrics) | Vector database |

## Accessing Services

### MinIO Console (Web UI)

Access the MinIO web console to browse stored data:

- **URL**: http://localhost:9001
- **Username**: `minioadmin`
- **Password**: `minioadmin`

### Milvus Health Check

```bash
# Check Milvus health endpoint
curl http://localhost:9091/healthz
```

## Common Operations

### Stop Services

```bash
# Stop all services (preserves data)
docker compose stop

# Or stop and remove containers (preserves volumes)
docker compose down
```

### View Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs milvus
docker compose logs minio
docker compose logs etcd

# Follow logs in real-time
docker compose logs -f milvus
```

### Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart milvus
```

### Reset Data (Clean Start)

⚠️ **Warning**: This will delete all stored data!

```bash
# Stop and remove containers and volumes
docker compose down -v

# Start fresh
docker compose up -d
```

## Configuration

The default configuration uses hardcoded values suitable for local development:

| Setting | Default Value | Description |
|---------|---------------|-------------|
| Milvus gRPC Port | 19530 | Vector database API |
| Milvus Metrics Port | 9091 | Health checks and metrics |
| MinIO API Port | 9000 | Object storage API |
| MinIO Console Port | 9001 | Web UI |
| MinIO Credentials | minioadmin/minioadmin | Access credentials |

To customize ports or credentials for production, modify the values directly in [`docker-compose.yml`](../docker-compose.yml).

## Troubleshooting

### Services Won't Start

1. **Check Docker is running**:
   ```bash
   docker info
   ```

2. **Check for port conflicts**:
   ```bash
   # Check if ports are in use
   lsof -i :19530
   lsof -i :9000
   lsof -i :9001
   ```

3. **Check available resources**:
   ```bash
   docker system info | grep -E "Memory|CPUs"
   ```

### Milvus Fails Health Check

1. **Check dependencies are healthy**:
   ```bash
   docker compose ps
   ```
   Both `milvus-etcd` and `milvus-minio` should show "healthy" status.

2. **Check Milvus logs for errors**:
   ```bash
   docker compose logs milvus | tail -50
   ```

3. **Increase start period** (if on slower hardware):
   Edit `docker-compose.yml` and increase `start_period` for the milvus service.

### Connection Refused from Application

1. **Verify Milvus is running and healthy**:
   ```bash
   docker compose ps milvus
   ```

2. **Test connectivity**:
   ```bash
   # Test gRPC port
   nc -zv localhost 19530
   
   # Test health endpoint
   curl http://localhost:9091/healthz
   ```

3. **Check firewall settings** (Linux):
   ```bash
   sudo ufw status
   ```

### Out of Disk Space

```bash
# Check Docker disk usage
docker system df

# Clean up unused resources
docker system prune -a --volumes
```

## Architecture Reference

For detailed architecture information, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).
