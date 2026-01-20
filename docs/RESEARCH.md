# Research: Milvus Vector Database Setup with S3 Storage

> **Date**: January 2026  
> **Purpose**: Semantic search solution for WhatsApp conversations

---

## Table of Contents

1. [Milvus Architecture with S3](#1-milvus-architecture-with-s3)
2. [Docker Compose Setup](#2-docker-compose-setup-for-milvus)
3. [Embedding Models for Semantic Search](#3-embedding-models-for-semantic-search)
4. [WhatsApp Chat Export Format](#4-whatsapp-chat-export-format)
5. [Milvus Python SDK (pymilvus)](#5-milvus-python-sdk-pymilvus)
6. [Recommended Approach](#6-recommended-approach)
7. [FastMCP and Model Context Protocol (MCP)](#7-fastmcp-and-model-context-protocol-mcp)
8. [FastMCP Installation and Setup](#8-fastmcp-installation-and-setup)
9. [FastMCP Tool Definitions for Semantic Search](#9-fastmcp-tool-definitions-for-semantic-search)
10. [Docker Deployment for FastMCP](#10-docker-deployment-for-fastmcp)
11. [Claude Desktop Integration](#11-claude-desktop-integration)
12. [Best Practices for MCP + Vector Database](#12-best-practices-for-mcp--vector-database)

---

## 1. Milvus Architecture with S3

### Overview

Milvus is a cloud-native vector database with a **shared-storage architecture** featuring fully disaggregated storage and compute layers. This design enables horizontal scaling and flexible resource allocation.

### Core Components

| Component | Purpose | Notes |
|-----------|---------|-------|
| **Milvus** | Core vector database engine | Handles vector search, indexing, and queries |
| **etcd** | Metadata storage | Distributed key-value store for internal metadata |
| **MinIO/S3** | Object storage | Persists data files, index files, and log snapshots |
| **Pulsar** (cluster only) | Message broker | Required only for distributed deployments |

### Storage Architecture

Milvus uses object storage (MinIO/S3) for:
- **Data Files**: Final storage of vector data, metadata, and segments
- **Index Files**: Pre-built indexes for faster search
- **Log Snapshots**: For recovery and durability

### S3/MinIO Configuration

Key configuration parameters in `milvus.yaml`:

```yaml
minio:
  address: localhost        # MinIO/S3 endpoint
  port: 9000               # Default MinIO port
  accessKeyID: minioadmin  # Access credentials
  secretAccessKey: minioadmin
  bucketName: a-bucket     # Storage bucket name
  rootPath: files          # Root path within bucket
  useSSL: false            # Enable for production S3
```

**Environment Variables:**
- `MINIO_ADDRESS` - MinIO/S3 endpoint
- `MINIO_ACCESS_KEY_ID` - Access key
- `MINIO_SECRET_ACCESS_KEY` - Secret key

### Standalone vs Cluster

| Mode | Components | Use Case |
|------|------------|----------|
| **Standalone** | Milvus + etcd + MinIO | Development, testing, small-scale |
| **Cluster** | Milvus + etcd + MinIO + Pulsar | Production, high-throughput |

**Recommendation**: Use **Standalone** mode for local development with WhatsApp chat semantic search.

### Official Documentation

- [Milvus Architecture Overview](https://milvus.io/docs/architecture_overview.md)
- [MinIO Configuration](https://milvus.io/docs/configure_minio.md)
- [Object Storage with Milvus Operator](https://milvus.io/docs/object_storage_operator.md)

---

## 2. Docker Compose Setup for Milvus

### Quick Start

```bash
# Download the official docker-compose file
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus
docker compose up -d

# Verify containers are running
docker compose ps
```

### Expected Containers

After startup, three containers should be running:

| Container | Ports | Purpose |
|-----------|-------|---------|
| `milvus-standalone` | 19530, 9091 | Main Milvus service |
| `milvus-etcd` | 2379, 2380 | Metadata storage |
| `milvus-minio` | 9000, 9001 | Object storage |

### Sample Docker Compose Configuration

```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

### Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 2 vCPUs | 4+ vCPUs |
| **Memory** | 8 GB | 16+ GB |
| **Disk** | 20 GB | 100+ GB SSD |

**macOS Note**: For Docker Desktop on macOS (especially M1/M2), allocate at least 2 vCPUs and 8 GB RAM in Docker settings.

### Custom Configuration

To override default settings, mount a custom `milvus.yaml`:

```yaml
volumes:
  - /local/path/to/milvus.yaml:/milvus/configs/milvus.yaml
  - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
```

### Useful Commands

```bash
# Stop Milvus
docker compose down

# Delete all data
sudo rm -rf volumes

# View logs
docker compose logs -f milvus-standalone

# Access MinIO console
# Open http://localhost:9001 (login: minioadmin/minioadmin)
```

### Official Documentation

- [Install Milvus Standalone with Docker Compose](https://milvus.io/docs/install_standalone-docker-compose.md)
- [Configure Milvus with Docker Compose](https://milvus.io/docs/configure-docker.md)

---

## 3. Embedding Models for Semantic Search

### Model Comparison

| Model | Dimensions | Multilingual | Cost | Hosting |
|-------|------------|--------------|------|---------|
| **OpenAI text-embedding-3-small** | 1536 (configurable) | ✅ Yes | $0.00002/1K tokens | API |
| **OpenAI text-embedding-3-large** | 3072 (configurable) | ✅ Yes | $0.00013/1K tokens | API |
| **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** | 384 | ✅ 50+ languages | Free | Local |
| **sentence-transformers/paraphrase-multilingual-mpnet-base-v2** | 768 | ✅ 50+ languages | Free | Local |
| **intfloat/multilingual-e5-large** | 1024 | ✅ 100+ languages | Free | Local |
| **dangvantuan/french-document-embedding** | 768 | 🇫🇷 French-optimized | Free | Local |
| **Cohere embed-multilingual-v3** | 1024 | ✅ 100+ languages | API pricing | API |

### Recommended for French WhatsApp Chats

#### Option 1: OpenAI (Best Quality, API-based)

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
```

**Pros**: Best multilingual performance (MIRACL benchmark: 44% for small, 54.9% for large)  
**Cons**: API costs, requires internet, data privacy considerations

#### Option 2: Sentence Transformers (Free, Local)

```python
from sentence_transformers import SentenceTransformer

# Multilingual model supporting French
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def get_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()
```

**Pros**: Free, runs locally, no API calls, good multilingual support  
**Cons**: Requires GPU for fast inference, slightly lower quality than OpenAI

#### Option 3: French-Specific Model

```python
from sentence_transformers import SentenceTransformer

# Optimized for French documents
model = SentenceTransformer('dangvantuan/french-document-embedding')

def get_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()
```

**Pros**: Optimized for French, long context support (8192 tokens)  
**Cons**: May not handle code-switching (French/English mix) as well

### Multilingual Considerations

For WhatsApp chats that may contain:
- French text
- English words/phrases
- Emojis and special characters
- Informal/colloquial language

**Recommendation**: Use `paraphrase-multilingual-mpnet-base-v2` or OpenAI's `text-embedding-3-small` for best handling of mixed-language content.

### Installation

```bash
# For sentence-transformers (local)
pip install sentence-transformers

# For OpenAI
pip install openai
```

### Official Documentation

- [Sentence Transformers Pretrained Models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Hugging Face French Embedding Model](https://huggingface.co/dangvantuan/french-document-embedding)

---

## 4. WhatsApp Chat Export Format

### Export Process

1. Open WhatsApp chat
2. Tap menu (⋮) → More → Export chat
3. Choose "Without media" for text-only export
4. Save/share the `.txt` file

### Standard Format

WhatsApp exports vary by OS and locale:

**Android (English)**:
```
12/25/23, 14:30 - John Doe: Hello, how are you?
12/25/23, 14:31 - Jane Smith: I'm good, thanks!
12/25/23, 14:32 - John Doe: This is a message
that spans multiple lines
12/25/23, 14:33 - Jane Smith: <Media omitted>
```

**iOS (English)**:
```
[12/25/23, 2:30:45 PM] John Doe: Hello, how are you?
[12/25/23, 2:31:12 PM] Jane Smith: I'm good, thanks!
```

**French locale**:
```
25/12/2023 à 14:30 - Jean Dupont: Bonjour, comment ça va?
25/12/2023 à 14:31 - Marie Martin: Ça va bien, merci!
```

### Parsing Regex Patterns

```python
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

@dataclass
class WhatsAppMessage:
    timestamp: datetime
    sender: str
    content: str
    is_media: bool = False
    is_system: bool = False

# Universal regex pattern (handles multiple formats)
PATTERNS = [
    # Android: "12/25/23, 14:30 - Name: Message"
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:AM|PM|am|pm)?\s*-\s*([^:]+):\s*(.+)$',
    # iOS: "[12/25/23, 2:30:45 PM] Name: Message"
    r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:AM|PM|am|pm)?\]\s*([^:]+):\s*(.+)$',
    # French: "25/12/2023 à 14:30 - Name: Message"
    r'^(\d{1,2}/\d{1,2}/\d{2,4})\s*[àa]\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.+)$',
]

def parse_whatsapp_line(line: str) -> Optional[WhatsAppMessage]:
    """Parse a single line from WhatsApp export."""
    for pattern in PATTERNS:
        match = re.match(pattern, line.strip(), re.UNICODE)
        if match:
            date_str, time_str, sender, content = match.groups()
            
            # Handle media and system messages
            is_media = '<Media omitted>' in content or '<image omitted>' in content.lower()
            is_system = sender.lower() in ['system', 'whatsapp']
            
            return WhatsAppMessage(
                timestamp=parse_datetime(date_str, time_str),
                sender=sender.strip(),
                content=content.strip(),
                is_media=is_media,
                is_system=is_system
            )
    return None

def parse_whatsapp_chat(file_path: str) -> list[WhatsAppMessage]:
    """Parse entire WhatsApp chat export file."""
    messages = []
    current_message = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Remove BOM and special characters
            line = line.strip('\ufeff\u200e\u200f')
            
            parsed = parse_whatsapp_line(line)
            if parsed:
                if current_message:
                    messages.append(current_message)
                current_message = parsed
            elif current_message and line.strip():
                # Multi-line message continuation
                current_message.content += '\n' + line.strip()
        
        if current_message:
            messages.append(current_message)
    
    return messages
```

### Python Libraries for WhatsApp Parsing

| Library | Features | Installation |
|---------|----------|--------------|
| **whatstk** | Full parsing, visualization, pandas integration | `pip install whatstk` |
| **WhatsR** (R) | Parsing, anonymization, visualization | R package |
| **whatsapp-chat-exporter** | Database parsing (Android/iOS) | `pip install whatsapp-chat-exporter` |

#### Using whatstk

```python
from whatstk import WhatsAppChat

# Load and parse chat
chat = WhatsAppChat.from_source("path/to/chat.txt")

# Access as pandas DataFrame
df = chat.df
print(df.columns)  # ['date', 'username', 'message']

# Filter by user
user_messages = df[df['username'] == 'John Doe']
```

### Special Message Types

| Type | Pattern | Handling |
|------|---------|----------|
| Media omitted | `<Media omitted>` | Skip or flag |
| Deleted message | `This message was deleted` | Skip or flag |
| System message | `John created group "..."` | Skip |
| Location | `Location: https://maps...` | Extract URL |
| Contact | `Contact card: ...` | Extract name |

### Official Documentation

- [WhatsApp Help: Export Chat History](https://faq.whatsapp.com/1180414079177245)
- [whatstk Documentation](https://whatstk.readthedocs.io/)

---

## 5. Milvus Python SDK (pymilvus)

### Installation

```bash
pip install pymilvus
```

### Collection Schema Design for Chat Messages

```python
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

# Connect to Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# Define schema for WhatsApp messages
fields = [
    # Primary key (auto-generated)
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    ),
    # Message content
    FieldSchema(
        name="content",
        dtype=DataType.VARCHAR,
        max_length=65535
    ),
    # Sender name
    FieldSchema(
        name="sender",
        dtype=DataType.VARCHAR,
        max_length=256
    ),
    # Timestamp (Unix epoch)
    FieldSchema(
        name="timestamp",
        dtype=DataType.INT64
    ),
    # Chat/conversation identifier
    FieldSchema(
        name="chat_id",
        dtype=DataType.VARCHAR,
        max_length=256
    ),
    # Dense vector embedding
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=768  # Adjust based on embedding model
    )
]

schema = CollectionSchema(
    fields=fields,
    description="WhatsApp chat messages with embeddings"
)

# Create collection
collection = Collection(
    name="whatsapp_messages",
    schema=schema,
    consistency_level="Bounded"
)
```

### Index Types for Semantic Search

| Index Type | Use Case | Memory | Speed | Accuracy |
|------------|----------|--------|-------|----------|
| **FLAT** | Small datasets (<100K) | High | Slow | 100% |
| **IVF_FLAT** | Medium datasets | Medium | Fast | High |
| **IVF_SQ8** | Large datasets | Low | Fast | Good |
| **HNSW** | Production (recommended) | High | Very Fast | High |
| **DISKANN** | Very large datasets | Low | Fast | Good |

#### Recommended Index Configuration

```python
# Create HNSW index for production use
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",  # or "L2", "IP"
    "params": {
        "M": 16,              # Max connections per node
        "efConstruction": 256  # Build-time accuracy
    }
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)

# Create scalar index for filtering
collection.create_index(
    field_name="sender",
    index_params={"index_type": "INVERTED"}
)

collection.create_index(
    field_name="timestamp",
    index_params={"index_type": "STL_SORT"}
)

# Load collection into memory
collection.load()
```

### Inserting Data

```python
from sentence_transformers import SentenceTransformer
import time

# Initialize embedding model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def insert_messages(messages: list[WhatsAppMessage], chat_id: str):
    """Insert parsed WhatsApp messages into Milvus."""
    
    # Prepare data
    contents = [msg.content for msg in messages]
    senders = [msg.sender for msg in messages]
    timestamps = [int(msg.timestamp.timestamp()) for msg in messages]
    chat_ids = [chat_id] * len(messages)
    
    # Generate embeddings
    embeddings = model.encode(contents).tolist()
    
    # Insert into collection
    data = [
        contents,
        senders,
        timestamps,
        chat_ids,
        embeddings
    ]
    
    collection.insert(data)
    collection.flush()  # Ensure data is persisted
```

### Semantic Search

```python
def semantic_search(
    query: str,
    chat_id: str = None,
    sender: str = None,
    limit: int = 10
) -> list[dict]:
    """Search for semantically similar messages."""
    
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    
    # Build filter expression
    filters = []
    if chat_id:
        filters.append(f'chat_id == "{chat_id}"')
    if sender:
        filters.append(f'sender == "{sender}"')
    
    expr = " and ".join(filters) if filters else None
    
    # Search parameters
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 64}  # Search-time accuracy
    }
    
    # Execute search
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        expr=expr,
        output_fields=["content", "sender", "timestamp", "chat_id"]
    )
    
    # Format results
    return [
        {
            "content": hit.entity.get("content"),
            "sender": hit.entity.get("sender"),
            "timestamp": hit.entity.get("timestamp"),
            "chat_id": hit.entity.get("chat_id"),
            "score": hit.score
        }
        for hit in results[0]
    ]
```

### Hybrid Search (Vector + Scalar Filtering)

```python
from pymilvus import AnnSearchRequest, WeightedRanker

def hybrid_search(
    query: str,
    time_start: int = None,
    time_end: int = None,
    senders: list[str] = None,
    limit: int = 10
) -> list[dict]:
    """Hybrid search with time range and sender filtering."""
    
    query_embedding = model.encode(query).tolist()
    
    # Build complex filter
    filters = []
    if time_start:
        filters.append(f"timestamp >= {time_start}")
    if time_end:
        filters.append(f"timestamp <= {time_end}")
    if senders:
        sender_list = ", ".join([f'"{s}"' for s in senders])
        filters.append(f"sender in [{sender_list}]")
    
    expr = " and ".join(filters) if filters else None
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 64}
    }
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        expr=expr,
        output_fields=["content", "sender", "timestamp"]
    )
    
    return [
        {
            "content": hit.entity.get("content"),
            "sender": hit.entity.get("sender"),
            "timestamp": hit.entity.get("timestamp"),
            "score": hit.score
        }
        for hit in results[0]
    ]
```

### Official Documentation

- [pymilvus Documentation](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md)
- [Collection Schema Design](https://milvus.io/docs/schema-hands-on.md)
- [Index Types](https://milvus.io/docs/index-explained.md)
- [Hybrid Search](https://milvus.io/docs/hybrid_search_with_milvus.md)

---

## 6. Recommended Approach

### Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  WhatsApp Chat  │────▶│  Parser/Loader   │────▶│  Embedding      │
│  Export (.txt)  │     │  (whatstk/custom)│     │  Model          │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Search API     │◀────│  Milvus          │◀────│  Vector         │
│  (FastAPI)      │     │  (Docker)        │     │  Embeddings     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Recommended Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Vector DB** | Milvus Standalone | Easy setup, production-ready |
| **Storage** | MinIO (Docker) | S3-compatible, local development |
| **Embedding** | `paraphrase-multilingual-mpnet-base-v2` | Free, multilingual, good quality |
| **Parser** | Custom + whatstk | Flexibility for French locale |
| **API** | FastAPI | Modern Python async framework |

### Implementation Steps

1. **Setup Infrastructure**
   ```bash
   # Start Milvus with Docker Compose
   docker compose up -d
   ```

2. **Parse WhatsApp Exports**
   ```python
   from whatstk import WhatsAppChat
   chat = WhatsAppChat.from_source("chat.txt")
   ```

3. **Generate Embeddings**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
   embeddings = model.encode(messages)
   ```

4. **Store in Milvus**
   ```python
   collection.insert([contents, senders, timestamps, embeddings])
   ```

5. **Search**
   ```python
   results = semantic_search("Quand est-ce qu'on se voit?")
   ```

### Performance Considerations

- **Batch Processing**: Insert messages in batches of 1000-5000
- **Index Building**: Build index after bulk insert for better performance
- **Memory**: HNSW index requires ~1.5x vector size in memory
- **Embedding Caching**: Cache embeddings to avoid recomputation

### Security Considerations

- Store MinIO credentials in environment variables
- Use TLS for production deployments
- Consider data anonymization for sensitive chats
- Implement access control for multi-user scenarios

---

## References

### Official Documentation
- [Milvus Documentation](https://milvus.io/docs)
- [pymilvus API Reference](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [whatstk Documentation](https://whatstk.readthedocs.io/)

### GitHub Repositories
- [Milvus](https://github.com/milvus-io/milvus)
- [pymilvus](https://github.com/milvus-io/pymilvus)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [whatstk](https://github.com/lucasrodes/whatstk)

---

## 7. FastMCP and Model Context Protocol (MCP)

### Overview

The **Model Context Protocol (MCP)** is an open protocol that enables seamless integration between LLM applications and external data sources and tools. It provides a standardized way to connect AI models (like Claude, Gemini, GPT) with the context they need.

**FastMCP** is a high-level Python framework that simplifies building MCP servers, handling all protocol complexities (JSON-RPC, session state, message formatting) so developers can focus on writing Python functions.

### MCP Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Host           │────▶│  MCP Client      │────▶│  MCP Server     │
│  (Claude, etc.) │     │  (Connector)     │     │  (Your Service) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

| Component | Description |
|-----------|-------------|
| **Host** | LLM application that initiates connections (Claude Desktop, Cursor, VS Code) |
| **Client** | Connector within the host that speaks MCP protocol |
| **Server** | Service that provides context, tools, and resources to the LLM |

### MCP Core Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Tools** | Functions the LLM can execute | `search_messages()`, `get_chat_history()` |
| **Resources** | Read-only data exposed to the LLM | Configuration, static files, database schemas |
| **Prompts** | Reusable prompt templates | Pre-defined search queries |

### Transport Mechanisms

MCP supports multiple transport methods for client-server communication:

| Transport | Use Case | Description |
|-----------|----------|-------------|
| **stdio** | Local development | Client spawns server as subprocess, communicates via stdin/stdout |
| **HTTP/SSE** | Remote deployment | Server runs independently, uses HTTP POST + Server-Sent Events |
| **Streamable HTTP** | Production (newer) | Modern HTTP transport replacing SSE |

**stdio Transport Flow:**
```
Client (Parent Process)
    │
    ├── Spawns Server (Child Process)
    │
    ├── Writes JSON-RPC to stdin ──────▶ Server reads stdin
    │
    └── Reads stdout ◀────────────────── Server writes JSON-RPC to stdout
```

**Important**: When using stdio, logs must go to `stderr`, not `stdout`, to avoid corrupting the protocol stream.

---

## 8. FastMCP Installation and Setup

### Installation with pyproject.toml (Recommended)

Add FastMCP to your `pyproject.toml`:

```toml
[project]
name = "whatsapp-mcp"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "fastmcp>=2.0.0,<3",
    "pymilvus>=2.4.0",
    "sentence-transformers>=2.2.0",
]

[project.scripts]
whatsapp-mcp = "whatsapp_mcp.server:main"
```

Then install with uv:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
uv pip install -e .
```

### Alternative: Direct pip installation

```bash
# Basic installation
pip install fastmcp

# For development with all extras
pip install "fastmcp[dev]"

# Pin to v2 for stability (v3 in development)
pip install "fastmcp<3"
```

### Creating a Basic Server

```python
from fastmcp import FastMCP

# Create server instance
mcp = FastMCP(name="WhatsApp Search Server")

# Define a tool
@mcp.tool
def search_messages(query: str, limit: int = 10) -> list[dict]:
    """Search WhatsApp messages by semantic similarity.
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return
    
    Returns:
        List of matching messages with metadata
    """
    # Implementation here
    return results

# Define a resource
@mcp.resource("config://settings")
def get_settings() -> dict:
    """Provides server configuration."""
    return {"version": "1.0", "embedding_model": "multilingual-mpnet"}

# Run the server
if __name__ == "__main__":
    mcp.run()  # Default: stdio transport
```

### Running the Server

```bash
# Using Python directly (stdio transport)
python server.py

# Using FastMCP CLI
fastmcp run server.py:mcp

# With HTTP transport for remote access
fastmcp run server.py:mcp --transport http --port 8000

# Or in code:
mcp.run(transport="http", host="0.0.0.0", port=8000)
```

---

## 9. FastMCP Tool Definitions for Semantic Search

### WhatsApp Search Tool Examples

```python
from fastmcp import FastMCP
from datetime import datetime
from typing import Optional
from pymilvus import Collection

mcp = FastMCP(
    name="WhatsApp MCP Server",
    instructions="""
    This MCP server provides semantic search capabilities for WhatsApp conversations.
    Use the search tools to find messages by meaning, date range, or sender.
    """
)

# Initialize Milvus connection (done at startup)
collection: Collection = None

@mcp.tool
def semantic_search(
    query: str,
    limit: int = 10,
    chat_id: Optional[str] = None
) -> list[dict]:
    """Search WhatsApp messages using semantic similarity.
    
    Args:
        query: Natural language search query (e.g., "discussions about vacation plans")
        limit: Maximum number of results (1-50)
        chat_id: Optional chat ID to filter results
    
    Returns:
        List of matching messages with sender, timestamp, content, and relevance score
    """
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    
    # Build filter
    expr = f'chat_id == "{chat_id}"' if chat_id else None
    
    # Search Milvus
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=min(limit, 50),
        expr=expr,
        output_fields=["content", "sender", "timestamp", "chat_id"]
    )
    
    return [
        {
            "content": hit.entity.get("content"),
            "sender": hit.entity.get("sender"),
            "timestamp": datetime.fromtimestamp(hit.entity.get("timestamp")).isoformat(),
            "chat_id": hit.entity.get("chat_id"),
            "score": round(hit.score, 4)
        }
        for hit in results[0]
    ]

@mcp.tool
def search_by_date_range(
    start_date: str,
    end_date: str,
    query: Optional[str] = None,
    sender: Optional[str] = None,
    limit: int = 20
) -> list[dict]:
    """Search messages within a specific date range.
    
    Args:
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)
        query: Optional semantic search query
        sender: Optional sender name filter
        limit: Maximum results
    
    Returns:
        Messages matching the criteria
    """
    start_ts = int(datetime.fromisoformat(start_date).timestamp())
    end_ts = int(datetime.fromisoformat(end_date).timestamp())
    
    filters = [f"timestamp >= {start_ts}", f"timestamp <= {end_ts}"]
    if sender:
        filters.append(f'sender == "{sender}"')
    
    expr = " and ".join(filters)
    
    if query:
        # Semantic search with date filter
        query_embedding = model.encode(query).tolist()
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit,
            expr=expr,
            output_fields=["content", "sender", "timestamp"]
        )
        return format_results(results[0])
    else:
        # Query without semantic search
        results = collection.query(
            expr=expr,
            output_fields=["content", "sender", "timestamp"],
            limit=limit
        )
        return results

@mcp.tool
def list_chats() -> list[dict]:
    """List all available WhatsApp chats.
    
    Returns:
        List of chat IDs with message counts
    """
    # Query distinct chat_ids
    results = collection.query(
        expr="",
        output_fields=["chat_id"],
        limit=1000
    )
    
    # Count messages per chat
    chat_counts = {}
    for r in results:
        chat_id = r["chat_id"]
        chat_counts[chat_id] = chat_counts.get(chat_id, 0) + 1
    
    return [
        {"chat_id": cid, "message_count": count}
        for cid, count in chat_counts.items()
    ]

@mcp.tool
def get_conversation_context(
    message_timestamp: int,
    chat_id: str,
    context_messages: int = 5
) -> list[dict]:
    """Get surrounding messages for context.
    
    Args:
        message_timestamp: Unix timestamp of the target message
        chat_id: Chat ID containing the message
        context_messages: Number of messages before and after
    
    Returns:
        Messages surrounding the target timestamp
    """
    # Get messages around the timestamp
    results = collection.query(
        expr=f'chat_id == "{chat_id}" and timestamp >= {message_timestamp - 3600} and timestamp <= {message_timestamp + 3600}',
        output_fields=["content", "sender", "timestamp"],
        limit=context_messages * 2 + 1
    )
    
    # Sort by timestamp
    return sorted(results, key=lambda x: x["timestamp"])
```

### Resource Templates

```python
@mcp.resource("chats://{chat_id}/stats")
def get_chat_stats(chat_id: str) -> dict:
    """Get statistics for a specific chat."""
    results = collection.query(
        expr=f'chat_id == "{chat_id}"',
        output_fields=["sender", "timestamp"],
        limit=10000
    )
    
    senders = {}
    for r in results:
        sender = r["sender"]
        senders[sender] = senders.get(sender, 0) + 1
    
    timestamps = [r["timestamp"] for r in results]
    
    return {
        "chat_id": chat_id,
        "total_messages": len(results),
        "participants": list(senders.keys()),
        "messages_per_participant": senders,
        "date_range": {
            "first": datetime.fromtimestamp(min(timestamps)).isoformat(),
            "last": datetime.fromtimestamp(max(timestamps)).isoformat()
        }
    }
```

---

## 10. Docker Deployment for FastMCP

### Project Structure with pyproject.toml

```
whatsapp-mcp/
├── pyproject.toml          # Project configuration and dependencies
├── Dockerfile
├── docker-compose.yml
├── src/
│   └── whatsapp_mcp/
│       ├── __init__.py
│       └── server.py
└── data/
    └── *.zip               # WhatsApp chat exports
```

### pyproject.toml

```toml
[project]
name = "whatsapp-mcp"
version = "0.1.0"
description = "MCP server for semantic search of WhatsApp conversations"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]

dependencies = [
    "fastmcp>=2.0.0,<3",
    "pymilvus>=2.4.0",
    "sentence-transformers>=2.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
whatsapp-mcp = "whatsapp_mcp.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/whatsapp_mcp"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
```

### Dockerfile (using uv for fast builds)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install dependencies using uv (faster than pip)
RUN uv pip install --system -e .

# Environment variables
ENV MILVUS_HOST=milvus-standalone
ENV MILVUS_PORT=19530
ENV EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

# Expose port for HTTP transport
EXPOSE 8000

# Run with HTTP transport for Docker deployment
CMD ["python", "-m", "whatsapp_mcp.server", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
```

### Alternative Dockerfile (using pip)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install dependencies using pip with pyproject.toml
RUN pip install --no-cache-dir -e .

# Environment variables
ENV MILVUS_HOST=milvus-standalone
ENV MILVUS_PORT=19530
ENV EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

# Expose port for HTTP transport
EXPOSE 8000

# Run with HTTP transport for Docker deployment
CMD ["python", "-m", "whatsapp_mcp.server", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Integration

```yaml
version: '3.8'

services:
  # Existing Milvus services...
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    # ... (existing config)

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    # ... (existing config)

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    # ... (existing config)

  # FastMCP Server
  whatsapp-mcp:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: whatsapp-mcp-server
    environment:
      - MILVUS_HOST=milvus-standalone
      - MILVUS_PORT=19530
      - EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
    ports:
      - "8000:8000"
    depends_on:
      milvus:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - milvus

networks:
  milvus:
    name: milvus
```

### Server Entry Point (src/whatsapp_mcp/server.py)

```python
from fastmcp import FastMCP
import argparse
import os

mcp = FastMCP(
    name="WhatsApp MCP Server",
    instructions="""
    This MCP server provides semantic search capabilities for WhatsApp conversations.
    Use the search tools to find messages by meaning, date range, or sender.
    """
)

# ... tool definitions ...

def main():
    parser = argparse.ArgumentParser(description="WhatsApp MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type (default: stdio)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    
    args = parser.parse_args()
    
    if args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

---

## 11. Claude Desktop Integration

### Configuration File Location

| OS | Path |
|----|------|
| **macOS** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Windows** | `%APPDATA%\Claude\claude_desktop_config.json` |

### Configuration with uv and pyproject.toml (Recommended)

Using `uv` with `pyproject.toml` is the recommended approach for MCP servers. It ensures consistent dependency management and fast startup.

```json
{
  "mcpServers": {
    "whatsapp-search": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/whatsapp-mcp",
        "run",
        "whatsapp-mcp"
      ],
      "env": {
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": "19530"
      }
    }
  }
}
```

**Note**: The `whatsapp-mcp` command is defined in `pyproject.toml` under `[project.scripts]`.

### Alternative: Running the module directly with uv

```json
{
  "mcpServers": {
    "whatsapp-search": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/whatsapp-mcp",
        "run",
        "python",
        "-m",
        "whatsapp_mcp.server"
      ],
      "env": {
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": "19530"
      }
    }
  }
}
```

### Configuration for Docker (stdio via Docker)

```json
{
  "mcpServers": {
    "whatsapp-search": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--network", "milvus",
        "-e", "MILVUS_HOST=milvus-standalone",
        "-e", "MILVUS_PORT=19530",
        "whatsapp-mcp-server",
        "python", "-m", "whatsapp_mcp.server", "--transport", "stdio"
      ]
    }
  }
}
```

### Local Development Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create virtual environment and install dependencies**:
   ```bash
   cd /path/to/whatsapp-mcp
   uv venv
   uv pip install -e .
   ```

3. **Test the server locally**:
   ```bash
   uv run whatsapp-mcp
   ```

4. **Configure Claude Desktop** with the JSON configuration above

5. **Restart Claude Desktop** and verify the server is running

### Verifying the Integration

1. Restart Claude Desktop after configuration changes
2. Open Settings → Developer → MCP Servers
3. Verify "whatsapp-search" shows as "running"
4. Test with a query: "Search my WhatsApp messages for vacation plans"

---

## 12. Best Practices for MCP + Vector Database

### Tool Design Principles

1. **Clear Descriptions**: Write detailed docstrings - they become the tool's description for the LLM
2. **Type Hints**: Use Python type hints for automatic schema generation
3. **Sensible Defaults**: Provide defaults for optional parameters
4. **Error Handling**: Return structured error messages, not exceptions
5. **Pagination**: Support limit/offset for large result sets

### Performance Considerations

| Aspect | Recommendation |
|--------|----------------|
| **Connection Pooling** | Reuse Milvus connections across requests |
| **Embedding Caching** | Cache frequently used query embeddings |
| **Batch Operations** | Group multiple searches when possible |
| **Index Optimization** | Use HNSW index with appropriate `ef` parameter |
| **Result Limiting** | Always enforce maximum result limits |

### Security Considerations

```python
# Validate inputs
@mcp.tool
def search_messages(query: str, limit: int = 10) -> list[dict]:
    # Sanitize and validate
    if not query or len(query) > 1000:
        return {"error": "Invalid query length"}
    
    limit = max(1, min(limit, 100))  # Enforce bounds
    
    # ... rest of implementation
```

### Error Handling Pattern

```python
@mcp.tool
def search_messages(query: str) -> dict:
    """Search messages with proper error handling."""
    try:
        results = perform_search(query)
        return {"success": True, "results": results}
    except ConnectionError:
        return {"success": False, "error": "Database connection failed"}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {str(e)}"}
```

---

## References (FastMCP & MCP)

### Official Documentation
- [FastMCP Documentation](https://gofastmcp.com/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

### GitHub Repositories
- [FastMCP](https://github.com/jlowin/fastmcp)
- [MCP Server for Milvus](https://github.com/zilliztech/mcp-server-milvus)

### Tutorials
- [FastMCP Quickstart](https://gofastmcp.com/getting-started/quickstart)
- [Create an MCP Server Tutorial](https://gofastmcp.com/tutorials/create-mcp-server)
- [Claude Desktop MCP Integration](https://modelcontextprotocol.io/docs/develop/connect-local-servers)
