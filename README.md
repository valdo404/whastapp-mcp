# WhatsApp MCP

MCP server for semantic search of WhatsApp conversations using vector embeddings and Milvus.

## Features

- **Semantic Search**: Find messages by meaning, not just keywords
- **Multi-language Support**: Works with any language using multilingual embeddings
- **Date & Sender Filtering**: Narrow down searches by time range or participant
- **Claude Desktop Integration**: Use as an MCP server with Claude Desktop
- **CLI Tools**: Command-line interface for data ingestion and search

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Claude Desktop │────▶│  WhatsApp MCP    │────▶│     Milvus      │
│  (MCP Client)   │     │  (FastMCP Server)│     │  (Vector DB)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ Sentence         │
                        │ Transformers     │
                        │ (Embeddings)     │
                        └──────────────────┘
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/laurentvaldes/whatsapp-mcp.git
cd whatsapp-mcp
```

### 2. Create Virtual Environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate

# Or using Python
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Using uv
uv pip install -e ".[server]"

# Or using pip
pip install -e ".[server]"
```

### 4. Start Infrastructure

```bash
docker compose up -d
```

This starts:
- **Milvus** (port 19530): Vector database for storing embeddings
- **etcd**: Metadata storage for Milvus
- **MinIO** (ports 9000-9001): Object storage for Milvus

Verify containers are running:
```bash
docker compose ps
```

## Usage

### Export WhatsApp Conversations

1. Open WhatsApp on your phone
2. Go to a conversation → Settings → Export Chat
3. Choose "Without Media" (recommended for faster processing)
4. Save the ZIP file to the `data/` directory

### Ingest Data

```bash
# Ingest all ZIP files from a directory
whatsapp-mcp-cli ingest data/

# Ingest a specific file
whatsapp-mcp-cli ingest data/WhatsApp\ Chat\ -\ John.zip
```

### CLI Commands

```bash
# View collection statistics
whatsapp-mcp-cli stats

# List all indexed conversations
whatsapp-mcp-cli list-chats

# Semantic search
whatsapp-mcp-cli search "vacation plans"

# Search with filters
whatsapp-mcp-cli search "birthday party" --chat-id john_doe --limit 20

# Drop all data (use with caution!)
whatsapp-mcp-cli drop --yes
```

### Run MCP Server

```bash
# Start with stdio transport (for Claude Desktop)
whatsapp-mcp

# Start with HTTP transport (for development/testing)
whatsapp-mcp --transport http --port 8000
```

## Claude Desktop Integration

### Configuration

Add the following to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "whatsapp": {
      "command": "/path/to/whatsapp-mcp/.venv/bin/whatsapp-mcp",
      "args": [],
      "env": {
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": "19530"
      }
    }
  }
}
```

Replace `/path/to/whatsapp-mcp` with the actual path to your installation.

### Available Tools

Once configured, Claude can use these tools:

| Tool | Description |
|------|-------------|
| `semantic_search` | Find messages by meaning using natural language |
| `search_by_date` | Search within specific date ranges |
| `search_by_sender` | Filter messages by participant |
| `list_chats` | View all available conversations |
| `get_chat_stats` | Get statistics about indexed data |

### Example Prompts

- "Search my WhatsApp messages for discussions about vacation plans"
- "Find messages from January 2025 about the birthday party"
- "What did John say about the project deadline?"
- "Show me all conversations I have indexed"

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `localhost` | Milvus server hostname |
| `MILVUS_PORT` | `19530` | Milvus server port |

## Troubleshooting

### Milvus Connection Failed

```
Failed to connect to Milvus: ...
```

**Solution**: Ensure Docker containers are running:
```bash
docker compose ps
docker compose up -d
```

### No Messages Found

```
Collection Statistics: 0 entities
```

**Solution**: Ingest your WhatsApp exports:
```bash
whatsapp-mcp-cli ingest data/
```

### Slow First Search

The first search may take 10-30 seconds as the embedding model loads into memory. Subsequent searches are much faster.

### Memory Issues

If you encounter memory issues with large datasets:
1. Reduce batch size during ingestion
2. Ensure Docker has sufficient memory allocated (4GB+ recommended)

### Claude Desktop Not Connecting

1. Verify the path in `claude_desktop_config.json` is correct
2. Ensure the virtual environment is activated in the command
3. Check Claude Desktop logs for errors
4. Restart Claude Desktop after configuration changes

## Development

### Run Tests

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/

# Run type checker
mypy src/
```

### Project Structure

```
whatsapp-mcp/
├── src/whatsapp_mcp/
│   ├── __init__.py
│   ├── cli.py           # CLI commands
│   ├── embeddings.py    # Embedding service
│   ├── milvus_client.py # Vector database client
│   ├── models.py        # Data models
│   ├── parser.py        # WhatsApp export parser
│   └── server.py        # MCP server
├── data/                # WhatsApp exports (gitignored)
├── docs/                # Documentation
├── docker-compose.yml   # Infrastructure
└── pyproject.toml       # Project configuration
```

## License

MIT
