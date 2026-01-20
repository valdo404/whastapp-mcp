# WhatsApp MCP

MCP server for semantic search of WhatsApp conversations.

## Installation

```bash
# Install with uv
uv pip install -e ".[server]"
```

## Usage

### Start Milvus Infrastructure

```bash
docker compose up -d
```

### Ingest WhatsApp Exports

```bash
whatsapp-mcp-cli ingest data/
```

### Search Messages

```bash
whatsapp-mcp-cli search "vacation plans"
```

## License

MIT
