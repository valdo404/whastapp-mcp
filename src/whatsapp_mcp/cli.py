"""CLI for WhatsApp MCP operations.

This module provides command-line tools for ingesting WhatsApp chat exports
into the Milvus vector database.
"""

import logging
import sys
from pathlib import Path

import click

from whatsapp_mcp.embeddings import EmbeddingService
from whatsapp_mcp.milvus_client import MilvusClient
from whatsapp_mcp.parser import parse_whatsapp_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="whatsapp-mcp")
def cli() -> None:
    """WhatsApp MCP CLI - Tools for managing WhatsApp chat data."""
    pass


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--batch-size",
    default=100,
    help="Number of messages to process in each batch.",
    show_default=True,
)
@click.option(
    "--milvus-host",
    default=None,
    envvar="MILVUS_HOST",
    help="Milvus server hostname.",
)
@click.option(
    "--milvus-port",
    default=None,
    type=int,
    envvar="MILVUS_PORT",
    help="Milvus server port.",
)
@click.option(
    "--model",
    default=None,
    envvar="EMBEDDING_MODEL",
    help="Sentence-transformers model name.",
)
@click.option(
    "--skip-media/--include-media",
    default=True,
    help="Skip media placeholder messages.",
)
@click.option(
    "--skip-system/--include-system",
    default=True,
    help="Skip system messages.",
)
@click.option(
    "--skip-deleted/--include-deleted",
    default=True,
    help="Skip deleted messages.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Parse files without inserting into Milvus.",
)
def ingest(
    data_dir: str,
    batch_size: int,
    milvus_host: str | None,
    milvus_port: int | None,
    model: str | None,
    skip_media: bool,
    skip_system: bool,
    skip_deleted: bool,
    dry_run: bool,
) -> None:
    """Ingest WhatsApp chat exports into Milvus.

    DATA_DIR: Directory containing WhatsApp chat export files (.zip or .txt)

    Example:
        whatsapp-mcp-cli ingest ./data/
        whatsapp-mcp-cli ingest ./data/ --batch-size 50
        whatsapp-mcp-cli ingest ./data/ --dry-run
    """
    data_path = Path(data_dir)

    # Find all chat files
    zip_files = list(data_path.glob("*.zip"))
    txt_files = list(data_path.glob("*.txt"))
    all_files = zip_files + txt_files

    if not all_files:
        click.echo(f"No WhatsApp export files found in {data_dir}", err=True)
        click.echo("Looking for .zip or .txt files", err=True)
        sys.exit(1)

    click.echo(f"Found {len(all_files)} file(s) to process:")
    for f in all_files:
        click.echo(f"  - {f.name}")

    # Initialize services
    click.echo("\nInitializing services...")

    embedding_service = EmbeddingService(model_name=model)
    click.echo(f"  Embedding model: {embedding_service.model_name}")

    if not dry_run:
        milvus_client = MilvusClient(host=milvus_host, port=milvus_port)
        click.echo(f"  Milvus: {milvus_client.host}:{milvus_client.port}")

        try:
            milvus_client.ensure_collection(dimension=embedding_service.dimension)
            click.echo(f"  Collection: {milvus_client.collection_name}")
        except ConnectionError as e:
            click.echo(f"\nError connecting to Milvus: {e}", err=True)
            click.echo("Make sure Milvus is running: docker compose up -d", err=True)
            sys.exit(1)
    else:
        click.echo("  Milvus: [DRY RUN - not connecting]")
        milvus_client = None

    total_messages = 0
    total_skipped = 0
    total_inserted = 0

    for file_path in all_files:
        click.echo(f"\nProcessing: {file_path.name}")

        try:
            chat = parse_whatsapp_file(file_path)
        except Exception as e:
            click.echo(f"  Error parsing file: {e}", err=True)
            continue

        click.echo(f"  Chat: {chat.chat_name}")
        click.echo(f"  Total messages: {chat.message_count}")

        # Filter messages
        messages = chat.messages
        original_count = len(messages)

        if skip_media:
            messages = [m for m in messages if not m.is_media]
        if skip_system:
            messages = [m for m in messages if not m.is_system]
        if skip_deleted:
            messages = [m for m in messages if not m.is_deleted]

        skipped = original_count - len(messages)
        total_skipped += skipped

        if skipped > 0:
            click.echo(f"  Skipped: {skipped} (media/system/deleted)")

        click.echo(f"  Messages to ingest: {len(messages)}")

        if not messages:
            click.echo("  No messages to ingest after filtering")
            continue

        total_messages += len(messages)

        if dry_run:
            click.echo("  [DRY RUN - skipping insertion]")
            continue

        # Process in batches
        num_batches = (len(messages) + batch_size - 1) // batch_size

        with click.progressbar(
            range(0, len(messages), batch_size),
            label="  Ingesting",
            length=num_batches,
        ) as batch_starts:
            for i in batch_starts:
                batch = messages[i : i + batch_size]

                # Generate embeddings
                contents = [m.content for m in batch]
                embeddings = embedding_service.encode_batch(
                    contents,
                    batch_size=batch_size,
                    show_progress=False,
                )

                # Insert into Milvus
                if milvus_client is not None:
                    inserted = milvus_client.insert(batch, embeddings)
                    total_inserted += inserted

        # Flush after each file
        if milvus_client is not None:
            milvus_client.flush()

    # Summary
    click.echo("\n" + "=" * 50)
    click.echo("Ingestion Summary")
    click.echo("=" * 50)
    click.echo(f"Files processed: {len(all_files)}")
    click.echo(f"Total messages found: {total_messages + total_skipped}")
    click.echo(f"Messages skipped: {total_skipped}")
    click.echo(f"Messages processed: {total_messages}")

    if not dry_run:
        click.echo(f"Messages inserted: {total_inserted}")
    else:
        click.echo("[DRY RUN - no data inserted]")


@cli.command()
@click.option(
    "--milvus-host",
    default=None,
    envvar="MILVUS_HOST",
    help="Milvus server hostname.",
)
@click.option(
    "--milvus-port",
    default=None,
    type=int,
    envvar="MILVUS_PORT",
    help="Milvus server port.",
)
def stats(milvus_host: str | None, milvus_port: int | None) -> None:
    """Show statistics about the Milvus collection."""
    milvus_client = MilvusClient(host=milvus_host, port=milvus_port)

    try:
        stats_data = milvus_client.get_collection_stats()
    except ConnectionError as e:
        click.echo(f"Error connecting to Milvus: {e}", err=True)
        sys.exit(1)

    click.echo("Collection Statistics")
    click.echo("=" * 50)
    click.echo(f"Collection: {stats_data['collection_name']}")
    click.echo(f"Exists: {stats_data['exists']}")

    if stats_data["exists"]:
        click.echo(f"Number of entities: {stats_data['num_entities']}")


@cli.command()
@click.option(
    "--milvus-host",
    default=None,
    envvar="MILVUS_HOST",
    help="Milvus server hostname.",
)
@click.option(
    "--milvus-port",
    default=None,
    type=int,
    envvar="MILVUS_PORT",
    help="Milvus server port.",
)
def list_chats(milvus_host: str | None, milvus_port: int | None) -> None:
    """List all chats in the collection."""
    milvus_client = MilvusClient(host=milvus_host, port=milvus_port)

    try:
        chats = milvus_client.list_chats()
    except ConnectionError as e:
        click.echo(f"Error connecting to Milvus: {e}", err=True)
        sys.exit(1)

    if not chats:
        click.echo("No chats found in the collection.")
        return

    click.echo("Available Chats")
    click.echo("=" * 50)

    for chat in chats:
        click.echo(f"\n{chat['chat_name']} ({chat['chat_id']})")
        click.echo(f"  Messages: {chat['message_count']}")
        click.echo(f"  Participants: {', '.join(chat['participants'])}")
        click.echo(f"  Date range: {chat['date_range']['first']} to {chat['date_range']['last']}")


@cli.command()
@click.argument("query")
@click.option(
    "--limit",
    default=10,
    help="Maximum number of results.",
    show_default=True,
)
@click.option(
    "--chat-id",
    default=None,
    help="Filter by chat ID.",
)
@click.option(
    "--milvus-host",
    default=None,
    envvar="MILVUS_HOST",
    help="Milvus server hostname.",
)
@click.option(
    "--milvus-port",
    default=None,
    type=int,
    envvar="MILVUS_PORT",
    help="Milvus server port.",
)
@click.option(
    "--model",
    default=None,
    envvar="EMBEDDING_MODEL",
    help="Sentence-transformers model name.",
)
def search(
    query: str,
    limit: int,
    chat_id: str | None,
    milvus_host: str | None,
    milvus_port: int | None,
    model: str | None,
) -> None:
    """Search messages using semantic similarity.

    QUERY: Natural language search query

    Example:
        whatsapp-mcp-cli search "vacation plans"
        whatsapp-mcp-cli search "birthday party" --limit 5
    """
    from datetime import datetime

    embedding_service = EmbeddingService(model_name=model)
    milvus_client = MilvusClient(host=milvus_host, port=milvus_port)

    try:
        milvus_client.load_collection()
    except ConnectionError as e:
        click.echo(f"Error connecting to Milvus: {e}", err=True)
        sys.exit(1)

    # Generate query embedding
    click.echo(f"Searching for: {query}")
    query_embedding = embedding_service.encode(query)

    # Search
    results = milvus_client.search(
        query_embedding=query_embedding,
        limit=limit,
        chat_id=chat_id,
    )

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nFound {len(results)} result(s):\n")

    for i, result in enumerate(results, 1):
        timestamp = datetime.fromtimestamp(result["timestamp"])
        click.echo(f"{i}. [{result['score']:.4f}] {result['chat_name']}")
        click.echo(f"   {result['sender']} - {timestamp.strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"   {result['content'][:100]}{'...' if len(result['content']) > 100 else ''}")
        click.echo()


@cli.command()
@click.option(
    "--milvus-host",
    default=None,
    envvar="MILVUS_HOST",
    help="Milvus server hostname.",
)
@click.option(
    "--milvus-port",
    default=None,
    type=int,
    envvar="MILVUS_PORT",
    help="Milvus server port.",
)
@click.confirmation_option(prompt="Are you sure you want to drop the collection?")
def drop(milvus_host: str | None, milvus_port: int | None) -> None:
    """Drop the Milvus collection (delete all data).

    WARNING: This operation is irreversible!
    """
    milvus_client = MilvusClient(host=milvus_host, port=milvus_port)

    try:
        milvus_client.drop_collection()
        click.echo("Collection dropped successfully.")
    except ConnectionError as e:
        click.echo(f"Error connecting to Milvus: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
