"""WhatsApp chat export parser.

This module provides functions to parse WhatsApp chat exports from ZIP files
and text content. It handles multiple formats including iOS, Android, and
French locale variations.
"""

import re
import zipfile
from datetime import datetime
from pathlib import Path

from whatsapp_mcp.models import Chat, Message

# Unicode characters commonly found in WhatsApp exports
# Left-to-right mark, right-to-left mark, object replacement character
UNICODE_CLEANUP_CHARS = "\u200e\u200f\ufeff\u202a\u202c"

# Regex patterns for different WhatsApp export formats
# iOS format: [DD/MM/YYYY, HH:MM:SS] Sender: Message
IOS_PATTERN = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.*)$"
)

# iOS format with AM/PM: [DD/MM/YYYY, HH:MM:SS AM/PM] Sender: Message
IOS_AMPM_PATTERN = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}:\d{2}\s*[APap][Mm])\]\s*([^:]+):\s*(.*)$"
)

# Android format: DD/MM/YYYY, HH:MM - Sender: Message
ANDROID_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2})\s*-\s*([^:]+):\s*(.*)$"
)

# Android format with AM/PM: DD/MM/YYYY, HH:MM AM/PM - Sender: Message
ANDROID_AMPM_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}\s*[APap][Mm])\s*-\s*([^:]+):\s*(.*)$"
)

# French locale format: DD/MM/YYYY à HH:MM - Sender: Message
FRENCH_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4})\s*[àa]\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.*)$"
)

# System message patterns (no sender, just timestamp and message)
IOS_SYSTEM_PATTERN = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}:\d{2})\]\s*([^:]+):\s*‎(.*)$"
)

# All patterns to try in order
MESSAGE_PATTERNS = [
    ("ios", IOS_PATTERN),
    ("ios_ampm", IOS_AMPM_PATTERN),
    ("android", ANDROID_PATTERN),
    ("android_ampm", ANDROID_AMPM_PATTERN),
    ("french", FRENCH_PATTERN),
]

# Media markers in different languages
MEDIA_MARKERS = [
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "document omitted",
    "gif omitted",
    "contact card omitted",
    "<media omitted>",
    "<image omitted>",
    "<video omitted>",
    "<audio omitted>",
    # French
    "image absente",
    "vidéo absente",
    "audio absent",
    "document absent",
]

# System message indicators
SYSTEM_INDICATORS = [
    "your security code",
    "security code changed",
    "you blocked",
    "you unblocked",
    "messages and calls are end-to-end encrypted",
    "created group",
    "added you",
    "removed you",
    "left the group",
    "changed the subject",
    "changed this group's icon",
    "changed the group description",
    # French
    "code de sécurité",
    "vous avez bloqué",
    "vous avez débloqué",
    "a créé le groupe",
    "vous a ajouté",
    "vous a retiré",
    "a quitté le groupe",
]

# Deleted message indicators
DELETED_INDICATORS = [
    "this message was deleted",
    "you deleted this message",
    "ce message a été supprimé",
    "vous avez supprimé ce message",
]

# Edited message indicator
EDITED_INDICATOR = "<this message was edited>"


def _clean_text(text: str) -> str:
    """Remove special Unicode characters from text.

    Args:
        text: The text to clean.

    Returns:
        Cleaned text with special characters removed.
    """
    for char in UNICODE_CLEANUP_CHARS:
        text = text.replace(char, "")
    return text.strip()


def _parse_datetime(date_str: str, time_str: str) -> datetime:
    """Parse date and time strings into a datetime object.

    Handles multiple date formats:
    - DD/MM/YYYY or DD/MM/YY
    - MM/DD/YYYY or MM/DD/YY (US format)
    - With or without AM/PM

    Args:
        date_str: Date string (e.g., "18/03/2025" or "03/18/2025").
        time_str: Time string (e.g., "07:37:36" or "7:37 AM").

    Returns:
        Parsed datetime object.

    Raises:
        ValueError: If the date/time format is not recognized.
    """
    # Clean the strings
    date_str = date_str.strip()
    time_str = time_str.strip()

    # Parse date - try DD/MM/YYYY first (European format)
    date_parts = date_str.split("/")
    if len(date_parts) != 3:
        raise ValueError(f"Invalid date format: {date_str}")

    day, month, year = date_parts

    # Handle 2-digit years
    if len(year) == 2:
        year = "20" + year

    # Parse time
    time_str_upper = time_str.upper()
    has_ampm = "AM" in time_str_upper or "PM" in time_str_upper

    if has_ampm:
        # Remove AM/PM and parse
        is_pm = "PM" in time_str_upper
        time_clean = re.sub(r"\s*[APap][Mm]\s*", "", time_str).strip()
        time_parts = time_clean.split(":")

        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2]) if len(time_parts) > 2 else 0

        # Convert to 24-hour format
        if is_pm and hour != 12:
            hour += 12
        elif not is_pm and hour == 12:
            hour = 0
    else:
        time_parts = time_str.split(":")
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2]) if len(time_parts) > 2 else 0

    return datetime(
        year=int(year),
        month=int(month),
        day=int(day),
        hour=hour,
        minute=minute,
        second=second,
    )


def _is_media_message(content: str) -> bool:
    """Check if a message content indicates media.

    Args:
        content: The message content to check.

    Returns:
        True if the message is a media placeholder.
    """
    content_lower = content.lower()
    return any(marker in content_lower for marker in MEDIA_MARKERS)


def _is_system_message(content: str, sender: str) -> bool:
    """Check if a message is a system message.

    Args:
        content: The message content to check.
        sender: The sender name.

    Returns:
        True if the message is a system message.
    """
    content_lower = content.lower()

    # Check for system indicators in content
    if any(indicator in content_lower for indicator in SYSTEM_INDICATORS):
        return True

    # Check if sender is system-like
    sender_lower = sender.lower()
    if sender_lower in ["system", "whatsapp"]:
        return True

    return False


def _is_deleted_message(content: str) -> bool:
    """Check if a message was deleted.

    Args:
        content: The message content to check.

    Returns:
        True if the message was deleted.
    """
    content_lower = content.lower()
    return any(indicator in content_lower for indicator in DELETED_INDICATORS)


def _is_edited_message(content: str) -> bool:
    """Check if a message was edited.

    Args:
        content: The message content to check.

    Returns:
        True if the message was edited.
    """
    return EDITED_INDICATOR in content.lower()


def _parse_line(line: str) -> tuple[str, str, str, str] | None:
    """Parse a single line from a WhatsApp export.

    Args:
        line: A line from the chat export.

    Returns:
        Tuple of (date_str, time_str, sender, content) if the line matches
        a message pattern, None otherwise.
    """
    line = _clean_text(line)

    for _format_name, pattern in MESSAGE_PATTERNS:
        match = pattern.match(line)
        if match:
            date_str, time_str, sender, content = match.groups()
            return date_str, time_str, sender.strip(), content.strip()

    return None


def _generate_chat_id(chat_name: str) -> str:
    """Generate a chat ID from the chat name.

    Args:
        chat_name: The human-readable chat name.

    Returns:
        A normalized chat ID suitable for use as an identifier.
    """
    # Convert to lowercase, replace spaces and special chars with underscores
    chat_id = chat_name.lower()
    chat_id = re.sub(r"[^a-z0-9]+", "_", chat_id)
    chat_id = chat_id.strip("_")
    return chat_id


def parse_whatsapp_text(text: str, chat_id: str, chat_name: str) -> list[Message]:
    """Parse WhatsApp chat text content into a list of messages.

    This function handles multi-line messages by detecting when a line
    doesn't match the message pattern and appending it to the previous message.

    Args:
        text: The raw text content of a WhatsApp chat export.
        chat_id: Unique identifier for this chat.
        chat_name: Human-readable name of the chat.

    Returns:
        List of Message objects parsed from the text.

    Example:
        >>> text = '''[18/03/2025, 07:37:36] John: Hello
        ... [18/03/2025, 07:38:00] Jane: Hi there!
        ... How are you?'''
        >>> messages = parse_whatsapp_text(text, "john_jane", "John & Jane")
        >>> len(messages)
        2
        >>> messages[1].content
        'Hi there!\\nHow are you?'
    """
    messages: list[Message] = []
    current_message_data: dict | None = None

    for line in text.splitlines():
        # Skip empty lines at the start
        if not line.strip() and current_message_data is None:
            continue

        parsed = _parse_line(line)

        if parsed:
            # Save the previous message if exists
            if current_message_data:
                messages.append(Message(**current_message_data))

            date_str, time_str, sender, content = parsed

            # Remove edited indicator from content but track it
            is_edited = _is_edited_message(content)
            if is_edited:
                content = re.sub(
                    r"\s*‎?<this message was edited>\s*",
                    "",
                    content,
                    flags=re.IGNORECASE,
                ).strip()

            try:
                timestamp = _parse_datetime(date_str, time_str)
            except ValueError:
                # If date parsing fails, skip this message
                continue

            current_message_data = {
                "content": content,
                "sender": sender,
                "timestamp": timestamp,
                "chat_id": chat_id,
                "chat_name": chat_name,
                "is_media": _is_media_message(content),
                "is_system": _is_system_message(content, sender),
                "is_deleted": _is_deleted_message(content),
                "is_edited": is_edited,
            }
        elif current_message_data and line.strip():
            # This is a continuation of the previous message (multi-line)
            cleaned_line = _clean_text(line)
            if cleaned_line:
                current_message_data["content"] += "\n" + cleaned_line

    # Don't forget the last message
    if current_message_data:
        messages.append(Message(**current_message_data))

    return messages


def parse_whatsapp_zip(zip_path: Path | str) -> Chat:
    """Parse a WhatsApp chat export ZIP file.

    WhatsApp exports chats as ZIP files containing a `_chat.txt` file
    and optionally media files. This function extracts and parses the
    text content.

    Args:
        zip_path: Path to the WhatsApp chat export ZIP file.

    Returns:
        Chat object containing all parsed messages.

    Raises:
        FileNotFoundError: If the ZIP file doesn't exist.
        ValueError: If the ZIP file doesn't contain a chat text file.

    Example:
        >>> chat = parse_whatsapp_zip("WhatsApp Chat - John Doe.zip")
        >>> print(f"Chat with {chat.chat_name}: {chat.message_count} messages")
        Chat with John Doe: 150 messages
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Extract chat name from ZIP filename
    # Format: "WhatsApp Chat - Contact Name.zip"
    filename = zip_path.stem  # Remove .zip extension
    if filename.startswith("WhatsApp Chat - "):
        chat_name = filename[16:]  # Remove "WhatsApp Chat - " prefix
    else:
        chat_name = filename

    chat_id = _generate_chat_id(chat_name)

    # Open and read the ZIP file
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the chat text file
        chat_file = None
        for name in zf.namelist():
            if name.endswith(".txt"):
                chat_file = name
                break

        if not chat_file:
            raise ValueError(f"No text file found in ZIP: {zip_path}")

        # Read and decode the chat content
        with zf.open(chat_file) as f:
            # Try UTF-8 first, then fall back to other encodings
            content = f.read()
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = content.decode("utf-8-sig")  # UTF-8 with BOM
                except UnicodeDecodeError:
                    text = content.decode("latin-1")  # Fallback

    # Parse the text content
    messages = parse_whatsapp_text(text, chat_id, chat_name)

    return Chat(
        chat_id=chat_id,
        chat_name=chat_name,
        messages=messages,
    )


def parse_whatsapp_file(file_path: Path | str) -> Chat:
    """Parse a WhatsApp chat export file (ZIP or TXT).

    This is a convenience function that automatically detects the file type
    and calls the appropriate parser.

    Args:
        file_path: Path to the WhatsApp chat export file.

    Returns:
        Chat object containing all parsed messages.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file type is not supported.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".zip":
        return parse_whatsapp_zip(file_path)
    elif file_path.suffix.lower() == ".txt":
        # For text files, derive chat name from filename
        chat_name = file_path.stem
        if chat_name.startswith("WhatsApp Chat - "):
            chat_name = chat_name[16:]
        chat_id = _generate_chat_id(chat_name)

        text = file_path.read_text(encoding="utf-8")
        messages = parse_whatsapp_text(text, chat_id, chat_name)

        return Chat(
            chat_id=chat_id,
            chat_name=chat_name,
            messages=messages,
        )
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
