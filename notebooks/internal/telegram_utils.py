"""
Utilities for sending messages and files to a Telegram chat using a bot.

Requires the TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables to be set.
"""
import os
import requests

def send_message(
        markdown_message: str,
        raise_on_failure: bool = True
) -> dict:
    """
    Send a message to a Telegram chat using a bot.

    Args:
        markdown_message (str): The message to send.
        raise_on_failure (bool): Whether to raise an exception on failure.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        if raise_on_failure:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment variables.")
        return {}

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": markdown_message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        if raise_on_failure:
            raise Exception(f"Failed to send message: {response.text}")
        return {}

    return response.json()

def send_file(
        file_path: str,
        caption: str,
        raise_on_failure: bool = True
) -> dict:
    """
    Send a file to a Telegram chat using a bot.

    Args:
        caption (str): The caption message to send with the file.
        file_path (str): The path to the file to send.
        raise_on_failure (bool): Whether to raise an exception on failure.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        if raise_on_failure:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment variables.")
        return {}

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    with open(file_path, "rb") as file:
        files = {"document": file}
        data = {
            "chat_id": chat_id,
            "caption": caption,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=data, files=files)

    if response.status_code != 200:
        if raise_on_failure:
            raise Exception(f"Failed to send file: {response.text}")
        return {}

    return response.json()

def send_files(
        file_paths: list[str],
        caption: str,
        raise_on_failure: bool = True
) -> dict:
    """
    Send multiple files to a Telegram chat using a bot.

    Args:
        caption (str): The caption message to send with the files.
        file_paths (list[str]): A list of paths to the files to send.
        raise_on_failure (bool): Whether to raise an exception on failure.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        if raise_on_failure:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment variables.")
        return {}

    url = f"https://api.telegram.org/bot{bot_token}/sendMediaGroup"
    media = []
    files = {}

    for idx, file_path in enumerate(file_paths):
        file_key = f"file{idx}"
        media.append({
            "type": "document",
            "media": f"attach://{file_key}",
            "caption": caption if idx == 0 else ""
        })
        files[file_key] = open(file_path, "rb")

    data = {
        "chat_id": chat_id,
        "media": str(media).replace("'", '"')  # Telegram API requires double quotes in JSON
    }

    response = requests.post(url, data=data, files=files)

    for file in files.values():
        file.close()

    if response.status_code != 200:
        if raise_on_failure:
            raise Exception(f"Failed to send files: {response.text}")
        return {}

    return response.json()
