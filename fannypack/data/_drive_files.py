import os
import signal

import requests
from tqdm.auto import tqdm

_cache_path = os.path.expanduser("~/.cache/fannypack-drive-files")


def set_cache_path(path: str):
    """Set the cache location for :func:`fannypack.data.cached_drive_file`.

    Args:
        _cache_path (str): New location for cached files. Defaults to
            `~/.cache/fannypack-drive-files/`.
    """
    global _cache_path
    _cache_path = path


def cached_drive_file(name: str, url: str) -> str:
    """Return a local path to a file from Google Drive. Downloads the file if it doesn't
    exist yet locally.

    By default, cached files live in `~/.cache/fannypack-drive-files/`. It often makes
    sense to move this directory (eg to an NFS): see
    :func:`fannypack.data.set_cache_path`.

    Args:
        name (str): Name of path, eg `secret_key.pem`.
        url (str): URL, eg
            `https://drive.google.com/file/d/1AsY9Cs3xE0RSlr0FKlnSKHp6zIwFSvXe/view`.

    Returns:
        str: Local path to file.
    """

    # Generated cached filename
    cached_filename = f"{_drive_id_from_url(url)}-{name}"
    cached_filepath = os.path.join(_cache_path, cached_filename)

    if not os.path.exists(cached_filepath):
        print(f"[fannypack-drive] Downloading file to {cached_filepath}")
        download_drive_file(url, cached_filepath)
        assert os.path.exists(cached_filepath)
    return cached_filepath


def download_drive_file(url: str, target_path: str, chunk_size=32768) -> None:
    """Download a file via a public Google Drive url.

    Example usage:
    ```
    download_file_from_google_drive(
        "https://drive.google.com/file/d/1AsY9Cs3xE0RSlr0FKlnSKHp6zIwFSvXe/view",
        "/home/brent/Downloads/test.pdf"
    )
    ```

    Args:
        url (str): Google Drive url.
        target_path (str): Destination to write to.
    """
    # Create directory if it doesn't exist yet
    directory = os.path.dirname(target_path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print("[fannypack-drive] Created directory:", directory)

    # Parse URL
    drive_id = _drive_id_from_url(url)
    download_url = "https://docs.google.com/uc?export=download"

    # Download file
    session = requests.Session()
    response = session.get(
        download_url,
        params={"id": drive_id},
        stream=True,
        headers={"Accept-Encoding": None},
    )

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):  # pragma: no cover
            params = {"id": drive_id, "confirm": value}
            response = session.get(download_url, params=params, stream=True)
            break

    # Delete partially downloaded files if we hit interrupt (Ctrl+C) before download
    # finishes
    try:
        orig_handler = signal.getsignal(signal.SIGINT)

        def sigint_handler(sig, frame):  # pragma: no cover
            print("[fannypack-drive] Deleting file:", target_path)
            os.remove(target_path)
            orig_handler(sig, frame)
            # Restore SIGINT handler
            if orig_handler is not None:
                signal.signal(signal.SIGINT, orig_handler)

        signal.signal(signal.SIGINT, sigint_handler)
    except ValueError as e:  # pragma: no cover
        # signal throws a ValueError if we're not in the main thread
        print("[fannypack-drive] Error while attaching SIGINT handler:", e)
        orig_handler = None

    # Download file
    progress_bar = tqdm(unit="iB", unit_scale=True)
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            # Filter out keep-alive new chunks
            if chunk:
                progress_bar.update(len(chunk))
                f.write(chunk)

    # Restore SIGINT handler
    if orig_handler is not None:
        signal.signal(signal.SIGINT, orig_handler)


def _drive_id_from_url(url: str) -> str:
    """Get an ID from a Google Drive URL.

    Args:
        url (str): Publicly visible Google Drive URL.

    Returns:
        str: Google Drive ID.
    """

    url_prefixes = [
        "https://drive.google.com/file/d/",
        "https://drive.google.com/open?id=",
    ]

    for prefix in url_prefixes:
        if url.startswith(prefix):
            parts = url[len(prefix) :].split("/")
            assert len(parts) in (1, 2)
            return parts[0]
    assert False, f"Malformed Google Drive URL: {url}"
