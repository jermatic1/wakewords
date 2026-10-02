"""Download helpers shared by the data setup steps."""

import logging
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

log = logging.getLogger(__name__)


def download_url(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)
    return dest


def download_hf_file(repo_id: str, filename: str, dest_dir: Path) -> Path:
    """Downloads one file from a Hugging Face dataset repo, resuming if needed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=dest_dir,
    )
    return Path(path)
