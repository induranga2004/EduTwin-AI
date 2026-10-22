"""Download helper and licence manifest (``DATA_LICENCES.md``) for the public text corpora."""

import hashlib
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


class DownloadError(OSError):
    """A download failed or would overwrite an existing file."""


@dataclass(frozen=True)
class DataSource:
    key: str
    name: str
    filename: str
    url: str
    licence: str
    licence_url: str
    restrictions: str
    target_semantics: str


SOURCES = (
    DataSource(
        key="clear",
        name="CLEAR corpus (CommonLit Ease of Readability)",
        filename="CLEAR_corpus_final.xlsx",
        url="https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/main/CLEAR_corpus_final.xlsx",
        licence="CC BY-NC-SA 4.0",
        licence_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        restrictions=(
            "Research and non-commercial use only; attribution and share-alike apply. The data "
            "is never committed, never copied into images and never redistributed; weights "
            "trained on it carry the non-commercial restriction (state it in the model card)."
        ),
        target_semantics="BT_easiness (Bradley-Terry ease, teacher judgements); higher = easier",
    ),
    DataSource(
        key="onestop",
        name="OneStopEnglish corpus",
        filename="onestop.zip",
        url="https://codeload.github.com/nishkalavallabhi/OneStopEnglishCorpus/zip/refs/heads/master",
        licence="CC BY-SA 4.0",
        licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
        restrictions=(
            "Attribution and share-alike apply. The data is never committed and never copied "
            "into images; derived artefacts must keep the attribution."
        ),
        target_semantics="reading level Ele=0, Int=1, Adv=2 (three versions of each article); "
        "higher = harder",
    ),
)

LICENCE_REMINDER = (
    "Licence reminder: CLEAR is CC BY-NC-SA 4.0 (research and non-commercial use, no "
    "redistribution); OneStopEnglish is CC BY-SA 4.0. Keep the files in the git-ignored "
    "data/public folder."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_url(url: str, timeout: float = 120.0) -> bytes:
    """Plain HTTPS GET (stdlib only)."""
    request = urllib.request.Request(url, headers={"User-Agent": "edusync-c2-model/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download(
    source: DataSource,
    folder: Path,
    *,
    force: bool = False,
    fetch: Callable[[str], bytes] = fetch_url,
) -> Path:
    """Fetch ``source`` into ``folder``; refuses to overwrite unless ``force``."""
    dest = Path(folder) / source.filename
    if dest.exists() and not force:
        raise DownloadError(f"{dest} exists; pass --force to download it again")
    try:
        data = fetch(source.url)
    except OSError as exc:
        raise DownloadError(f"download of {source.url} failed: {type(exc).__name__}") from exc
    if not data:
        raise DownloadError(f"download of {source.url} returned no data")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def render_licences(
    sources: Sequence[DataSource],
    folder: Path,
    row_counts: dict[str, int],
    access_date: str,
) -> str:
    """Markdown for ``DATA_LICENCES.md``; sha256 is read from the files in ``folder``."""
    lines = [
        "# Public data licences",
        "",
        "The text models are trained and evaluated on public corpora with human readability",
        "judgements. The files are downloaded by `scripts/download_public_data.py` into the",
        "git-ignored `data/public/` folder and are never committed, copied into images or",
        "redistributed. They are general English, not technical lecture material.",
        "",
    ]
    for source in sources:
        path = Path(folder) / source.filename
        lines += [
            f"## {source.name}",
            "",
            f"- File: `{source.filename}`",
            f"- Source URL: {source.url}",
            f"- Licence: {source.licence} ({source.licence_url})",
            f"- Accessed: {access_date}",
            f"- SHA-256: `{sha256_file(path)}`",
            f"- Rows used: {row_counts[source.key]}",
            f"- Target: {source.target_semantics}",
            f"- Usage restrictions: {source.restrictions}",
            "",
        ]
    return "\n".join(lines)
