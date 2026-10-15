"""
core/loaders.py
================
Scans a folder and reads the supported documents in it.
"""

from pathlib import Path
from pypdf import PdfReader
from docx import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_docx(path: Path) -> str:
    document = Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


def scan_folder(folder: str, recursive: bool = True) -> list[Path]:
    """Finds all supported documents in the folder.
    If recursive=True, subfolders are scanned as well (rglob).
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    iterator = folder_path.rglob("*") if recursive else folder_path.glob("*")
    return sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
