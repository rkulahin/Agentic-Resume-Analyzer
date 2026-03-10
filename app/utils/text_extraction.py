"""Utilities for extracting text from PDF, TXT files and URLs."""

from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import io


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from PDF file bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode TXT file bytes to string."""
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return file_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace").strip()


def extract_text_from_url(url: str, timeout: int = 15) -> str:
    """Fetch a URL and extract readable text content."""
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "ResumeAnalyzer/1.0"})
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")

    if "application/pdf" in content_type:
        return extract_text_from_pdf(resp.content)

    if "text/plain" in content_type:
        return resp.text.strip()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
