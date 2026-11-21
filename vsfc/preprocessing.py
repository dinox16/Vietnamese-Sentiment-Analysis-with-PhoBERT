"""Tiền xử lý văn bản — PHẢI giống hệt bước dùng lúc huấn luyện (notebook)."""

import re
import unicodedata

from underthesea import word_tokenize


def normalize_text(text) -> str:
    """Chuẩn hóa Unicode NFC, chữ thường, gộp khoảng trắng."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text) -> str:
    """Chuẩn hóa + tách từ (VD: 'giảng viên' -> 'giảng_viên') cho PhoBERT."""
    text = normalize_text(text)
    return word_tokenize(text, format="text") if text else ""
