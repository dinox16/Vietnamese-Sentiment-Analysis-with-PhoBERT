from vsfc.preprocessing import normalize_text, preprocess_text


def test_normalize_text():
    assert normalize_text("  Thầy   DẠY  Hay ") == "thầy dạy hay"
    assert normalize_text(None) == ""


def test_normalize_unicode_nfc():
    decomposed = "hòa"  # 'hòa' dạng NFD
    assert normalize_text(decomposed) == "hòa"


def test_preprocess_word_segmentation():
    out = preprocess_text("Giảng viên dạy rất dễ hiểu, sinh viên thích.")
    assert "giảng_viên" in out
    assert "sinh_viên" in out


def test_preprocess_empty():
    assert preprocess_text("   ") == ""
