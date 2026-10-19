import pytest

from edusync_c1.documents import (
    combine_page_text,
    extract_document_pages,
    extract_pdf_pages,
    extract_pptx_pages,
)
from edusync_c1.schemas import Page
from edusync_c1.validation import UploadError


def test_extract_pdf_text(lecture_pdf):
    pages = extract_pdf_pages(lecture_pdf)
    assert len(pages) == 1
    assert "Gradient Descent" in pages[0].text


def test_blank_pdf_has_empty_text(blank_pdf):
    assert extract_pdf_pages(blank_pdf)[0].text == ""


def test_encrypted_pdf_rejected(encrypted_pdf):
    with pytest.raises(UploadError) as err:
        extract_document_pages("a.pdf", encrypted_pdf)
    assert err.value.status_code == 400


def test_extract_pptx_text_and_table(lecture_pptx):
    pages = extract_pptx_pages(lecture_pptx)
    assert [p.page for p in pages] == [1, 2]
    assert "Gradient Descent minimizes" in pages[0].text
    assert "Loss Function | Error" in pages[0].text


def test_invalid_pptx_zip():
    with pytest.raises(UploadError) as err:
        extract_document_pages("a.pptx", b"PK not really")
    assert err.value.status_code == 400


def test_corrupt_pdf_maps_to_422():
    with pytest.raises(UploadError) as err:
        extract_document_pages("a.pdf", b"%PDF-1.4 garbage")
    assert err.value.status_code == 422


def test_dispatch_by_extension(lecture_pdf, lecture_pptx):
    assert extract_document_pages("L.PDF", lecture_pdf)[0].page == 1
    assert len(extract_document_pages("deck.pptx", lecture_pptx)) == 2


def test_combine_page_text_skips_empty():
    pages = [Page(page=1, text="A"), Page(page=2, text=""), Page(page=3, text="C")]
    assert combine_page_text(pages) == "[Page 1]\nA\n\n[Page 3]\nC"
    assert combine_page_text([]) == ""
