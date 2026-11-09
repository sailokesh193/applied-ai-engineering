from claimlens.chunking import approx_tokens, chunk_document
from claimlens.providers.local import split_markdown
from claimlens.types import ParsedDocument, Section


def test_split_markdown_tracks_headings_and_page_breaks():
    md = "# Title\n\nIntro text.\n\n## A\nalpha\n<!-- PageBreak -->\n<!-- PageNumber=\"2\" -->\n## B\nbeta"
    title, sections = split_markdown(md)
    assert title == "Title"
    assert [s.title for s in sections] == ["Title", "A", "B"]
    assert sections[-1].page == 2
    assert "PageNumber" not in sections[-1].text


def test_chunks_respect_size_and_section_boundaries():
    long_text = " ".join(f"Sentence number {i} talks about knee arthroscopy." for i in range(200))
    doc = ParsedDocument("d1", "d1.md", "D1", [Section("Long", long_text, 1), Section("Short", "One line.", 2)])
    chunks = chunk_document(doc, max_tokens=120, overlap_tokens=20)
    assert len({c.section for c in chunks}) == 2
    assert all(approx_tokens(c.text) <= 140 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    long_chunks = [c for c in chunks if c.section == "Long"]
    # overlap: the last sentence of one chunk starts the next
    first_tail = long_chunks[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in long_chunks[1].text


def test_chunk_ids_are_stable():
    doc = ParsedDocument("d1", "d1.md", "D1", [Section("S", "Alpha beta. Gamma delta.", 1)])
    assert [c.chunk_id for c in chunk_document(doc)] == [c.chunk_id for c in chunk_document(doc)]
