"""Tests for citation remapping utilities."""

import pytest

from api.message_citation_utils import remap_response_citations
from api.shared import SourceReference


class TestCitationRemapping:
    """Test citation remapping from chunk positions to deduplicated source indexes."""

    @pytest.fixture
    def chunks_by_source(self):
        """Sample chunks grouped by source ID."""
        return {
            100: [(1, 0.9, "chunk1"), (3, 0.8, "chunk3")],  # source 100 has positions 1,3
            200: [(2, 0.85, "chunk2")],  # source 200 has position 2
            300: [(4, 0.75, "chunk4"), (5, 0.7, "chunk5")],  # source 300 has positions 4,5
        }

    @pytest.fixture
    def source_references(self):
        """Sample source references (sorted by relevance)."""
        return [
            SourceReference(
                source_id=100,
                document_name="doc1.pdf",
                chunk_text="chunk1",
                relevance_score=0.9,
            ),
            SourceReference(
                source_id=200,
                document_name="doc2.pdf",
                chunk_text="chunk2",
                relevance_score=0.85,
            ),
            SourceReference(
                source_id=300,
                document_name="doc3.pdf",
                chunk_text="chunk4",
                relevance_score=0.75,
            ),
        ]

    def test_single_citation(self, chunks_by_source, source_references):
        """Test remapping a single citation."""
        response = "According to the document [§1], we can see..."
        result = remap_response_citations(response, chunks_by_source, source_references)
        assert result == "According to the document [§1], we can see..."

    def test_group_citation(self, chunks_by_source, source_references):
        """Test remapping a group of citations."""
        # Positions 1,2,3 map to sources 100,200,100 which become indices 1,2,1
        # After deduplication and sorting: [§1, §2]
        response = "Multiple sources [§1, §2, §3] confirm this."
        result = remap_response_citations(response, chunks_by_source, source_references)
        assert result == "Multiple sources [§1, §2] confirm this."

    def test_invalid_citation_removal(self, chunks_by_source, source_references):
        """Test that invalid citations are removed."""
        response = "Based on [§99] which doesn't exist."
        result = remap_response_citations(response, chunks_by_source, source_references)
        # Invalid citation should be removed entirely
        assert result == "Based on  which doesn't exist."

    def test_standalone_citation(self, chunks_by_source, source_references):
        """Test standalone §N citations not in brackets."""
        response = "As mentioned in §2 earlier..."
        result = remap_response_citations(response, chunks_by_source, source_references)
        assert result == "As mentioned in §2 earlier..."

    def test_mixed_formats(self, chunks_by_source, source_references):
        """Test handling both bracketed and standalone citations."""
        response = "First §1 then [§2, §3] and finally §4."
        result = remap_response_citations(response, chunks_by_source, source_references)
        # §1 -> §1, [§2,§3] -> [§1,§2], §4 -> §3
        assert result == "First §1 then [§1, §2] and finally §3."

    def test_empty_input(self, chunks_by_source, source_references):
        """Test handling empty input."""
        response = ""
        result = remap_response_citations(response, chunks_by_source, source_references)
        assert result == ""

    def test_no_citations(self, chunks_by_source, source_references):
        """Test text without any citations."""
        response = "This text has no citations at all."
        result = remap_response_citations(response, chunks_by_source, source_references)
        assert result == "This text has no citations at all."

    def test_all_invalid_group(self, chunks_by_source, source_references):
        """Test group with all invalid citations is removed."""
        response = "Based on [§98, §99] invalid sources."
        result = remap_response_citations(response, chunks_by_source, source_references)
        assert result == "Based on  invalid sources."

    def test_plain_number_format(self, chunks_by_source, source_references):
        """Test plain number citations without § symbol."""
        response = "Sources [1, 2, 3] show..."
        result = remap_response_citations(response, chunks_by_source, source_references)
        # Should remap same as with § symbols
        assert result == "Sources [§1, §2] show..."

    def test_position_beyond_chunks(self, chunks_by_source, source_references):
        """Test citation position that doesn't exist in chunks."""
        response = "Invalid position [§10] here."
        result = remap_response_citations(response, chunks_by_source, source_references)
        assert result == "Invalid position  here."

    def test_deduplication_within_group(self, chunks_by_source, source_references):
        """Test that duplicate sources in a group are deduplicated."""
        # Positions 1 and 3 both map to source 100 (index 1)
        response = "Citations [§1, §3] from same source."
        result = remap_response_citations(response, chunks_by_source, source_references)
        # Should deduplicate to single §1
        assert result == "Citations [§1] from same source."
