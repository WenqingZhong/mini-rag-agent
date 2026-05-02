"""
Unit tests for rag/ingestion.py

Tests chunking logic (pure function) and ingest_text / ingest_file
with the lower-level ingest() call mocked out.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, call

from rag.ingestion import chunk_text, ingest_text, ingest_file


class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   ") == []

    def test_short_text_returns_single_chunk(self):
        result = chunk_text("hello world", chunk_size=10)
        assert result == ["hello world"]

    def test_exact_chunk_size_returns_single_chunk(self):
        words = " ".join(["word"] * 5)
        result = chunk_text(words, chunk_size=5)
        assert len(result) == 1
        assert result[0] == words

    def test_long_text_splits_into_multiple_chunks(self):
        words = " ".join([f"w{i}" for i in range(20)])
        result = chunk_text(words, chunk_size=5, overlap=0)
        assert len(result) == 4  # 20 words / 5 per chunk

    def test_overlap_creates_shared_words(self):
        words = "a b c d e f g h i j"
        chunks = chunk_text(words, chunk_size=5, overlap=2)
        # chunk 0: a b c d e
        # chunk 1: d e f g h  (d and e overlap)
        assert chunks[0].endswith("d e")
        assert chunks[1].startswith("d e")

    def test_each_chunk_has_at_most_chunk_size_words(self):
        text = " ".join([f"word{i}" for i in range(100)])
        for chunk in chunk_text(text, chunk_size=10, overlap=2):
            assert len(chunk.split()) <= 10

    def test_all_words_covered(self):
        words = [f"w{i}" for i in range(30)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        all_chunk_words = " ".join(chunks).split()
        # Every original word must appear in at least one chunk
        for word in words:
            assert word in all_chunk_words

    def test_single_word(self):
        result = chunk_text("hello", chunk_size=5)
        assert result == ["hello"]


class TestIngestText:
    def test_calls_ingest_with_chunks_and_metadata(self):
        with patch("rag.ingestion.ingest") as mock_ingest:
            count = ingest_text("my_collection", "hello world foo bar", source="test")
            mock_ingest.assert_called_once()
            args = mock_ingest.call_args
            assert args[0][0] == "my_collection"
            chunks = args[0][1]
            assert isinstance(chunks, list)
            assert len(chunks) > 0

    def test_returns_chunk_count(self):
        with patch("rag.ingestion.ingest"):
            count = ingest_text("col", "one two three", source="s")
            assert isinstance(count, int)
            assert count >= 1

    def test_metadata_includes_source_and_index(self):
        with patch("rag.ingestion.ingest") as mock_ingest:
            ingest_text("col", "a b c d e", source="my-source")
            metadatas = mock_ingest.call_args[0][2]
            assert metadatas[0]["source"] == "my-source"
            assert metadatas[0]["chunk_index"] == 0

    def test_empty_text_calls_ingest_with_empty_list(self):
        with patch("rag.ingestion.ingest") as mock_ingest:
            count = ingest_text("col", "")
            # chunk_text("") returns [] so ingest is called with []
            assert count == 0


class TestIngestFile:
    def test_reads_file_and_ingests(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world from file")
            path = f.name
        try:
            with patch("rag.ingestion.ingest") as mock_ingest:
                count = ingest_file("col", path)
                assert count >= 1
                mock_ingest.assert_called_once()
        finally:
            os.unlink(path)

    def test_file_path_used_as_source(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text here for testing")
            path = f.name
        try:
            with patch("rag.ingestion.ingest") as mock_ingest:
                ingest_file("col", path)
                metadatas = mock_ingest.call_args[0][2]
                assert metadatas[0]["source"] == path
        finally:
            os.unlink(path)

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ingest_file("col", "/does/not/exist.txt")
