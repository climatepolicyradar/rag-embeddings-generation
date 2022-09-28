from pathlib import Path

import pytest

from src.base import IndexerInput
from cli.index_data import get_document_generator


@pytest.fixture()
def test_input_dir() -> Path:
    return (Path(__file__).parent / "test_data" / "load_data_input").resolve()


def test_get_document_generator(test_input_dir: Path):
    """Test that the document generator returns documents in the correct format."""

    tasks = [
        IndexerInput.parse_raw(path.read_text())
        for path in list(test_input_dir.glob("*.json"))
    ]

    doc_generator = get_document_generator(tasks, test_input_dir)

    for_search_name_count = 0
    for_search_description_count = 0

    for doc in doc_generator:
        assert all(
            [
                field in doc
                for field in {
                    "document_id",
                    "document_name",
                    "document_description",
                    "document_url",
                    "translated",
                    "document_slug",
                    "document_content_type",
                    "document_source_url",
                }
            ]
        )

        if "for_search_document_name" in doc:
            for_search_name_count += 1
            assert all(
                [
                    field not in doc
                    for field in {
                        "for_search_document_description",
                        "document_description_embedding",
                        "text_block_id",
                        "text",
                        "text_embedding",
                    }
                ]
            )

        if "for_search_document_description" in doc:
            for_search_description_count += 1
            assert all(
                [
                    field not in doc
                    for field in {
                        "for_search_document_name",
                        "text_block_id",
                        "text",
                        "text_embedding",
                    }
                ]
            )

            assert "document_description_embedding" in doc

        if "text_block_id" in doc:
            assert all(
                [
                    field not in doc
                    for field in {
                        "for_search_document_name",
                        "for_search_document_description",
                        "document_description_embedding",
                    }
                ]
            )

            assert "text" in doc
            assert "text_embedding" in doc
            assert "text_block_coords" in doc
            assert "text_block_page" in doc

    assert for_search_name_count == len(tasks)
    assert for_search_description_count == len(tasks)