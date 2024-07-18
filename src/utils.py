import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union, List, Set, Sequence

import numpy as np

from src import config

from cpr_data_access.parser_models import ParserOutput, TextBlock, BlockType
from src.ml import SentenceEncoder
from src.s3 import get_s3_keys_with_prefix, s3_object_read_text

logger = logging.getLogger(__name__)


def encoder_name_as_slug(encoder_name: str) -> str:
    """Convert an encoder name to a slug by replacing all punctuation with - and lowercasing."""

    return "".join(c if c.isalnum() else "-" for c in encoder_name).lower()


def replace_text_blocks(block: ParserOutput, new_text_blocks: Sequence[TextBlock]):
    """Updates the text blocks in the ParserOutput object."""
    if block.pdf_data:
        block.pdf_data.text_blocks = new_text_blocks  # type: ignore
    elif block.html_data:
        block.html_data.text_blocks = new_text_blocks  # type: ignore

    return block


def filter_blocks(
    parser_output: ParserOutput, remove_block_types: Sequence[str]
) -> Sequence[TextBlock]:
    """
    Given an ParserOutput filter the contained TextBlocks.

    Return this as a list of TextBlocks.
    """
    filtered_blocks = []
    # TODO: this denotes a bug in the data access library that should be fixed
    for block in parser_output.get_text_blocks(including_invalid_html=True):
        if block.type.title() not in remove_block_types:
            filtered_blocks.append(block)
        else:
            logger.info(
                f"Filtered {block.type} block from {parser_output.document_id}.",
                extra={
                    "props": {
                        "document_id": parser_output.document_id,
                        "block_type": block.type,
                        "remove_block_types": remove_block_types,
                    }
                },
            )
    return filtered_blocks


def filter_on_block_type(
    inputs: Sequence[ParserOutput], remove_block_types: List[str]
) -> Sequence[ParserOutput]:
    """
    Filter a sequence of ParserOutputs.

    Remove the text blocks that are of the types declared in the remove block types
    array.
    """
    for _filter in remove_block_types:
        try:
            BlockType(_filter)
        except NameError:
            logger.warning(
                f"Blocks to filter should be of a known block type, removing {_filter} "
                f"from the list. "
            )
            remove_block_types.remove(_filter)

    return [
        replace_text_blocks(
            block=_input,
            new_text_blocks=filter_blocks(
                parser_output=_input, remove_block_types=remove_block_types
            ),
        )
        for _input in inputs
    ]


def get_ids_with_suffix(files: Sequence[str], suffix: str) -> Set[str]:
    """Get a set of the ids of the files with the given suffix."""
    files = [file for file in files if file.endswith(suffix)]
    return set([os.path.splitext(os.path.basename(file))[0] for file in files])


def encode_parser_output(
    encoder: SentenceEncoder,
    input_obj: ParserOutput,
    batch_size: int,
    device: Optional[str] = None,
    use_text_block_window: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Encode a parser output object.

    Produce a numpy array of description embedding and a numpy array of text
    embeddings for a parser output.

    :param encoder: sentence encoder
    :param input_obj: parser output object
    :param batch_size: batch size for encoding text blocks
    :param device: device to use for encoding
    :param use_text_block_window: whether to use a block around each text block for encoding
    """

    description_embedding = encoder.encode(
        input_obj.document_description, device=device
    )

    text_blocks = input_obj.get_text_blocks()
    text_to_encode = []

    if use_text_block_window:
        text_to_encode = []

        for idx in range(len(text_blocks)):
            if idx == 0:
                idxs_in_window = [idx, idx + 1]
            elif idx == len(text_blocks) - 1:
                idxs_in_window = [idx - 1, idx]
            else:
                idxs_in_window = [
                    idx - 1,
                    idx,
                    idx + 1,
                ]

            text_to_encode.append(
                "\n".join(text_blocks[i].to_string() for i in idxs_in_window)
            )

    else:
        text_to_encode = [block.to_string() for block in text_blocks]

    if text_blocks:
        text_embeddings = encoder.encode_batch(
            text_to_encode,
            batch_size=batch_size,
            device=device,
            # TODO: we always use a sliding window here. Do we want this?
            experimental_use_sliding_window=True,
        )
    else:
        text_embeddings = None

    return description_embedding, text_embeddings


def get_files_to_process(
    s3: bool, input_dir: str, output_dir: str, redo: bool, limit: Union[None, int]
) -> Sequence[str]:
    """
    Get the list of files to process.

    Either from the config or from the input directory.
    """
    if s3:
        document_paths_previously_parsed = get_s3_keys_with_prefix(output_dir)
    else:
        document_paths_previously_parsed = os.listdir(output_dir)

    document_ids_previously_parsed = get_ids_with_suffix(
        document_paths_previously_parsed, ".npy"
    )

    if config.FILES_TO_PROCESS is not None:
        files_to_process_subset = config.FILES_TO_PROCESS.split("$")[1:]
        files_to_process = [os.path.join(input_dir, f) for f in files_to_process_subset]
    else:
        if s3:
            files_to_process = get_s3_keys_with_prefix(input_dir)
        else:
            files_to_process = os.listdir(input_dir)

    files_to_process_ids = get_ids_with_suffix(files_to_process, ".json")
    files_already_processed = document_ids_previously_parsed.intersection(
        files_to_process_ids
    )
    if not redo and files_already_processed:
        logger.warning(
            f"{len(files_already_processed)} "
            f"documents found that have already been encoded. Skipping. "
        )

    files_to_process_ids_sequence = [
        id_ for id_ in files_to_process_ids if id_ not in document_ids_previously_parsed
    ]
    if not files_to_process_ids_sequence:
        logger.warning("No more documents to encode. Exiting.")

    if limit:
        logger.info(
            f"Limiting to {files_to_process_ids} documents as the --limit flag has "
            f"been passed. "
        )
        return files_to_process_ids_sequence[:limit]

    return files_to_process_ids_sequence


def get_Text2EmbeddingsInput_array(
    input_dir: str, s3: bool, files_to_process_ids: Sequence[str]
) -> List[ParserOutput]:
    """Construct ParserOutput objects from parser output jsons.

    These objects will be used to generate embeddings and are either read in from S3
    or from the local file system.
    """
    return [
        ParserOutput.model_validate_json(
            s3_object_read_text(os.path.join(input_dir, id_ + ".json"))
            if s3
            else Path(os.path.join(input_dir, id_ + ".json")).read_text()
        )
        for id_ in files_to_process_ids
    ]


def sliding_window(text: str, window_size: int, stride: int):
    """Split the text into overlapping windows."""
    windows = [
        text[i : i + window_size]
        for i in range(0, len(text), stride)
        if i + window_size <= len(text)
    ]
    return windows
