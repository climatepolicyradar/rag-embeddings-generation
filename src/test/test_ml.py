import numpy as np

from src import config
from src.ml import SBERTEncoder


def test_encoder():
    """Assert that we can instantiate an encoder object and encode textual data using the class methods."""

    encoder = SBERTEncoder(config.SBERT_MODELS[0])

    assert encoder is not None

    assert isinstance(encoder.encode("Hello world!"), np.ndarray)

    assert isinstance(encoder.encode_batch(["Hello world!"] * 100), np.ndarray)

    assert encoder.dimension == 384


def test_encoder_sliding_window():
    """Assert that we can encode long texts using a sliding window."""

    encoder = SBERTEncoder(config.SBERT_MODELS[0])

    long_text = "Hello world! " * 50
    short_text = "Hello world!"

    batch_to_encode = [short_text, long_text, short_text, short_text]
    embeddings = encoder._encode_batch_using_sliding_window(
        batch_to_encode, batch_size=32
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(batch_to_encode)
    assert embeddings.shape[1] == encoder.dimension

    assert np.array_equal(embeddings[0, :], embeddings[2, :])
    assert np.array_equal(embeddings[0, :], embeddings[3, :])
    assert not np.array_equal(embeddings[0, :], embeddings[1, :])
