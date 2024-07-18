"""Text encoders."""

from abc import ABC, abstractmethod
from typing import List, Optional

from sentence_transformers import SentenceTransformer
import numpy as np

from src import config


def sliding_window(text: str, window_size: int, stride: int):
    """Split the text into overlapping windows."""
    windows = [
        text[i : i + window_size]
        for i in range(0, len(text), stride)
        if i + window_size <= len(text)
    ]
    return windows


class SentenceEncoder(ABC):
    """Base class for a sentence encoder"""

    @abstractmethod
    def encode(self, text: str, device: Optional[str] = None) -> np.ndarray:
        """Encode a string, return a numpy array."""
        raise NotImplementedError

    @abstractmethod
    def encode_batch(
        self,
        text_batch: List[str],
        batch_size: int,
        device: Optional[str] = None,
        experimental_use_sliding_window: bool = False,
    ) -> np.ndarray:
        """Encode a batch of strings, return a numpy array."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings produced by the encoder."""
        raise NotImplementedError


class SBERTEncoder(SentenceEncoder):
    """Encoder which uses the sentence-transformers library.

    A list of pretrained models is available at
    https://www.sbert.net/docs/pretrained_models.html.
    """

    def __init__(self, model_name: str):
        super().__init__()

        self.encoder = SentenceTransformer(
            model_name, cache_folder=config.INDEX_ENCODER_CACHE_FOLDER
        )

    def get_n_tokens(self, text: str) -> int:
        """Return the number of tokens in the text."""

        tokenized = self.encoder[0].tokenizer(
            text, return_attention_mask=False, return_token_type_ids=False
        )

        return len(tokenized["input_ids"])

    def encode(self, text: str, device: Optional[str] = None) -> np.ndarray:
        """Encode a string, return a numpy array.

        Args:
            text (str): string to encode.
            device (str): torch.device to use for encoding.

        Returns:
            np.ndarray
        """
        return self.encoder.encode(text, device=device, show_progress_bar=False)

    def encode_batch(
        self,
        text_batch: List[str],
        batch_size: int = 32,
        device: Optional[str] = None,
        experimental_use_sliding_window: bool = False,
    ) -> np.ndarray:
        """Encode a batch of strings, return a numpy array.

        Args:
            text_batch (List[str]): list of strings to encode.
            device (str): torch.device to use for encoding.
            batch_size (int, optional): batch size to encode strings in. Defaults to 32.

        Returns:
            np.ndarray
        """
        if experimental_use_sliding_window:
            return self._encode_batch_using_sliding_window(
                text_batch, batch_size=batch_size, device=device
            )

        return self.encoder.encode(
            text_batch, batch_size=batch_size, show_progress_bar=False, device=device
        )

    def _encode_batch_using_sliding_window(
        self, text_batch: list[str], batch_size: int = 32, device: Optional[str] = None
    ):
        """
        Encode a batch of strings accommodating long texts using a sliding window.

        For args, see encode_batch.
        """

        max_seq_length = self.encoder.max_seq_length
        assert isinstance(max_seq_length, int)

        # Split the texts based on length and apply sliding window only to longer texts
        processed_texts = []
        window_lengths = []

        for text in text_batch:
            if self.get_n_tokens(text) > max_seq_length:
                windows = sliding_window(
                    text, window_size=max_seq_length, stride=max_seq_length // 2
                )  # Use max_seq_length as window size and half of it as stride
                processed_texts.extend(windows)
                window_lengths.append(len(windows))
            else:
                processed_texts.append(text)
                window_lengths.append(1)

        embeddings = self.encode_batch(
            processed_texts, batch_size=batch_size, device=device
        )

        reduced_embeddings = []

        for length in window_lengths:
            if length > 1:
                reduced_embeddings.append(np.mean(embeddings[:length], axis=0))
                embeddings = embeddings[length:]
            else:
                reduced_embeddings.append(embeddings[0])
                embeddings = embeddings[1:]

        return np.vstack(reduced_embeddings)

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding."""
        return self.encoder.get_sentence_embedding_dimension()
