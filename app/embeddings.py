# app/embeddings.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Dict, Any
import os


class DeepSeekEmbedder:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-r1-8b",
                 device: str = None, max_length: int = 512):
        """
        Initialize the DeepSeek embedder.

        Args:
            model_name: HuggingFace model name
            device: Computing device ('cuda', 'cpu', etc.)
            max_length: Maximum sequence length for the tokenizer
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading DeepSeek model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

        print("Model loaded successfully")

    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings.

        Args:
            model_output: Model's output
            attention_mask: Attention mask from tokenizer

        Returns:
            Pooled embeddings
        """
        # Use the last hidden state
        token_embeddings = model_output.last_hidden_state

        # Mask padded tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum token embeddings and divide by the total number of tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding as numpy array
        """
        # Tokenize and prepare for the model
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**inputs)

        # Mean pooling
        embeddings = self._mean_pooling(model_output, inputs["attention_mask"])

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy and return
        return embeddings[0].cpu().numpy()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Embeddings as a numpy array
        """
        embeddings = []

        # Process in batches to avoid memory issues
        batch_size = 8  # Adjust based on your GPU memory

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize and prepare for the model
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**inputs)

            # Mean pooling
            batch_embeddings = self._mean_pooling(model_output, inputs["attention_mask"])

            # Normalize embeddings
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

            # Add to the list
            embeddings.append(batch_embeddings.cpu().numpy())

        # Concatenate all batches
        return np.vstack(embeddings)

    def embed_documents(self, documents: List[Dict[str, Any]], text_field: str = "text") -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of document dictionaries
            text_field: Field name containing the text to embed

        Returns:
            Embeddings as a numpy array
        """
        texts = [doc[text_field] for doc in documents]
        return self.generate_embeddings(texts)
