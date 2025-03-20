# app/pdf_processor.py
import os
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF processor.

        Args:
            chunk_size: The size of text chunks in characters
            chunk_overlap: The overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            The extracted text as a string
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")

        return text

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: The text to clean

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        # Other cleaning operations as needed
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to be chunked

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # If we're not at the end, try to find a good break point
            if end < text_length:
                # Look for a period, question mark, or exclamation point followed by a space
                match = re.search(r'[.!?]\s', text[end - 30:end])
                if match:
                    end = end - 30 + match.end()

            # Create the chunk
            chunk = text[start:end]
            chunks.append(chunk)

            # Calculate the next start position with overlap
            start = end - self.chunk_overlap

            # If we can't advance, force advancement to avoid infinite loop
            if start >= end - 10:
                start = end

        return chunks

    def process_pdf(self, pdf_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process a PDF and return chunked documents with metadata.

        Args:
            pdf_path: Path to the PDF file
            metadata: Additional metadata to include with each chunk

        Returns:
            List of document chunks with metadata
        """
        if metadata is None:
            metadata = {}

        # Add filename to metadata
        filename = os.path.basename(pdf_path)
        metadata["source"] = filename

        # Extract and clean text
        raw_text = self.extract_text_from_pdf(pdf_path)
        cleaned_text = self.clean_text(raw_text)

        # Chunk the text
        chunks = self.chunk_text(cleaned_text)

        # Create documents with metadata
        documents = []
        for i, chunk_text in enumerate(chunks):
            doc = {
                "text": chunk_text,
                "chunk_id": i,
                "total_chunks": len(chunks),
                **metadata
            }
            documents.append(doc)

        return documents
