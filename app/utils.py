# app/utils.py
import os
import json
import time
from typing import List, Dict, Any
import functools


def get_pdf_files(directory: str) -> List[str]:
    """
    Get all PDF files in a directory.

    Args:
        directory: Directory to scan

    Returns:
        List of PDF file paths
    """
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract metadata from filename.
    Expects format like 'AuthorName_BookTitle_Year.pdf'

    Args:
        filename: Filename to parse

    Returns:
        Dictionary of metadata
    """
    # Default metadata
    metadata = {
        "filename": filename,
        "author": "Unknown",
        "title": "Unknown",
        "year": "Unknown"
    }

    # Try to parse if in expected format
    try:
        base = os.path.basename(filename)
        name = os.path.splitext(base)[0]
        parts = name.split('_')

        if len(parts) >= 3:
            metadata["author"] = parts[0]
            metadata["title"] = parts[1]
            metadata["year"] = parts[2]
        elif len(parts) == 2:
            metadata["author"] = parts[0]
            metadata["title"] = parts[1]
    except:
        pass

    return metadata


def timing_decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        # Just for logging purposes, we'll print the processing time
        print(f"Function {func.__name__} took {processing_time_ms:.2f} ms to execute")
        return result
    return wrapper
