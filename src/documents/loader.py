from pathlib import Path
from typing import List

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core.schema import Document as LlamaDocument
from llama_index.readers.docling import DoclingReader


def _build_converter() -> DocumentConverter:
    """Build a Docling DocumentConverter with MPS acceleration for Apple Silicon."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.MPS
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def load_document(path: str) -> List[LlamaDocument]:
    """
    Convert a document file to a list of LlamaIndex Document objects.

    Uses DoclingReader with markdown export — one Document per file,
    preserving the full structure (headings, tables, LaTeX) as markdown.
    The document is then split into TextNodes by DocumentService using
    LlamaIndex's MarkdownNodeParser for semantic chunking.

    MPS acceleration is enabled for Apple Silicon.

    Args:
        path: Absolute path to the document file (PDF, DOCX, etc.)

    Returns:
        List of LlamaIndex Document objects with metadata attached.

    Raises:
        FileNotFoundError: If the path does not exist.
        RuntimeError: If Docling fails to convert the document.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    try:
        reader = DoclingReader(
            export_type=DoclingReader.ExportType.MARKDOWN,
            doc_converter=_build_converter(),
        )
        docs = reader.load_data(str(p))

        # Attach source metadata to every document chunk
        for doc in docs:
            doc.metadata.setdefault("source", str(p))
            doc.metadata.setdefault("filename", p.name)

        return docs

    except Exception as e:
        raise RuntimeError(f"Docling failed to convert '{p.name}': {e}") from e
