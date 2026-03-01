"""
Document Loader Factory

Factory for creating document loaders based on file type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from RAGService.Data.DocumentProcessors.base import (
    BaseDocumentLoader,
    ProcessedDocument,
    SupportedFileType,
)


# Loader registry - maps extensions to loader classes
_LOADER_REGISTRY: Dict[str, Type[BaseDocumentLoader]] = {}


def _load_default_loaders() -> None:
    """Load default loader implementations."""
    global _LOADER_REGISTRY
    
    if _LOADER_REGISTRY:
        return
    
    # Text loaders (always available)
    try:
        from RAGService.Data.DocumentProcessors.loaders.text_loader import (
            TextLoader,
            MarkdownLoader,
        )
        _LOADER_REGISTRY["txt"] = TextLoader
        _LOADER_REGISTRY["md"] = MarkdownLoader
        _LOADER_REGISTRY["markdown"] = MarkdownLoader
    except ImportError:
        pass
    
    # JSON loader (always available)
    try:
        from RAGService.Data.DocumentProcessors.loaders.json_loader import JSONLoader
        _LOADER_REGISTRY["json"] = JSONLoader
    except ImportError:
        pass
    
    # CSV loader (always available)
    try:
        from RAGService.Data.DocumentProcessors.loaders.csv_loader import CSVLoader
        _LOADER_REGISTRY["csv"] = CSVLoader
    except ImportError:
        pass
    
    # PDF loader (requires pypdf)
    try:
        from RAGService.Data.DocumentProcessors.loaders.pdf_loader import PDFLoader
        _LOADER_REGISTRY["pdf"] = PDFLoader
    except ImportError:
        pass
    
    # DOCX loader (requires python-docx)
    try:
        from RAGService.Data.DocumentProcessors.loaders.docx_loader import DOCXLoader
        _LOADER_REGISTRY["docx"] = DOCXLoader
    except ImportError:
        pass


class DocumentLoaderFactory:
    """
    Factory for creating document loaders.
    
    Automatically selects the appropriate loader based on file extension.
    
    Example:
        # Get loader by file path
        loader = DocumentLoaderFactory.get_loader("document.pdf")
        doc = loader.load("document.pdf")
        
        # Load directly
        doc = DocumentLoaderFactory.load("document.pdf")
        
        # Register custom loader
        DocumentLoaderFactory.register_loader("custom", MyCustomLoader)
    """
    
    @staticmethod
    def get_loader(
        file_path: Union[str, Path],
        **loader_kwargs
    ) -> BaseDocumentLoader:
        """
        Get an appropriate loader for a file.
        
        Args:
            file_path: Path to the file
            **loader_kwargs: Additional arguments to pass to loader constructor
            
        Returns:
            Configured loader instance
            
        Raises:
            ValueError: If no loader supports the file type
        """
        _load_default_loaders()
        
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")
        
        if extension not in _LOADER_REGISTRY:
            raise ValueError(
                f"No loader available for extension '.{extension}'. "
                f"Supported extensions: {list(_LOADER_REGISTRY.keys())}"
            )
        
        loader_class = _LOADER_REGISTRY[extension]
        return loader_class(**loader_kwargs)
    
    @staticmethod
    def get_loader_for_type(
        file_type: SupportedFileType,
        **loader_kwargs
    ) -> BaseDocumentLoader:
        """
        Get a loader for a specific file type.
        
        Args:
            file_type: The file type enum
            **loader_kwargs: Additional arguments to pass to loader constructor
            
        Returns:
            Configured loader instance
        """
        _load_default_loaders()
        
        extension = file_type.value
        if extension not in _LOADER_REGISTRY:
            raise ValueError(
                f"No loader available for type '{file_type.value}'."
            )
        
        loader_class = _LOADER_REGISTRY[extension]
        return loader_class(**loader_kwargs)
    
    @staticmethod
    def load(
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        **loader_kwargs
    ) -> ProcessedDocument:
        """
        Load a document directly.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata to include
            **loader_kwargs: Additional arguments for the loader
            
        Returns:
            ProcessedDocument
        """
        loader = DocumentLoaderFactory.get_loader(file_path, **loader_kwargs)
        return loader.load(file_path, metadata)
    
    @staticmethod
    async def async_load(
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        **loader_kwargs
    ) -> ProcessedDocument:
        """Async version of load."""
        loader = DocumentLoaderFactory.get_loader(file_path, **loader_kwargs)
        return await loader.async_load(file_path, metadata)
    
    @staticmethod
    def register_loader(
        extension: str,
        loader_class: Type[BaseDocumentLoader],
        override: bool = False
    ) -> None:
        """
        Register a custom loader for an extension.
        
        Args:
            extension: File extension (without dot)
            loader_class: The loader class
            override: Whether to override existing registration
        """
        _load_default_loaders()
        
        ext = extension.lower().lstrip(".")
        if ext in _LOADER_REGISTRY and not override:
            raise ValueError(
                f"Loader for '.{ext}' is already registered. "
                "Use override=True to replace."
            )
        
        _LOADER_REGISTRY[ext] = loader_class
    
    @staticmethod
    def list_supported_extensions() -> List[str]:
        """List all supported file extensions."""
        _load_default_loaders()
        return list(_LOADER_REGISTRY.keys())
    
    @staticmethod
    def is_supported(file_path: Union[str, Path]) -> bool:
        """Check if a file type is supported."""
        _load_default_loaders()
        
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")
        return extension in _LOADER_REGISTRY
