# -*- coding: utf-8 -*-
"""
    SCoPE
    Compression Functions
    Jesus Alan Hernandez Galvan
"""
from typing import Union
from abc import ABC, abstractmethod
    

class BaseCompressor(ABC):
    """Abstract base class for all compression algorithms."""

    def __init__(self, compressor_name: str, compression_level: int = 9):
        """Initialize the base compressor with specified parameters.

        Args:
            compressor_name: Name of the compression method (e.g., 'gzip', 'bz2')
            compression_level: Compression level (1-9, where 9 is maximum)
        Raises:
            ValueError: If compression level is not between 1-9 or invalid padding method
        """
        if not 1 <= compression_level <= 9:
            raise ValueError("Compression level must be between 1 and 9")

        self._compressor_name: str = compressor_name
        self._compression_level: int = compression_level
    
    @abstractmethod
    def compress(self, sequence: bytes) -> bytes:
        """Abstract method for compressing an input sequence.

        Args:
            sequence: Input data to compress

        Returns:
            Compressed data as bytes
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    def __repr__(self) -> str:
        """Return string representation of the compressor."""
        return f'Compressor(name={self._compressor_name}, level={self._compression_level})'

    def __call__(self, sequence: Union[str, bytes]) -> bytes:
        """Compress the input sequence and return compressed data.

        Args:
            sequence: Input data to compress (str or bytes)

        Returns:
            Compressed data as bytes
            
        Raises:
            ValueError: If a sequence is empty
            TypeError: If the input type is invalid
        """
        if len(sequence) == 0:
            raise ValueError(
                f"Empty sequence provided to {self._compressor_name} compressor.", 
                "Compression requires non-empty input data."
            )
                
        if not isinstance(sequence, (bytes, str)): # noqa
            raise TypeError(
                "Input sequence must be of type 'str' or 'bytes'"
            )
        
        # Convert string to bytes if necessary
        sequence_encoded = sequence.encode('utf-8') if isinstance(sequence, str) else sequence
        
        return self.compress(sequence_encoded)
