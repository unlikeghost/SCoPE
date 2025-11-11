# -*- coding: utf-8 -*-
"""
    SCoPE
    Compression Functions
    Jesus Alan Hernandez Galvan
"""
import gzip
import bz2
import zlib
import smaz
import smilez

from .base import BaseCompressor


class SmazCompressor(BaseCompressor):
    def __init__(self, compression_level: int = 1):
        super().__init__(
            compressor_name="smaz",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        sequence_: str = sequence.decode("utf-8")
        return smaz.compress(sequence_)

class Bz2Compressor(BaseCompressor):
    """BZ2 compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="bz2",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using the BZ2 algorithm, removing header for size optimization."""
        return bz2.compress(sequence, compresslevel=self._compression_level)
        
        
class ZlibCompressor(BaseCompressor):
    """ZLIB compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using the ZLIB algorithm with raw deflating (no headers)."""
        return zlib.compress(sequence, level=self._compression_level)


class GZipCompressor(BaseCompressor):
    """ZLIB compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        """Compress using the ZLIB algorithm with raw deflating (no headers)."""
        return gzip.compress(sequence, compresslevel=self._compression_level)


class SmilezCompressor(BaseCompressor):
    """Smilez compression algorithm implementation."""
    
    def __init__(self, compression_level: int = 1):
        super().__init__(
            compressor_name="smilez",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        return smilez.compress(sequence)
