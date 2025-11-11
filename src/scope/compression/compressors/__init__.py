from enum import Enum
from typing import Union

from .base import BaseCompressor
from .custom import RLECompressor, LZ77Compressor
from .integrated import Bz2Compressor, GZipCompressor, ZlibCompressor, SmilezCompressor, SmazCompressor


class CompressorType(Enum):
    # """Enumeration of supported compression algorithms."""
    GZIP = "gzip"
    BZ2 = "bz2"
    ZLIB = "zlib"
    RLE = "rle"
    LZ77 = "lz77"
    SMAZ = "smaz"
    SMILEZ = "smilez"
    
    
COMPRESSOR_STRATEGIES = {
    CompressorType.SMILEZ: SmilezCompressor,
    CompressorType.SMAZ: SmazCompressor,
    CompressorType.GZIP: GZipCompressor,
    CompressorType.BZ2: Bz2Compressor,
    CompressorType.ZLIB: ZlibCompressor,
    CompressorType.RLE: RLECompressor,
    CompressorType.LZ77: LZ77Compressor
}


def get_compressor(name: Union[str, CompressorType], compression_level: int = 9,) -> BaseCompressor:
    """Factory function to get a specific compressor instance.
    
    Args:
        name: Compressor name ('rle', 'huffman', 'lz77', 'zlib', 'bz2', 'zstd')
        compression_level: Compression level (1-9)    
    Returns:
        Compressor instance
        
    Raises:
        ValueError: If compressor name is invalid
        TypeError: If name type is invalid
    """
    if isinstance(name, str):
        try:
            compressor_enum = CompressorType(name.lower())
        except ValueError:
            allowed = sorted(c.value for c in CompressorType)
            raise ValueError(
                f"'{name}' is not a valid compressor name.",
                f"Expected one of: {', '.join(allowed)}"
            )
    elif isinstance(name, CompressorType):
        compressor_enum = name
    else:
        raise TypeError("Expected 'name' to be str or CompressorType")
    
    compressor_class = COMPRESSOR_STRATEGIES[compressor_enum]
    return compressor_class(
        compression_level=compression_level
    )


def compute_compression(
    sequence: Union[str, bytes], 
    compressor: str, 
    compression_level: int,
) -> bytes:
    """Compute compression for a given sequence using specified compressor.
    
    Args:
        sequence: Input string to compress
        compressor: Name of the compressor to use
        compression_level: Compression level (1-9)
        
    Returns:
        Compressed data as bytes
    """
    compressor_instance = get_compressor(
        name=compressor.lower(),
        compression_level=compression_level,
    )
    
    return compressor_instance(sequence)
    
__all__ = [
    'get_compressor',
    'compute_compression',
    'CompressorType'
]