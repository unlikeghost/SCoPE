# -*- coding: utf-8 -*-
"""
    SCoPE
    Compression Functions - Improved
    Jesus Alan Hernandez Galvan
"""
from .base import BaseCompressor


class RLECompressor(BaseCompressor):
    """Run Length Encoding - robust version using bytearray."""

    def __init__(self, compression_level: int = 1):
        super().__init__(
            compressor_name="rle",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        if not sequence:
            return b""

        compressed = bytearray()
        current_byte = sequence[0]
        count = 1

        for byte in sequence[1:]:
            if byte == current_byte and count < 255:
                count += 1
            else:
                compressed.extend([count, current_byte])
                current_byte = byte
                count = 1

        compressed.extend([count, current_byte])
        return bytes(compressed)


class LZ77Compressor(BaseCompressor):
    """LZ77 algorithm with safe offsets and lengths."""

    def __init__(self, compression_level: int = 1):
        super().__init__(
            compressor_name="lz77",
            compression_level=compression_level,
        )
        self.window_size = 10 + (compression_level * 2)
        self.lookahead_size = 5 + compression_level

    def compress(self, sequence: bytes) -> bytes:
        if not sequence:
            return b""

        compressed = bytearray()
        i = 0
        while i < len(sequence):
            start = max(0, i - self.window_size)
            search_window = sequence[start:i]
            lookahead_end = min(len(sequence), i + self.lookahead_size)
            lookahead = sequence[i:lookahead_end]
            offset, length = self._find_longest_match(search_window, lookahead)
            if length >= 3:
                # Store offset and length as 2 bytes each
                compressed.extend([1])
                compressed.extend(offset.to_bytes(2, "big"))
                compressed.extend(length.to_bytes(2, "big"))
                i += length
            else:
                compressed.extend([0, sequence[i]])
                i += 1
        return bytes(compressed)

    def _find_longest_match(self, search_window: bytes, lookahead: bytes) -> tuple[int, int]:
        best_offset, best_length = 0, 0
        sw_len = len(search_window)
        for j in range(sw_len):
            match_len = 0
            while (match_len < len(lookahead) and
                   j + match_len < sw_len and
                   search_window[j + match_len] == lookahead[match_len]):
                match_len += 1
            if match_len > best_length:
                best_length = match_len
                best_offset = sw_len - j
        return best_offset, best_length
