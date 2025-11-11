# -*- coding: utf-8 -*-
"""
    SCoPE
    Dissimilarity Compression Metrics
    Jesus Alan Hernandez Galvan
"""
from typing import Optional


def _safe_division(numerator: float, denominator: float, error_msg: str = "Division by zero") -> float:
    """Safely perform division with zero denominator check."""
    if denominator == 0:
        raise ZeroDivisionError(error_msg)
        
    return numerator / denominator


def ncd(c_x1: float, c_x2: float, c_x1x2: float, c_x2x1: Optional[float] = None) -> float:
    """Normalized Compression Distance"""
    _ = c_x2x1
    
    numerator_: float = c_x1x2 - min(c_x1, c_x2)
    denominator_: float = max(c_x1, c_x2)
    
    return _safe_division(
        numerator=numerator_,
        denominator=denominator_,
        error_msg="Cannot compute NCD: max(c_x1, c_x2) is zero"
    )


def cdm(c_x1: float, c_x2: float, c_x1x2: float, c_x2x1: Optional[float] = None) -> float:
    """Compression Dissimilarity Measure"""
    _ = c_x2x1
    
    numerator = c_x1x2
    denominator = c_x1 + c_x2
    
    return _safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute CDM: c_x1 + c_x2 is zero"
    )
    

def cd(c_x1: float, c_x2: float, c_x1x2: float, c_x2x1: float) -> float:
    """Compression Dissimilarity"""
    numerator = min(c_x1, c_x2, c_x1x2, c_x2x1)
    denominator = c_x1 + c_x2
    
    return _safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute CD: c_x1 + c_x2 is zero"
    )


def ucd(c_x1: float, c_x2: float, c_x1x2: float, c_x2x1: float) -> float:
    """Universal Compression Dissimilarity"""
    numerator = max(
       c_x1x2 - c_x1,
        c_x2x1 - c_x2,
    )
    denominator = max(c_x1, c_x2)
    
    return _safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute UCD: max(c_x1, c_x2) is zero"
    )


def ncc(c_x1: float, c_x2: float, c_x1x2: Optional[float] = None, c_x2x1: Optional[float] = None) -> float:
    """Normalized Conditional Compression"""
    _ = c_x1x2
    
    if c_x2x1 is None:
        raise ValueError("NCC requires c_x2x1 parameter")
    
    numerator = c_x2x1 - c_x2
    denominator = c_x1
    
    return _safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute NCC: c_x1 is zero"
    )


def nccd(c_x1: float, c_x2: float, c_x1x2: float, c_x2x1: float) -> float:
    """Normalized Conditional Compression Dissimilarity"""
    numerator = max(
        c_x2x1 - c_x2,  # C(x1|x2) approximation
        c_x1x2 - c_x1   # C(x2|x1) approximation
    )
    denominator = max(c_x1, c_x2)
    
    return _safe_division(
        numerator=numerator,
        denominator=denominator,
        error_msg="Cannot compute NCCD: max(c_x1, c_x2) is zero"
    )


COMPRESSION_METRICS = {
    'ncd': ncd,
    'cdm': cdm,
    'cd': cd,
    'ucd': ucd,
    'ncc': ncc,
    'nccd': nccd
}


def compute_compression_metric(c_x1: float, c_x2: float, c_x1x2: float, c_x2x1: Optional[float], metric: str) -> float:
    """
    Compute the specified compression metric.
    
    Args:
        c_x1: Compression size of x1
        c_x2: Compression size of x2
        c_x1x2: Compression size of x1 concatenated with x2
        c_x2x1: Compression size of x2 concatenated with x1 (optional for some metrics)
        metric: Name of the metric to compute
        
    Returns:
        The computed metric value
        
    Raises:
        ValueError: If metric is not supported
    """
    metric_key = metric.lower()
    
    if metric_key not in COMPRESSION_METRICS:
        allowed = sorted(COMPRESSION_METRICS.keys())
        raise ValueError(f"'{metric}' is not a valid metric. Expected one of: {', '.join(allowed)}")
    
    metric_function = COMPRESSION_METRICS[metric_key]
    
    # Pass all parameters to maintain a consistent interface
    return metric_function(c_x1, c_x2, c_x1x2, c_x2x1)