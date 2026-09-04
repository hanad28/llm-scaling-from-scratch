"""Input checks shared by the CLIs and the dataclasses they build.

Each check either returns the validated value or raises with a message that names the
offending option, so bad input fails at the entry point rather than inside a loop or
a numerical routine. The `*_int` helpers double as argparse `type=` converters, where
raising `argparse.ArgumentTypeError` gives the usual "argument --x: ..." message.
"""

from __future__ import annotations

import argparse
from collections.abc import Hashable, Sequence
from typing import TypeVar

Item = TypeVar("Item", bound=Hashable)


def positive_int(text: str | int) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def non_negative_int(text: str | int) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {text}")
    return value


def require_positive(name: str, value: float) -> None:
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_non_negative(name: str, value: float) -> None:
    if not value >= 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def require_unit_interval(name: str, value: float) -> None:
    """Strictly between 0 and 1, as for confidence levels and optimiser betas."""
    if not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")


def require_fraction(name: str, value: float) -> None:
    """In [0, 1] inclusive, as for a fraction of a peak value."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie between 0 and 1 inclusive, got {value}")


def require_unique(name: str, values: Sequence[Item]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    repeated = sorted({value for value in values if values.count(value) > 1}, key=str)
    if repeated:
        raise ValueError(f"{name} contains repeated entries {repeated}: {list(values)}")
