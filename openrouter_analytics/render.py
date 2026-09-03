"""Plain-text table rendering shared by the standalone scripts and the TUI.

Tables use a light horizontal rule (``─``), two-space column gaps, and no vertical bars.
"""

from typing import Iterable, List, NamedTuple, Optional, Sequence


class Column(NamedTuple):
    name: str
    width: int
    align: str = "<"  # "<" left, ">" right


def fmt_cell(value: str, col: Column) -> str:
    return f"{value:>{col.width}}" if col.align == ">" else f"{value:<{col.width}}"


def format_row(values: Sequence[str], cols: Sequence[Column]) -> str:
    return "  ".join(fmt_cell(str(v), c) for v, c in zip(values, cols))


def header_line(cols: Sequence[Column]) -> str:
    return format_row([c.name for c in cols], cols)


def divider(cols: Sequence[Column]) -> str:
    return "─" * len(header_line(cols))


def print_table(
    cols: Sequence[Column],
    rows: Iterable[Sequence[str]],
    title: str,
    subtitle_lines: Optional[List[str]] = None,
    footer: Optional[str] = None,
) -> None:
    """Print a titled table followed by an optional footer note."""
    rule = divider(cols)
    print()
    print(rule)
    print(title)
    for line in subtitle_lines or []:
        print(line)
    print(rule)
    print(header_line(cols))
    print(rule)
    for row in rows:
        print(format_row(row, cols))
    print(rule)
    if footer:
        print(footer)
    print()


# Formatting helpers for optional metrics. ``None`` renders as "--"; zero is a real value.

def fmt_seconds(v: Optional[float]) -> str:
    return f"{v:.2f}s" if v is not None else "--"


def fmt_tps(v: Optional[float], suffix: str = "") -> str:
    return f"{v:.0f}{suffix}" if v is not None else "--"


def fmt_pct(v: Optional[float]) -> str:
    return f"{v:.1f}%" if v is not None else "--"


def fmt_usd(v: Optional[float], digits: int = 4) -> str:
    return f"${v:.{digits}f}" if v is not None else "--"


def fmt_context(tokens: int) -> str:
    return f"{tokens // 1000}k" if tokens >= 1000 else (str(tokens) if tokens else "--")
