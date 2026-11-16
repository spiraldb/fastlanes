#!/usr/bin/env python3
"""Per-module machine-code size of the `codesize/` harness library, as a markdown table.

Each code symbol counts towards the first `fastlanes::<module>` in its demangled name, or, for
code inlined into a harness wrapper, the `<module>` in the wrapper's `codesize__<module>__...`
name. Other code (`std`, ...) is not counted.

Usage: codesize.py <head.so> [<base.so>]
"""

import re
import subprocess
import sys
from collections import Counter

MODULE = re.compile(r"^codesize__([a-z0-9_]+?)__|\bfastlanes::(\w+)")


def sizes(lib: str) -> Counter:
    nm = subprocess.run(
        ["nm", "--demangle", "--print-size", "--radix=d", lib],
        check=True, capture_output=True, text=True,
    ).stdout
    modules = Counter()
    for line in nm.splitlines():
        parts = line.split(maxsplit=3)  # <addr> <size> <type> <name>
        if len(parts) == 4 and parts[2] in "tTwW" and (m := MODULE.search(parts[3])):
            modules[m.group(1) or m.group(2)] += int(parts[1])
    return modules


def kib(n: int) -> str:
    return f"{n / 1024:.1f} KiB"


def change(head: int, base: int) -> str:
    if head == base:
        return "-"
    pct = f" ({(head - base) / base:+.1%})" if base else ""
    return f"{head - base:+,} B{pct}"


def main() -> None:
    head = sizes(sys.argv[1])
    base = sizes(sys.argv[2]) if len(sys.argv) > 2 else None
    print("## Code size\n")
    if base is None:
        print("| Module | Size |\n|---|---:|")
    else:
        print("| Module | Size | Base | Change |\n|---|---:|---:|---:|")
    for module in sorted(head.keys() | (base or {}).keys(), key=lambda m: -head[m]):
        row = f"| `{module}` | {kib(head[module])} |"
        if base is not None:
            row += f" {kib(base[module])} | {change(head[module], base[module])} |"
        print(row)
    total = f"| **Total** | **{kib(head.total())}** |"
    if base is not None:
        total += f" {kib(base.total())} | {change(head.total(), base.total())} |"
    print(total)
    print(
        "\n<sub>Machine code of every public kernel instantiated for `u8`/`u16`/`u32`/`u64` "
        "(`codesize/` harness, release, `runtime` feature, x86_64).</sub>"
    )


if __name__ == "__main__":
    main()
