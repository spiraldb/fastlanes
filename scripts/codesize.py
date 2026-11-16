#!/usr/bin/env python3
"""Per-module machine-code size of fastlanes, as a markdown table.

Reads a release build of the library's test binary: the tests call every kernel with concrete
types, so the binary holds each kernel's machine code. Each code symbol counts towards the first
`fastlanes::<module>` in its demangled name. Test code (`::test(s)`, `hegel`, the test `main`)
and code from other crates (`std`, ...) are not counted.

Usage: codesize.py <head-binary> [<base-binary>]
"""

import re
import subprocess
import sys
from collections import Counter

MODULE = re.compile(r"\bfastlanes::(\w+)")
TEST_CODE = re.compile(r"::tests?\b|\bhegel|^fastlanes::main$")


def sizes(lib: str) -> Counter:
    nm = subprocess.run(
        ["nm", "--demangle", "--print-size", "--radix=d", lib],
        check=True, capture_output=True, text=True,
    ).stdout
    modules = Counter()
    for line in nm.splitlines():
        parts = line.split(maxsplit=3)  # <addr> <size> <type> <name>
        if len(parts) != 4 or parts[2] not in "tTwW" or TEST_CODE.search(parts[3]):
            continue
        if m := MODULE.search(parts[3]):
            modules[m.group(1)] += int(parts[1])
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
        "\n<sub>Machine code per module in the release test binary, test code excluded "
        "(`runtime` feature, x86_64).</sub>"
    )


if __name__ == "__main__":
    main()
