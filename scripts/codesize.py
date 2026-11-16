#!/usr/bin/env python3
"""Report the machine-code size of the fastlanes kernels, grouped by module.

Reads the symbols of the `codesize/` harness library with `nm` and sums the size of every code
(text) symbol. The harness exports one wrapper per public kernel and element type, so the
library holds every kernel instantiation and nothing unreachable.

A symbol is attributed to a module either by the first `fastlanes::<module>` path in its
demangled name (`<u8 as fastlanes::bitpacking::BitPacking>::pack::<3, 384>` counts towards
`bitpacking`) or, for a harness wrapper with kernels inlined into it, by the wrapper's
`codesize__<module>__<function>__<type>` name. Everything else (`std`, the allocator, ...) is
the same for every build and is not counted.

Usage:
    codesize.py collect <path/to/libfastlanes_codesize.so> <out.json>
    codesize.py report <head.json> [<base.json>] > report.md
"""

import json
import re
import subprocess
import sys
from collections import defaultdict

MODULE_RE = re.compile(r"\bfastlanes::([A-Za-z_][A-Za-z0-9_]*)")
WRAPPER_RE = re.compile(r"^codesize__([a-z0-9_]+?)__([a-z0-9_]+?)(?:__u(?:8|16|32|64))?$")
TYPE_RE = re.compile(r"\bu(8|16|32|64)\b")
TOP_FUNCTIONS = 25


def strip_turbofish(name: str) -> str:
    """Remove every `::<...>` generic argument list, respecting nested brackets."""
    out = []
    i = 0
    while i < len(name):
        if name.startswith("::<", i):
            depth = 0
            j = i + 2
            while j < len(name):
                if name[j] == "<":
                    depth += 1
                elif name[j] == ">":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            i = j + 1
            continue
        out.append(name[i])
        i += 1
    return "".join(out)


def attribute(name: str) -> tuple[str, str] | None:
    """Return the (module, function) a symbol belongs to, or None if it is not fastlanes code."""
    wrapper = WRAPPER_RE.match(name)
    if wrapper:
        module, function = wrapper.groups()
        return module, f"fastlanes::{module}::{function} (inlined into harness wrapper)"
    match = MODULE_RE.search(name)
    if not match:
        return None
    # Collapse the instantiations of one function: drop const/type generic arguments and the
    # element type, e.g. `<u8 as ..::BitPacking>::pack::<3, 384>` becomes
    # `<T as ..::BitPacking>::pack`.
    return match.group(1), TYPE_RE.sub("T", strip_turbofish(name))


def collect(lib: str, out: str) -> None:
    output = subprocess.run(
        ["nm", "--demangle", "--print-size", "--radix=d", lib],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    modules: dict[str, int] = defaultdict(int)
    functions: dict[str, int] = defaultdict(int)
    symbols: dict[str, int] = {}

    for line in output.splitlines():
        parts = line.split(maxsplit=3)
        # Defined symbols with a size: <addr> <size> <type> <name>
        if len(parts) != 4 or parts[2] not in ("T", "t", "W", "w"):
            continue
        size = int(parts[1])
        name = parts[3]
        if size == 0:
            continue

        attributed = attribute(name)
        if attributed is None:
            continue
        module, function = attributed

        modules[module] += size
        functions[f"{module}\t{function}"] += size
        symbols[name] = symbols.get(name, 0) + size

    with open(out, "w") as f:
        json.dump(
            {
                "total": sum(modules.values()),
                "symbols": len(symbols),
                "modules": dict(modules),
                "functions": dict(functions),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def fmt_bytes(n: int) -> str:
    if abs(n) >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MiB"
    if abs(n) >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def fmt_delta(head: int, base: int | None) -> str:
    if base is None:
        return "new"
    diff = head - base
    if diff == 0:
        return "-"
    pct = f" ({diff / base:+.1%})" if base else ""
    sign = "+" if diff > 0 else "-"
    icon = " :small_red_triangle:" if diff > 0 else " :small_red_triangle_down:"
    return f"{sign}{fmt_bytes(abs(diff))}{pct}{icon}"


def report(head_path: str, base_path: str | None) -> None:
    with open(head_path) as f:
        head = json.load(f)
    base = None
    if base_path:
        with open(base_path) as f:
            base = json.load(f)

    out = ["## Code size report", ""]

    def base_of(kind: str, key: str) -> int | None:
        if base is None:
            return None
        return base[kind].get(key)

    total_line = f"**Total code size:** {fmt_bytes(head['total'])} ({head['total']} bytes)"
    if base is not None:
        change = fmt_delta(head["total"], base["total"])
        total_line += ", no change vs base" if change == "-" else f", {change} vs base"
    out += [total_line, ""]

    # Per-module table, including modules that disappeared relative to base.
    keys = set(head["modules"])
    if base is not None:
        keys |= set(base["modules"])
    rows = sorted(keys, key=lambda k: -head["modules"].get(k, 0))

    header = "| Module | Size | Share |"
    sep = "|---|---:|---:|"
    if base is not None:
        header += " Base | Change |"
        sep += "---:|---:|"
    out += [header, sep]
    for module in rows:
        size = head["modules"].get(module, 0)
        share = f"{size / head['total']:.1%}" if head["total"] else "-"
        row = f"| `{module}` | {fmt_bytes(size)} | {share} |"
        if base is not None:
            b = base_of("modules", module)
            row += f" {fmt_bytes(b) if b is not None else '-'} |"
            row += " removed |" if module not in head["modules"] else f" {fmt_delta(size, b)} |"
        out.append(row)
    out.append("")

    # Largest functions (all instantiations summed).
    functions = sorted(head["functions"].items(), key=lambda kv: -kv[1])[:TOP_FUNCTIONS]
    out += [
        "<details>",
        f"<summary>Largest {len(functions)} functions (all instantiations summed)</summary>",
        "",
        "| Module | Function | Size |" + (" Change |" if base is not None else ""),
        "|---|---|---:|" + ("---:|" if base is not None else ""),
    ]
    for key, size in functions:
        module, function = key.split("\t", 1)
        function = function.replace("|", "\\|")
        row = f"| `{module}` | `{function}` | {fmt_bytes(size)} |"
        if base is not None:
            row += f" {fmt_delta(size, base_of('functions', key))} |"
        out.append(row)
    out += ["", "</details>", ""]

    if base is not None:
        changed = []
        for key in set(head["functions"]) | set(base["functions"]):
            h = head["functions"].get(key, 0)
            b = base["functions"].get(key, 0)
            if h != b:
                changed.append((key, h, b))
        changed.sort(key=lambda c: -abs(c[1] - c[2]))
        if changed:
            out += [
                "<details>",
                f"<summary>Changed functions ({len(changed)})</summary>",
                "",
                "| Module | Function | Base | Head | Change |",
                "|---|---|---:|---:|---:|",
            ]
            for key, h, b in changed[:50]:
                module, function = key.split("\t", 1)
                function = function.replace("|", "\\|")
                change = "removed" if h == 0 else fmt_delta(h, b if b else None)
                out.append(
                    f"| `{module}` | `{function}` | {fmt_bytes(b)} | {fmt_bytes(h)} | {change} |"
                )
            if len(changed) > 50:
                out.append(f"| | ... and {len(changed) - 50} more | | | |")
            out += ["", "</details>", ""]

    out.append(
        "<sub>Code symbols of the `codesize/` harness (every public kernel instantiated for "
        "`u8`/`u16`/`u32`/`u64`), release build, default features, x86_64. "
        "Generated by `scripts/codesize.py`.</sub>"
    )
    print("\n".join(out))


def main() -> None:
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "collect":
        collect(args[1], args[2])
    elif len(args) in (2, 3) and args[0] == "report":
        report(args[1], args[2] if len(args) == 3 else None)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
